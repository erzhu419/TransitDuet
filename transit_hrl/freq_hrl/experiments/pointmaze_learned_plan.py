"""Native learned residual plans and velocity-conditioned lower policies."""

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl.plan_actions import LearnedPlanActionMapper
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, concat_hierarchical_batches
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from . import pointmaze_forecast_tracking as forecast
from .pointmaze_actor_acceptance import first_batch_pair
from .pointmaze_lower_learnability import feedback_gain
from .pointmaze_update_direction import load_pair
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_learned_plan_stage55_spec as spec


def make_model(source):
    model = FrequencySeparatedActorCriticPPO(replace(source.config, lower_state_dim=392, lower_value_state_dim=394,
        upper_action_dim=4, promotion_state_dim=0))
    weights = joint.inference_weights(source)
    for name in ("upper_actor", "lower_actor", "upper_value", "lower_value"):
        state = copy.deepcopy(weights[name])
        if name == "upper_actor":
            output = f"net.{len(model.upper_actor.net) - 1}"
            state[output + ".weight"] = torch.zeros_like(model.upper_actor.state_dict()[output + ".weight"])
            state[output + ".bias"] = torch.zeros_like(model.upper_actor.state_dict()[output + ".bias"])
            state["log_std"] = state["log_std"].repeat_interleave(2)
        elif name == "lower_actor":
            w = state["net.0.weight"]
            state["net.0.weight"] = torch.cat((w, w.new_zeros((w.shape[0], 2))), dim=1)
        elif name == "lower_value":
            w = state["net.0.weight"]
            state["net.0.weight"] = torch.cat((w[:, :390], w.new_zeros((w.shape[0], 2)), w[:, 390:]), dim=1)
        getattr(model, name).load_state_dict(state)
    return model


class ResidualPlan(forecast.PlanReference):
    def __init__(self, predictor, period, scale):
        super().__init__("ridge_velocity", predictor, period)
        self.mapper = LearnedPlanActionMapper(BernsteinPlanCurve(horizon_s=period * forecast.spec.DT_SECONDS,
            basis_dim=spec.PLAN_BASIS, n_entities=2), coefficient_scale=scale, anchor_first_coefficient=True)
        self.basis = np.array([self.mapper.curve.basis(t * forecast.spec.DT_SECONDS) for t in range(period + 1)])
        self.actions, self.coefficients = [], []
        self.executed_delta_squared_sum = 0.

    def decode(self, *, action, observation, history, step, world_low, world_high):
        self.bounds = np.asarray(world_low), np.asarray(world_high)
        targets = history.history.reshape(-1, 6)[-min(64, step + 1):, :2]
        base = forecast.plan_points(targets, "ridge_velocity", self.predictor, self.period, self.bounds)
        self.base_points = base
        coefficients = self.mapper.residual_coefficients(action)
        self.points = np.clip(base.astype(np.float64) + self.basis @ coefficients.reshape(2, spec.PLAN_BASIS).T,
                              *self.bounds).astype(np.float32)
        self.executed_delta_squared_sum += float(np.square(self.points.astype(np.float64) - base).sum())
        self.actions.append(action.copy())
        self.coefficients.append(coefficients)
        self.ols_fits += int(step > 0)
        self.ridge_predictions += int(step > 0)
        return self.points[0].copy()

    def __call__(self, *, age, **kwargs):
        self.calls += 1
        return self.points[age].copy()

    def value_context(self, *, age, step, horizon):
        velocity = (self.points[age + 1] - self.points[age]) / forecast.spec.DT_SECONDS
        return np.r_[velocity, clocks.time_context(age=age, step=step, horizon=horizon, clock=True)].astype(np.float32)


def audit_plan(raw, row, *, predictor, period, scale, bounds, batch=None):
    expected, velocity = np.zeros_like(raw["lower_reference"]), np.zeros_like(raw["lower_reference"])
    mapper = LearnedPlanActionMapper(BernsteinPlanCurve(horizon_s=period * forecast.spec.DT_SECONDS,
        basis_dim=spec.PLAN_BASIS, n_entities=2), coefficient_scale=scale, anchor_first_coefficient=True)
    basis = np.array([mapper.curve.basis(t * forecast.spec.DT_SECONDS) for t in range(period + 1)])
    coefficients = []
    for index, start in enumerate(row["decision_steps"]):
        target = raw["measurement"][max(0, start + 1 - 64):start + 1, :2].astype(np.float32)
        base = forecast.plan_points(target, "ridge_velocity", predictor, period, bounds)
        coeff = mapper.residual_coefficients(raw["upper_plan_action"][index])
        points = np.clip(base.astype(np.float64) + basis @ coeff.reshape(2, spec.PLAN_BASIS).T, *bounds).astype(np.float32)
        expected[start:start + period] = points[:-1]
        velocity[start:start + period] = np.diff(points, axis=0) / forecast.spec.DT_SECONDS
        coefficients.append(coeff)
    np.testing.assert_array_equal(raw["upper_plan_coefficients"], coefficients)
    np.testing.assert_array_equal(raw["lower_reference"], expected, err_msg="executed upper action did not produce the causal plan")
    np.testing.assert_array_equal(raw["lower_actor_context"], velocity)
    np.testing.assert_array_equal(raw["lower_value_context"][:, :2], velocity)
    clocks.audit_context(None, row, raw["lower_value_context"][:, 2:], clock=True)
    if batch is not None:
        np.testing.assert_array_equal(batch.upper.action, raw["upper_plan_action"])
        np.testing.assert_array_equal(batch.lower.state[:, -2:], velocity)
        np.testing.assert_array_equal(batch.lower.value_state[:, :392], batch.lower.state)
        np.testing.assert_array_equal(batch.lower.value_state[:, -2:], raw["lower_value_context"][:, 2:])
        np.testing.assert_array_equal(batch.lower.reward, raw["reward"].astype(np.float32))
    fits = len(row["decision_steps"]) - 1
    return {"audit_ols_fits": fits, "audit_ridge_predictions": fits,
        "reference_target_squared_error_integral": float(np.square(expected.astype(np.float64) - raw["measurement"][:, :2]).sum() * .01)}


_WORKER = None


def init_worker(source_config, config, args):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = FrequencySeparatedActorCriticPPO(source_config), FrequencySeparatedActorCriticPPO(config), args


def worker_rollout(job):
    weights, seed, policy, period, phase, mode, gain, predictor, path = job
    source, learned, args = _WORKER
    model = source if policy == "frozen" else learned
    model.load_state_dict(weights)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    reference, original = None, model.lower_actor
    if policy == "teacher":
        model.lower_actor = forecast.VelocityFeedbackActor(original, gain, "waypoint")
    kwargs = {}
    if policy != "frozen":
        reference = ResidualPlan(predictor, period, args.maximum_subgoal_delta)
        kwargs = {"upper_plan_decoder": reference.decode, "lower_reference_builder": reference,
                  "lower_actor_context_builder": reference.actor_context, "lower_value_context_builder": reference.value_context}
    else:
        kwargs["lower_value_context_builder"] = clocks.context_builder("task_clock")
    try:
        batch, row, raw = joint.rollout(model, args, f"fixed{period}", seed=seed, capture=True,
            lower_credit="task_option", **kwargs, **spec.rollout_arguments(args.optimizer_seed, seed, phase=phase, mode=mode))
        actor_type = type(model.lower_actor).__name__
    finally:
        model.lower_actor = original
    if reference is not None:
        raw.update(upper_plan_action=np.asarray(reference.actions), upper_plan_coefficients=np.asarray(reference.coefficients))
        row.update(plan_ols_fits=reference.ols_fits, plan_ridge_predictions=reference.ridge_predictions,
            reference_evaluations=reference.calls, actor_context_evaluations=reference.context_calls,
            upper_plan_decodes=len(reference.actions), bernstein_basis_evaluations=period + 1,
            audit_bernstein_basis_evaluations=period + 1,
            upper_plan_action_rms=float(np.sqrt(np.square(raw["upper_plan_action"].astype(np.float64)).mean())),
            executed_plan_delta_squared_sum=reference.executed_delta_squared_sum,
            **audit_plan(raw, row, predictor=predictor, period=period, scale=args.maximum_subgoal_delta,
                         bounds=reference.bounds, batch=batch))
    else:
        clocks.audit_context(None, row, raw["lower_value_context"], clock=True)
        row.update(**dict.fromkeys(EXTRA_COUNTS, 0), upper_plan_action_rms=0., executed_plan_delta_squared_sum=0.,
            reference_target_squared_error_integral=float(
            np.square(raw["subgoal"].astype(np.float64) - raw["measurement"][:, :2]).sum() * .01))
    row.update(policy_seed=spec.policy_seed(args.optimizer_seed, seed), policy=policy, period=period, phase=phase,
               deployment_mode=mode, lower_actor_type=actor_type, upper_action_dim=model.config.upper_action_dim)
    np.savez_compressed(path, **raw)
    return batch, row


EXTRA_COUNTS = ("plan_ols_fits", "audit_ols_fits", "plan_ridge_predictions", "audit_ridge_predictions",
                "reference_evaluations", "actor_context_evaluations", "upper_plan_decodes",
                "bernstein_basis_evaluations", "audit_bernstein_basis_evaluations")
COUNT_KEYS = ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls", *EXTRA_COUNTS)


def clone(model, batch, *, root, period, epochs, sham):
    result = copy.deepcopy(model)
    x = torch.as_tensor(batch.state, dtype=torch.float32)
    y = torch.tanh(torch.as_tensor(batch.action, dtype=torch.float32))
    permutation = np.random.default_rng(np.random.SeedSequence([55, root, period, 55029])).permutation(len(x))
    if sham:
        y = y[torch.as_tensor(permutation)]
    optimizer = torch.optim.Adam(result.lower_actor.net.parameters(), lr=spec.BC_LR)
    rng, steps = np.random.default_rng(spec.shuffle_seed(root, period)), 0
    with torch.no_grad():
        initial = float((result.lower_actor.net(x).tanh() - y).square().mean())
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), spec.BC_MINIBATCH):
            indices = torch.as_tensor(order[start:start + spec.BC_MINIBATCH])
            loss = (result.lower_actor.net(x[indices]).tanh() - y[indices]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
    with torch.no_grad():
        final = float((result.lower_actor.net(x).tanh() - y).square().mean())
    torch.testing.assert_close(result.lower_actor.log_std, model.lower_actor.log_std, atol=0, rtol=0)
    for name in ("upper_actor", "upper_value", "lower_value"):
        torch.testing.assert_close(getattr(result, name).state_dict(), getattr(model, name).state_dict(), atol=0, rtol=0)
    return result, {"steps": steps, "initial_action_mse": initial, "final_action_mse": final, "sham": sham}


def train(root, *, preflight, output):
    args, opt = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles, budget = spec.seed_roles(root, preflight=preflight), spec.budget(preflight=preflight)
    source = json.loads(spec.source_result(root, "task_clock", preflight=preflight).read_text())
    original = load_pair(source, root=root, method="task_clock", preflight=preflight)[0]
    if (original.config.lower_state_dim, original.config.lower_action_dim, original.config.state_encoder) != (390, 2, "mlp"):
        raise ValueError("Stage55 requires the native source MLP")
    template = make_model(original)
    raw, started, gain = raw_directory(output), time.monotonic(), feedback_gain()
    predictor, fitting = forecast.fit_forecaster(args, roles["fitting"])
    np.savez_compressed(raw / "forecaster.npz", **predictor)
    counts = {phase: dict.fromkeys(COUNT_KEYS, 0) for phase in ("labels", "warmup", "train", "eval")}
    evaluation, training, cloning, checkpoints, pairs, label_rows, calibration = {}, {}, {}, {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(original.config, template.config, args)) as pool:
        def episodes(model, policy, period, phase, mode, seeds, iteration=0):
            directory = raw / str(period) / policy / phase / str(iteration) / mode
            directory.mkdir(parents=True, exist_ok=True)
            weights = joint.inference_weights(model)
            outputs = list(pool.map(worker_rollout, [(weights, seed, policy, period, phase, mode, gain, predictor,
                           str(directory / f"episode_{seed}.npz")) for seed in seeds]))
            rows = [row for _, row in outputs]
            joint.audit_trajectories(rows, args=args, method=f"fixed{period}", raw_path=directory)
            for row in rows:
                counts[phase]["primitive_steps"] += row["episode_length"]
                for key in COUNT_KEYS[1:]:
                    counts[phase][key] += row[key]
            return outputs

        for period in spec.PERIODS:
            p = str(period)
            labels = episodes(template, "teacher", period, "labels", "deterministic", roles["labels"])
            label_rows[p] = [row for _, row in labels]
            labelled = concat_hierarchical_batches([batch for batch, _ in labels]).lower
            models, cloning[p] = {"frozen": original, "teacher": template}, {}
            for policy in ("clone", "sham"):
                models[policy], cloning[p][policy] = clone(template, labelled, root=root, period=period,
                    epochs=opt["bc_epochs"], sham=policy == "sham")
            calibrated = copy.deepcopy(models["clone"])
            before_calibration = joint.inference_weights(calibrated)
            warm_history = []
            for iteration in range(1, opt["critic_warmup_iterations"] + 1):
                offset = (iteration - 1) * opt["rollouts_per_iteration"]
                outputs = episodes(calibrated, "clone", period, "warmup", "training",
                    roles["warmup"][offset:offset + opt["rollouts_per_iteration"]], iteration)
                batch = concat_hierarchical_batches([b for b, _ in outputs])
                np.random.seed(spec.shuffle_seed(root, period, opt["learning_iterations"] + iteration))
                metrics = {}
                for level in ("upper", "lower"):
                    metrics.update(calibrated._update_level(level=level, batch=getattr(batch, level),
                        actor=getattr(calibrated, level + "_actor"), value_net=getattr(calibrated, level + "_value"),
                        actor_optimizer=getattr(calibrated, level + "_actor_optimizer"),
                        value_optimizer=getattr(calibrated, level + "_value_optimizer"), actor_updates_enabled=False))
                warm_history.append({"iteration": iteration, "rows": [r for _, r in outputs],
                    "optimizer_steps": {k: int(v) for k, v in metrics.items() if "optimizer_steps" in k}})
            after_calibration = joint.inference_weights(calibrated)
            for name in ("upper_actor", "lower_actor"):
                torch.testing.assert_close(after_calibration[name], before_calibration[name], atol=0, rtol=0)
            calibration[p] = {"history": warm_history, "parameter_changes": {
                name: float(np.sqrt(sum(float(torch.sum((after_calibration[name][k] - before_calibration[name][k]) ** 2))
                    for k in before_calibration[name]))) for name in before_calibration}}
            training[p], first, first_lower_update, pairs[p] = {}, None, None, {}
            for policy in ("lower_ppo", "joint_ppo"):
                model = copy.deepcopy(calibrated)
                before = joint.inference_weights(model)
                history = []
                for iteration in range(1, opt["learning_iterations"] + 1):
                    offset = (iteration - 1) * opt["rollouts_per_iteration"]
                    outputs = episodes(model, policy, period, "train", "training",
                        roles["training"][offset:offset + opt["rollouts_per_iteration"]], iteration)
                    batch = concat_hierarchical_batches([b for b, _ in outputs])
                    if iteration == 1:
                        if policy == "lower_ppo":
                            first = copy.deepcopy(batch)
                        else:
                            pairs[p] = {level: first_batch_pair(getattr(first, level), getattr(batch, level)) for level in ("upper", "lower")}
                    metrics = {}
                    if policy == "joint_ppo":
                        np.random.seed(spec.shuffle_seed(root, period, 5000 + iteration))
                        metrics.update(model._update_level(level="upper", batch=batch.upper, actor=model.upper_actor,
                            value_net=model.upper_value, actor_optimizer=model.upper_actor_optimizer,
                            value_optimizer=model.upper_value_optimizer))
                    np.random.seed(spec.shuffle_seed(root, period, iteration))
                    metrics.update(model._update_level(level="lower", batch=batch.lower, actor=model.lower_actor,
                        value_net=model.lower_value, actor_optimizer=model.lower_actor_optimizer,
                        value_optimizer=model.lower_value_optimizer))
                    if iteration == 1:
                        lower_update = (model.lower_actor.state_dict(), model.lower_value.state_dict(),
                                        model.lower_actor_optimizer.state_dict(), model.lower_value_optimizer.state_dict())
                        if policy == "lower_ppo":
                            first_lower_update = copy.deepcopy(lower_update)
                        else:
                            torch.testing.assert_close(lower_update, first_lower_update, atol=0, rtol=0)
                            pairs[p]["first_lower_update"] = "passed"
                    history.append({"iteration": iteration, "rows": [r for _, r in outputs],
                        "optimizer_steps": {k: int(v) for k, v in metrics.items() if "optimizer_steps" in k}})
                after = joint.inference_weights(model)
                changes = {name: float(np.sqrt(sum(float(torch.sum((after[name][k] - before[name][k]) ** 2))
                            for k in before[name]))) for name in before}
                if policy == "lower_ppo":
                    for name in ("upper_actor", "upper_value"):
                        torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)
                training[p][policy] = {"history": history, "parameter_changes": changes}
                models[policy] = model
                print(f"trained {root}/period{period}/{policy}", flush=True)
            checkpoints[p], evaluation[p] = {}, {policy: {} for policy in spec.POLICIES}
            for policy, model in models.items():
                if policy not in ("frozen", "teacher"):
                    checkpoint = raw / str(period) / f"{policy}_final.pt"
                    torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period,
                        "policy": policy, "state_dict": model.state_dict()}, checkpoint)
                    checkpoints[p][policy] = str(checkpoint)
                for mode in spec.MODES:
                    evaluation[p][policy][mode] = [r for _, r in episodes(model, policy, period, "eval", mode, roles["evaluation"])]
            print(f"evaluated {root}/period{period}/all", flush=True)
    for phase in counts:
        if counts[phase]["primitive_steps"] != budget["primitive_steps"][phase]:
            raise ValueError("Stage55 native training/evaluation budget changed")
    warm = spec.warmup_iterations(preflight=preflight)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "fitting": fitting,
        "inference_counts": counts, "evaluation_rows": evaluation, "training": training, "cloning": cloning, "calibration": calibration,
        "first_batch_pairs": pairs, "label_rows": label_rows, "checkpoints": checkpoints,
        "native_trace_audits": budget["native_trace_audits"], "source_checkpoint": source["snapshots"][str(warm)]["checkpoint"],
        "source_checkpoint_iteration": warm, "source_config": original.config.__dict__, "config": template.config.__dict__,
        "feedback_gain": gain.tolist(), "riccati_solves": 1, "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def bootstrap(root_rows):
    x = np.asarray([[r["endpoints"][k] for k in spec.ENDPOINTS] for r in root_rows])
    indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
    tail = .05 / (2 * len(spec.ENDPOINTS))
    bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
    return {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
        "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
        for i, k in enumerate(spec.ENDPOINTS)}


def aggregate(results, *, preflight):
    roots, opt, budget = spec.roots(preflight=preflight), spec.options(preflight=preflight), spec.budget(preflight=preflight)
    cells = {r["root"]: r for r in results}
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("Stage55 root roster incomplete")
    root_rows = []
    totals = {phase: dict.fromkeys(COUNT_KEYS, 0) for phase in ("labels", "warmup", "train", "eval")}
    optimizers = {k: 0 for k in ("supervised", "upper_actor", "upper_value", "lower_actor", "lower_value")}
    for root in roots:
        c, horizon, roles = cells[root], spec.arguments(root, preflight=preflight).horizon, spec.seed_roles(root, preflight=preflight)
        if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
                or c["preflight"] != preflight or c["options"] != opt or c["seed_roles"] != roles or c["budget"] != budget
                or c["native_trace_audits"] != budget["native_trace_audits"] or c["riccati_solves"] != 1
                or c["source_checkpoint_iteration"] != spec.warmup_iterations(preflight=preflight)
                or c["fitting"]["rows"] != budget["fitting_rows"] or c["fitting"]["native_steps"] != 0
                or c["fitting"]["observations"] != budget["fitting_observations"]
                or set(c["evaluation_rows"]) != {str(p) for p in spec.PERIODS}):
            raise ValueError("Stage55 result violates frozen protocol")
        observed, means = {phase: dict.fromkeys(COUNT_KEYS, 0) for phase in totals}, {}
        expected_config = {**c["source_config"], "lower_state_dim": 392, "lower_value_state_dim": 394,
                           "upper_action_dim": 4, "promotion_state_dim": 0}
        if c["config"] != expected_config:
            raise ValueError("Stage55 unregistered model or PPO hyperparameters")
        def rows_check(rows, *, phase, policy, period, mode, seeds):
            if [r["seed"] for r in rows] != seeds:
                raise ValueError("Stage55 paired native path roster changed")
            for row in rows:
                kwargs = spec.rollout_arguments(root, row["seed"], phase=phase, mode=mode)
                is_plan, fits = policy != "frozen", horizon // period - 1
                extras = {"plan_ols_fits": fits, "audit_ols_fits": fits, "plan_ridge_predictions": fits,
                    "audit_ridge_predictions": fits, "reference_evaluations": horizon, "actor_context_evaluations": horizon,
                    "upper_plan_decodes": horizon // period, "bernstein_basis_evaluations": period + 1,
                    "audit_bernstein_basis_evaluations": period + 1} if is_plan else dict.fromkeys(EXTRA_COUNTS, 0)
                if (row["policy"] != policy or row["period"] != period or row["phase"] != phase or row["deployment_mode"] != mode
                        or row["method"] != f"fixed{period}" or row["episode_length"] != horizon
                        or row["decision_steps"] != list(range(0, horizon, period))
                        or row["upper_inference_calls"] != horizon // period or row["lower_inference_calls"] != horizon
                        or row["gate_inference_calls"] or row["candidate_preview_calls"]
                        or row["policy_seed"] != spec.policy_seed(root, row["seed"])
                        or row["upper_action_dim"] != (4 if is_plan else 2)
                        or row["lower_actor_type"] != ("VelocityFeedbackActor" if policy == "teacher" else "GaussianActor")
                        or any(row[k] != v for k, v in kwargs.items() if k != "sample")
                        or any(row[k] != v for k, v in extras.items())):
                    raise ValueError("Stage55 executed learned policy or causal-call accounting changed")
                observed[phase]["primitive_steps"] += horizon
                for key in COUNT_KEYS[1:]:
                    observed[phase][key] += row[key]
        for period in spec.PERIODS:
            p = str(period)
            rows_check(c["label_rows"][p], phase="labels", policy="teacher", period=period, mode="deterministic", seeds=roles["labels"])
            for policy in ("clone", "sham"):
                bc = c["cloning"][p][policy]
                steps = opt["bc_epochs"] * math.ceil(opt["label_paths"] * horizon / spec.BC_MINIBATCH)
                if bc["steps"] != steps or bc["sham"] != (policy == "sham"):
                    raise ValueError("Stage55 supervised optimizer budget changed")
                optimizers["supervised"] += steps
            if (any(c["first_batch_pairs"][p][level]["status"] != "passed" for level in ("upper", "lower"))
                    or c["first_batch_pairs"][p]["first_lower_update"] != "passed"):
                raise ValueError("Stage55 initial PPO data differ")
            cfg = c["config"]
            expected = {"lower": max(1, cfg["epochs"]) * math.ceil(opt["rollouts_per_iteration"] * horizon / cfg["minibatch_size"]),
                        "upper": max(1, cfg["epochs"]) * math.ceil(opt["rollouts_per_iteration"] * (horizon // period) / cfg["minibatch_size"])}
            warm = c["calibration"][p]
            if ([h["iteration"] for h in warm["history"]] != list(range(1, opt["critic_warmup_iterations"] + 1))
                    or any(warm["parameter_changes"][k] != 0 for k in ("upper_actor", "lower_actor"))
                    or any(warm["parameter_changes"][k] <= 0 for k in ("upper_value", "lower_value"))):
                raise ValueError("Stage55 critic-only calibration was null or changed an actor")
            for h in warm["history"]:
                start = (h["iteration"] - 1) * opt["rollouts_per_iteration"]
                rows_check(h["rows"], phase="warmup", policy="clone", period=period, mode="training",
                    seeds=roles["warmup"][start:start + opt["rollouts_per_iteration"]])
                for level in ("upper", "lower"):
                    if h["optimizer_steps"].get(level + "_actor_optimizer_steps", 0) != 0 or h["optimizer_steps"].get(
                            level + "_value_optimizer_steps", 0) != expected[level]:
                        raise ValueError("Stage55 calibration optimizer budget changed")
                    optimizers[level + "_value"] += expected[level]
            for policy in ("lower_ppo", "joint_ppo"):
                item = c["training"][p][policy]
                if [h["iteration"] for h in item["history"]] != list(range(1, opt["learning_iterations"] + 1)):
                    raise ValueError("Stage55 PPO iteration roster changed")
                changes = item["parameter_changes"]
                if any(changes[k] <= 0 for k in ("lower_actor", "lower_value")) or (policy == "joint_ppo" and
                        any(changes[k] <= 0 for k in ("upper_actor", "upper_value"))) or (policy == "lower_ppo" and
                        any(changes[k] != 0 for k in ("upper_actor", "upper_value"))):
                    raise ValueError("Stage55 registered learned network never changed or frozen upper changed")
                for h in item["history"]:
                    start = (h["iteration"] - 1) * opt["rollouts_per_iteration"]
                    rows_check(h["rows"], phase="train", policy=policy, period=period, mode="training",
                        seeds=roles["training"][start:start + opt["rollouts_per_iteration"]])
                    for level in ("upper", "lower"):
                        for kind in ("actor", "value"):
                            key = f"{level}_{kind}"
                            wanted = expected[level] if level == "lower" or policy == "joint_ppo" else 0
                            if h["optimizer_steps"].get(key + "_optimizer_steps", 0) != wanted:
                                raise ValueError("Stage55 PPO optimizer budget changed")
                            optimizers[key] += wanted
            stage = c["evaluation_rows"][p]
            if set(stage) != set(spec.POLICIES):
                raise ValueError("Stage55 deployment policy roster incomplete")
            means[p] = {mode: {} for mode in spec.MODES}
            for policy in spec.POLICIES:
                if set(stage[policy]) != set(spec.MODES):
                    raise ValueError("Stage55 deployment mode roster incomplete")
                for mode, rows in stage[policy].items():
                    rows_check(rows, phase="eval", policy=policy, period=period, mode=mode, seeds=roles["evaluation"])
                    if policy == "joint_ppo":
                        if sum(r["executed_plan_delta_squared_sum"] for r in rows) <= 0:
                            raise ValueError("Stage55 learned upper changed no executed plan")
                    elif any(r["upper_plan_action_rms"] != 0 or r["executed_plan_delta_squared_sum"] != 0 for r in rows):
                        raise ValueError("Stage55 frozen upper acquired a residual mean")
                    means[p][mode][policy] = {k: float(np.mean([r[k] for r in rows])) for k in spec.METRICS}
                    means[p][mode][policy].update(upper_plan_action_rms=float(np.mean([r["upper_plan_action_rms"] for r in rows])),
                        executed_plan_delta_squared_sum=float(np.mean([r["executed_plan_delta_squared_sum"] for r in rows])))
        if observed != c["inference_counts"]:
            raise ValueError("Stage55 total inference accounting changed")
        for phase in totals:
            if (observed[phase]["primitive_steps"] != budget["primitive_steps"][phase]
                    or observed[phase]["upper_inference_calls"] != budget["upper_calls"][phase]):
                raise ValueError("Stage55 total native budget changed")
            for key in COUNT_KEYS:
                totals[phase][key] += observed[phase][key]
        root_rows.append({"root": root, "means": means, "endpoints": spec.contrasts(means), "cloning": c["cloning"],
            "parameter_changes": {p: {k: v["parameter_changes"] for k, v in stage.items()} for p, stage in c["training"].items()}})
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": root_rows, "method_cost": totals, "optimizer_steps": optimizers,
        "native_trace_audits": sum(c["native_trace_audits"] for c in results), "verification_primitive_steps": 0,
        "fitting_cost": {k: sum(c["fitting"][k] for c in results) for k in
            ("rows", "feature_ols_fits", "driver_paths", "observations", "ridge_solves", "native_steps", "label_displacements")}}
    if not preflight:
        endpoints = bootstrap(root_rows)
        summary["primary_endpoints"] = endpoints
        summary["learnability_gate"] = "passed" if all(endpoints[f"period{p}:clone_minus_{b}"]["effect"] == "positive"
            for p in spec.PERIODS for b in ("frozen", "sham")) else "failed"
        summary["joint_gain_gate"] = "passed" if all(endpoints[f"period{p}:joint_ppo_minus_{b}"]["effect"] == "positive"
            for p in spec.PERIODS for b in ("frozen", "clone", "lower_ppo")) else "failed"
    return summary
