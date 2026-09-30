"""Native PPO with normal and zero-residual execution during training."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig, concat_hierarchical_batches
from . import pointmaze_joint_renewal as joint
from . import pointmaze_learned_plan as learned
from . import pointmaze_upper_execution as execution
from .pointmaze_goal_validation import _json_ready
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_matched_upper_stage57_spec as spec


def load_source(root, *, preflight):
    path = spec.source_result(root, preflight=preflight)
    c, source = json.loads(path.read_text()), spec.SOURCE_SPEC
    if (c["status"] != "complete" or c["protocol"] != source.EXPERIMENT_PROTOCOL or c["contract"] != source.contract()
            or (c["root"], c["preflight"]) != (root, preflight) or c["options"] != source.options(preflight=preflight)
            or c["seed_roles"] != source.seed_roles(root, preflight=preflight)):
        raise ValueError("Stage57 source differs from frozen Stage55")
    forecaster = path.parent.with_name(path.parent.name + "_raw") / "forecaster.npz"
    with np.load(forecaster) as archive:
        predictor = {k: archive[k] for k in archive.files}
    models, checkpoints = {}, {}
    for period in spec.PERIODS:
        p, checkpoint = str(period), c["checkpoints"][str(period)]["clone"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload["protocol"], payload["root"], payload["period"], payload["policy"]) != (
                source.EXPERIMENT_PROTOCOL, root, period, "clone"):
            raise ValueError("Stage57 requires fixed final clone checkpoint")
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(**payload["state_dict"]["config"]))
        model.load_state_dict(payload["state_dict"])
        if _json_ready(model.config.__dict__) != c["config"] or (model.config.lower_state_dim,
                model.config.lower_value_state_dim, model.config.upper_action_dim, model.config.promotion_state_dim) != (392, 394, 4, 0):
            raise ValueError("Stage57 source learned architecture changed")
        for tensor in (model.upper_actor.net[-1].weight, model.upper_actor.net[-1].bias):
            torch.testing.assert_close(tensor, torch.zeros_like(tensor), atol=0, rtol=0)
        models[p], checkpoints[p] = model, checkpoint
    return models, predictor, {"result": str(path), "checkpoints": checkpoints, "forecaster": str(forecaster),
        "config": c["config"], "upstream_budget_per_root": c["budget"]}


_WORKER = None


def init_worker(config, args):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = FrequencySeparatedActorCriticPPO(config), args


def worker_rollout(job):
    weights, seed, arm, period, phase, mode, predictor, path = job
    model, args = _WORKER
    model.load_state_dict(weights)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    reference = execution.ExecutedPlan(predictor, period, args.maximum_subgoal_delta, spec.execution(arm))
    batch, row, raw = joint.rollout(model, args, f"fixed{period}", seed=seed, capture=True,
        lower_credit="task_option", upper_plan_decoder=reference.decode, lower_reference_builder=reference,
        lower_actor_context_builder=reference.actor_context, lower_value_context_builder=reference.value_context,
        **spec.rollout_arguments(args.optimizer_seed, seed, phase=phase, mode=mode))
    torch.testing.assert_close(joint.inference_weights(model), weights, atol=0, rtol=0)
    raw.update(upper_proposed_action=np.asarray(reference.proposed_actions), upper_plan_action=np.asarray(reference.actions),
               upper_plan_coefficients=np.asarray(reference.coefficients))
    row.update(arm=arm, policy=spec.execution(arm), period=period, phase=phase, deployment_mode=mode,
        policy_seed=spec.policy_seed(args.optimizer_seed, seed), lower_actor_type=type(model.lower_actor).__name__)
    row.update(rollout_network_check="passed", initial_upper_action=reference.proposed_actions[0].tolist(),
        proposed_action_rms=float(np.sqrt(np.square(raw["upper_proposed_action"].astype(np.float64)).mean())),
        executed_action_rms=float(np.sqrt(np.square(raw["upper_plan_action"].astype(np.float64)).mean())),
        executed_plan_delta_squared_sum=reference.executed_delta_squared_sum,
        plan_ols_fits=reference.ols_fits, plan_ridge_predictions=reference.ridge_predictions,
        reference_evaluations=reference.calls, actor_context_evaluations=reference.context_calls,
        upper_plan_decodes=len(reference.actions), bernstein_basis_evaluations=period + 1,
        audit_bernstein_basis_evaluations=period + 1,
        **execution.audit_intervention(raw, row, predictor=predictor, period=period,
                                      scale=args.maximum_subgoal_delta, bounds=reference.bounds))
    if batch is not None:
        np.testing.assert_array_equal(batch.upper.action, raw["upper_proposed_action"])
        np.testing.assert_array_equal(batch.lower.state[:, -2:], raw["lower_actor_context"])
        np.testing.assert_array_equal(batch.lower.value_state[:, :392], batch.lower.state)
        np.testing.assert_array_equal(batch.lower.value_state[:, -2:], raw["lower_value_context"][:, 2:])
        np.testing.assert_array_equal(batch.lower.reward, raw["reward"].astype(np.float32))
    np.savez_compressed(path, **raw)
    return batch, row


def parameter_changes(before, after):
    return {name: float(np.sqrt(sum(float(torch.sum((after[name][k] - before[name][k]) ** 2))
            for k in before[name]))) for name in before}


def update(model, batch, *, arm, phase, root, period, iteration):
    metrics = {}
    levels = ("upper", "lower") if phase == "warmup" or arm == "joint_ppo" else ("lower",)
    for level in levels:
        np.random.seed(spec.shuffle_seed(root, period, iteration, phase=phase, level=level))
        metrics.update(model._update_level(level=level, batch=getattr(batch, level),
            actor=getattr(model, level + "_actor"), value_net=getattr(model, level + "_value"),
            actor_optimizer=getattr(model, level + "_actor_optimizer"), value_optimizer=getattr(model, level + "_value_optimizer"),
            actor_updates_enabled=phase == "train"))
    return {k: int(v) for k, v in metrics.items() if "optimizer_steps" in k}


def train(root, *, preflight, output):
    args, opt = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles, budget = spec.seed_roles(root, preflight=preflight), spec.budget(preflight=preflight)
    clones, predictor, source = load_source(root, preflight=preflight)
    raw, started = raw_directory(output), time.monotonic()
    config = clones[str(spec.PERIODS[0])].config
    counts = {phase: dict.fromkeys(learned.COUNT_KEYS, 0) for phase in ("warmup", "train", "eval")}
    training, calibration, evaluation, checkpoints, pairs = {}, {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
                             initargs=(config, args)) as pool:
        def episodes(model, arm, period, phase, mode, seeds, iteration=0):
            directory = raw / str(period) / arm / phase / str(iteration) / mode
            directory.mkdir(parents=True, exist_ok=True)
            weights = joint.inference_weights(model)
            outputs = list(pool.map(worker_rollout, [(weights, seed, arm, period, phase, mode, predictor,
                str(directory / f"episode_{seed}.npz")) for seed in seeds]))
            rows = [row for _, row in outputs]
            joint.audit_trajectories(rows, args=args, method=f"fixed{period}", raw_path=directory)
            for row in rows:
                counts[phase]["primitive_steps"] += row["episode_length"]
                for key in learned.COUNT_KEYS[1:]:
                    counts[phase][key] += row[key]
            return outputs

        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            initial = joint.inference_weights(clone)
            models = {arm: copy.deepcopy(clone) for arm in spec.TRAIN_POLICIES}
            torch.testing.assert_close(joint.inference_weights(models["zero_train"]),
                                       joint.inference_weights(models["joint_ppo"]), atol=0, rtol=0)
            for level in ("upper", "lower"):
                for kind in ("actor", "value"):
                    name = f"{level}_{kind}_optimizer"
                    torch.testing.assert_close(getattr(models["zero_train"], name).state_dict(),
                                               getattr(models["joint_ppo"], name).state_dict(), atol=0, rtol=0)
            pairs[p] = {"initial_network_and_optimizer": "passed"}
            calibration[p], training[p] = {}, {}
            for arm, model in models.items():
                before, history = joint.inference_weights(model), []
                for iteration in range(1, opt["critic_warmup_iterations"] + 1):
                    start = (iteration - 1) * opt["rollouts_per_iteration"]
                    outputs = episodes(model, arm, period, "warmup", "training",
                        roles["warmup"][start:start + opt["rollouts_per_iteration"]], iteration)
                    metrics = update(model, concat_hierarchical_batches([b for b, _ in outputs]), arm=arm,
                        phase="warmup", root=root, period=period, iteration=iteration)
                    history.append({"iteration": iteration, "rows": [r for _, r in outputs], "optimizer_steps": metrics})
                after = joint.inference_weights(model)
                for name in ("upper_actor", "lower_actor"):
                    torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)
                calibration[p][arm] = {"history": history, "parameter_changes": parameter_changes(before, after)}
            for arm, model in models.items():
                before, history = joint.inference_weights(model), []
                for iteration in range(1, opt["learning_iterations"] + 1):
                    start = (iteration - 1) * opt["rollouts_per_iteration"]
                    outputs = episodes(model, arm, period, "train", "training",
                        roles["training"][start:start + opt["rollouts_per_iteration"]], iteration)
                    metrics = update(model, concat_hierarchical_batches([b for b, _ in outputs]), arm=arm,
                        phase="train", root=root, period=period, iteration=iteration)
                    history.append({"iteration": iteration, "rows": [r for _, r in outputs], "optimizer_steps": metrics})
                after = joint.inference_weights(model)
                if arm == "zero_train":
                    for name in ("upper_actor", "upper_value"):
                        torch.testing.assert_close(after[name], before[name], atol=0, rtol=0)
                training[p][arm] = {"history": history, "parameter_changes": parameter_changes(before, after)}
                print(f"trained {root}/period{period}/{arm}", flush=True)
            for phase, stage in (("warmup", calibration), ("train", training)):
                first = {arm: [r["initial_upper_action"] for r in stage[p][arm]["history"][0]["rows"]] for arm in spec.TRAIN_POLICIES}
                if first["zero_train"] != first["joint_ppo"]:
                    raise ValueError("Stage57 paired first upper proposals differ")
                pairs[p][phase + "_first_upper_proposals"] = "passed"
            torch.testing.assert_close(joint.inference_weights(clone), initial, atol=0, rtol=0)
            models["clone"] = clone
            checkpoints[p], evaluation[p] = {}, {}
            for arm in spec.POLICIES:
                model = models[arm]
                if arm != "clone":
                    checkpoint = raw / str(period) / f"{arm}_final.pt"
                    torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period,
                        "policy": arm, "state_dict": model.state_dict()}, checkpoint)
                    checkpoints[p][arm] = str(checkpoint)
                evaluation[p][arm] = {mode: [row for _, row in episodes(model, arm, period, "eval", mode,
                    roles["evaluation"])] for mode in spec.MODES}
            print(f"evaluated {root}/period{period}/all", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "source": source,
        "config": _json_ready(config.__dict__), "inference_counts": counts, "calibration": calibration, "training": training,
        "evaluation_rows": evaluation, "initial_pairs": pairs, "checkpoints": checkpoints,
        "native_trace_audits": budget["native_trace_audits"], "checkpoint_loads": len(clones), "forecaster_loads": 1,
        "new_forecaster_fits": 0, "supervised_steps": 0, "verification_primitive_steps": 0, "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    return result


def qualify(c, *, preflight):
    root, opt, budget = c["root"], spec.options(preflight=preflight), spec.budget(preflight=preflight)
    roles, horizon = spec.seed_roles(root, preflight=preflight), spec.arguments(root, preflight=preflight).horizon
    if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
            or c["preflight"] != preflight or c["options"] != opt or c["seed_roles"] != roles or c["budget"] != budget
            or c["config"] != c["source"]["config"]
            or any(c[k] != budget[k] for k in ("native_trace_audits", "checkpoint_loads", "forecaster_loads",
                                              "new_forecaster_fits", "supervised_steps", "verification_primitive_steps"))
            or any(set(c[k]) != {str(p) for p in spec.PERIODS} for k in ("calibration", "training", "evaluation_rows", "initial_pairs"))):
        raise ValueError("Stage57 frozen protocol or cost changed")
    observed = {phase: dict.fromkeys(learned.COUNT_KEYS, 0) for phase in ("warmup", "train", "eval")}
    optimizers = dict.fromkeys(("upper_actor", "upper_value", "lower_actor", "lower_value"), 0)
    means = {}
    def rows_check(rows, *, arm, period, phase, mode, seeds):
        if [r["seed"] for r in rows] != seeds:
            raise ValueError("Stage57 paired native path roster changed")
        for row in rows:
            kwargs = spec.rollout_arguments(root, row["seed"], phase=phase, mode=mode)
            fits = horizon // period - 1
            extras = {"plan_ols_fits": fits, "audit_ols_fits": fits, "plan_ridge_predictions": fits,
                "audit_ridge_predictions": fits, "reference_evaluations": horizon, "actor_context_evaluations": horizon,
                "upper_plan_decodes": horizon // period, "bernstein_basis_evaluations": period + 1,
                "audit_bernstein_basis_evaluations": period + 1}
            if (row["arm"] != arm or row["policy"] != spec.execution(arm) or row["period"] != period
                    or row["phase"] != phase or row["deployment_mode"] != mode or row["method"] != f"fixed{period}"
                    or row["episode_length"] != horizon or row["decision_steps"] != list(range(0, horizon, period))
                    or row["upper_inference_calls"] != horizon // period or row["lower_inference_calls"] != horizon
                    or row["gate_inference_calls"] or row["candidate_preview_calls"]
                    or row["policy_seed"] != spec.policy_seed(root, row["seed"])
                    or row["lower_actor_type"] != "GaussianActor" or row["rollout_network_check"] != "passed"
                    or any(row[k] != v for k, v in kwargs.items() if k != "sample")
                    or any(row[k] != v for k, v in extras.items())):
                raise ValueError("Stage57 actual execution or inference accounting changed")
            if arm != "joint_ppo":
                if row["executed_action_rms"] != 0 or row["executed_plan_delta_squared_sum"] != 0:
                    raise ValueError("Stage57 zero execution control changed plan")
            elif row["executed_action_rms"] != row["proposed_action_rms"]:
                raise ValueError("Stage57 normal plan did not execute proposed action")
            if phase != "eval":
                credit = row["lower_training_credit"]
                if (credit["mode"] != "task_option" or credit["primitive_steps"] != horizon
                        or credit["reward_sum"] != credit["task_reward_sum"] or credit["done_count"] != horizon // period):
                    raise ValueError("Stage57 training did not use native option credit")
            observed[phase]["primitive_steps"] += horizon
            for key in learned.COUNT_KEYS[1:]:
                observed[phase][key] += row[key]
    for period in spec.PERIODS:
        p, cfg = str(period), c["config"]
        if (any(v != "passed" for v in c["initial_pairs"][p].values())
                or set(c["initial_pairs"][p]) != {"initial_network_and_optimizer", "warmup_first_upper_proposals", "train_first_upper_proposals"}
                or any(set(c[k][p]) != set(spec.TRAIN_POLICIES) for k in ("calibration", "training"))):
            raise ValueError("Stage57 initial states or training-arm roster differ")
        expected = {"lower": max(1, cfg["epochs"]) * math.ceil(opt["rollouts_per_iteration"] * horizon / cfg["minibatch_size"]),
                    "upper": max(1, cfg["epochs"]) * math.ceil(opt["rollouts_per_iteration"] * (horizon // period) / cfg["minibatch_size"])}
        for phase, name, iterations, role in (("warmup", "calibration", opt["critic_warmup_iterations"], "warmup"),
                                              ("train", "training", opt["learning_iterations"], "training")):
            first = {arm: [r["initial_upper_action"] for r in c[name][p][arm]["history"][0]["rows"]] for arm in spec.TRAIN_POLICIES}
            if first["zero_train"] != first["joint_ppo"]:
                raise ValueError("Stage57 paired first upper proposals differ")
            for arm in spec.TRAIN_POLICIES:
                item, history = c[name][p][arm], c[name][p][arm]["history"]
                if [h["iteration"] for h in history] != list(range(1, iterations + 1)):
                    raise ValueError("Stage57 PPO iteration roster changed")
                changes = item["parameter_changes"]
                active = {"upper_value", "lower_value"} if phase == "warmup" else (
                    {"lower_actor", "lower_value"} if arm == "zero_train" else set(optimizers))
                if any(changes[k] <= 0 if k in active else changes[k] != 0 for k in optimizers):
                    raise ValueError("Stage57 learned or frozen network changes violate protocol")
                for h in history:
                    start = (h["iteration"] - 1) * opt["rollouts_per_iteration"]
                    rows_check(h["rows"], arm=arm, period=period, phase=phase, mode="training",
                               seeds=roles[role][start:start + opt["rollouts_per_iteration"]])
                    for key in optimizers:
                        wanted = expected[key.split("_")[0]] if key in active else 0
                        if h["optimizer_steps"].get(key + "_optimizer_steps", 0) != wanted:
                            raise ValueError("Stage57 PPO optimizer budget changed")
                        optimizers[key] += wanted
        stage = c["evaluation_rows"][p]
        if set(stage) != set(spec.POLICIES) or any(set(modes) != set(spec.MODES) for modes in stage.values()):
            raise ValueError("Stage57 fixed final deployment roster incomplete")
        means[p] = {mode: {} for mode in spec.MODES}
        for arm in spec.POLICIES:
            for mode, rows in stage[arm].items():
                rows_check(rows, arm=arm, period=period, phase="eval", mode=mode, seeds=roles["evaluation"])
                if arm == "joint_ppo" and sum(r["executed_plan_delta_squared_sum"] for r in rows) <= 0:
                    raise ValueError("Stage57 learned upper changed no executed plan")
                means[p][mode][arm] = {k: float(np.mean([r[k] for r in rows])) for k in
                    (*spec.METRICS, "proposed_action_rms", "executed_action_rms", "executed_plan_delta_squared_sum")}
    if observed != c["inference_counts"] or any(observed[phase]["primitive_steps"] != budget["primitive_steps"][phase]
            or observed[phase]["upper_inference_calls"] != budget["upper_calls"][phase] for phase in observed):
        raise ValueError("Stage57 total native budget changed")
    return {"root": root, "means": means, "endpoints": spec.contrasts(means), "parameter_changes": {
        p: {arm: item["parameter_changes"] for arm, item in arms.items()} for p, arms in c["training"].items()}}, observed, optimizers


def bootstrap(rows):
    x = np.asarray([[r["endpoints"][k] for k in spec.ENDPOINTS] for r in rows])
    indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
    tail = .05 / (2 * len(spec.ENDPOINTS))
    bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
    return {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
        "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
        for i, k in enumerate(spec.ENDPOINTS)}


def aggregate(results, *, preflight):
    cells, roots = {c["root"]: c for c in results}, spec.roots(preflight=preflight)
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("Stage57 root roster incomplete")
    rows, totals, optimizers = [], {p: dict.fromkeys(learned.COUNT_KEYS, 0) for p in ("warmup", "train", "eval")}, {}
    for root in roots:
        row, cost, updates = qualify(cells[root], preflight=preflight)
        rows.append(row)
        for phase in totals:
            for key in totals[phase]:
                totals[phase][key] += cost[phase][key]
        for key, value in updates.items():
            optimizers[key] = optimizers.get(key, 0) + value
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "method_cost": totals, "optimizer_steps": optimizers,
        **{k: sum(c[k] for c in results) for k in ("native_trace_audits", "checkpoint_loads", "forecaster_loads",
                                                 "new_forecaster_fits", "supervised_steps", "verification_primitive_steps")}}
    if not preflight:
        endpoints = bootstrap(rows)
        summary.update(primary_endpoints=endpoints,
            matched_upper_gate="passed" if all(endpoints[f"period{p}:joint_ppo_minus_zero_train"]["effect"] == "positive"
                for p in spec.PERIODS) else "failed",
            training_gain_gate="passed" if all(endpoints[f"period{p}:joint_ppo_minus_clone"]["effect"] == "positive"
                for p in spec.PERIODS) else "failed")
    return summary
