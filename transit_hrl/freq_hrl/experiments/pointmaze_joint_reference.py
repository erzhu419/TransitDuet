"""Joint learned plan/execution PPO above a frozen strong native controller."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from torch import nn

from freq_hrl.rl.smdp_actor_critic import (FrequencySeparatedActorCriticPPO, SMDPPPOConfig,
    LevelTrajectoryBatch, HierarchicalTrajectoryBatch, concat_hierarchical_batches)
from . import pointmaze_upper_residual_train as base
from . import pointmaze_upper_wide_plan_train as wide
from . import pointmaze_feasible_credit as statistics
from .pointmaze_root_response import write_json
from scripts import pointmaze_joint_reference_stage121_spec as spec

source = base.source
NETWORKS = ("upper_actor", "lower_actor", "upper_value", "lower_value")


class ReferenceResidualActor(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        self.teacher = copy.deepcopy(teacher).requires_grad_(False)
        self.readout = nn.Linear(spec.LOWER_STATE_DIM, 2)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def components(self, state):
        original = self.teacher.distribution(state[..., :396])
        feedback = self.teacher.flat_input(state[..., :396])
        shifted = feedback.clone()
        shifted[..., 4:6] += state[..., 396:398]
        shifted[..., 390:392] += state[..., 398:400]
        response = (self.teacher.base.distribution(shifted).mean
                    - self.teacher.base.distribution(feedback).mean)
        reference = spec.REFERENCE_LIMIT * torch.tanh(response / spec.REFERENCE_LIMIT)
        residual = spec.RESIDUAL_LIMIT * torch.tanh(self.readout(state))
        return torch.distributions.Normal(original.mean + reference + residual, original.stddev), reference, residual

    def distribution(self, state):
        return self.components(state)[0]

    def log_prob_entropy(self, state, action):
        distribution = self.distribution(state)
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)


def make_trainer(model, teacher_state, args):
    torch.manual_seed(args.optimizer_seed + 121003)
    trainer = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
        upper_state_dim=spec.UPPER_STATE_DIM, lower_state_dim=spec.LOWER_STATE_DIM,
        upper_action_dim=spec.UPPER_ACTION_DIM, lower_action_dim=2,
        lower_cost_critic=False, upper_cost_critic=False,
        upper_learning_rate=spec.LEARNING_RATE, lower_learning_rate=spec.LEARNING_RATE,
        gamma=1., gae_lambda=1., epochs=spec.EPOCHS, minibatch_size=spec.MINIBATCH,
        entropy_coef=0., init_log_std=float(np.log(spec.UPPER_STD))))
    trainer.lower_actor = ReferenceResidualActor(base.lower_branch(model, teacher_state))
    trainer.lower_actor_optimizer = torch.optim.Adam(trainer.lower_actor.readout.parameters(), lr=spec.LEARNING_RATE)
    trainer.upper_actor.log_std.requires_grad_(False)
    trainer.upper_actor_optimizer = torch.optim.Adam(trainer.upper_actor.net.parameters(), lr=spec.LEARNING_RATE)
    with torch.no_grad():
        for parameter in trainer.upper_actor.net.parameters():
            parameter.zero_()
        for critic in (trainer.upper_value, trainer.lower_value):
            for parameter in critic.parameters():
                parameter.zero_()
            # Fixed-horizon reward is exp(-distance) <= 1. Remove deterministic
            # time-to-go variation without fitting an evaluation-dependent prior.
            critic.net[0].weight[0, -1] = args.horizon
    return trainer


def weights(model):
    return source.native.joint.inference_weights(model)


def load_weights(model, state):
    for name in NETWORKS:
        getattr(model, name).load_state_dict(state[name])


def clocks(step, period, horizon):
    return np.asarray([(step % period) / period, (horizon - step) / horizon], dtype=np.float32)


def empty_upper():
    return LevelTrajectoryBatch(state=np.empty((0, spec.UPPER_STATE_DIM), dtype=np.float32),
        action=np.empty((0, spec.UPPER_ACTION_DIM), dtype=np.float32), reward=np.empty(0, dtype=np.float32),
        duration=np.empty(0, dtype=np.int64), done=np.empty(0, dtype=np.float32),
        old_logp=np.empty(0, dtype=np.float32), old_value=np.empty(0, dtype=np.float32))


def finish_batch(data, rewards, duration):
    n = len(rewards)
    done = np.zeros(n, dtype=np.float32)
    done[-1] = 1.
    return LevelTrajectoryBatch(**{k: np.asarray(v, dtype=np.float32) for k, v in data.items()},
        reward=np.asarray(rewards, dtype=np.float32), duration=np.full(n, duration, dtype=np.int64), done=done)


def native_episode(trainer, *, args, seed, noise_seed, arm, period, predictor, envelope, collect):
    policy_seed, lower_seed = source.scenario.spec.noise_seeds(args.optimizer_seed, seed, noise_seed)
    plan = (wide.WideBernsteinPlan(predictor, period, args.maximum_subgoal_delta, envelope) if arm == "joint"
        else source.baseline.forecast.PlanReference("ridge_velocity", predictor, period) if arm == "forecast" else None)
    task = source.native.joint._make_task(env_id=args.env_id, seed=seed, horizon=args.horizon,
                                        **source.native.joint._task_options(args))
    data = {level: {k: [] for k in ("state", "action", "old_logp", "old_value")} for level in ("upper", "lower")}
    rewards, distances, innovations, measurements, decisions = [], [], [], [], []
    reference_square = residual_square = delta_square = upper_mean_square = 0.
    reference_peak = residual_peak = 0.
    try:
        obs = task.reset()
        low, high = source.native.joint.pointmaze_goal_bounds(task.environment)
        scale = source.native.joint.scale_for(args)
        history = source.native.joint.PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(obs)
        for step in range(args.horizon):
            feedback = source.baseline.flat_state(history, obs)
            clock = clocks(step, period, args.horizon)
            measurements.append(obs.task_measurement.copy())
            if arm == "joint" and step % period == 0:
                state = np.r_[history.upper_state(obs, oracle_context=None), clock].astype(np.float32)
                with torch.inference_mode():
                    torch.manual_seed(policy_seed + step)
                    distribution = trainer.upper_actor.distribution(torch.as_tensor(state).view(1, -1))
                    raw = distribution.sample()[0] if collect else distribution.mean[0]
                    logp = distribution.log_prob(raw).sum().item()
                    value = trainer.upper_value(torch.as_tensor(state).view(1, -1)).item()
                plan.decode(action=raw.numpy(), observation=obs, history=history, step=step, world_low=low, world_high=high)
                upper_mean_square += float(distribution.mean.double().square().sum())
                decisions.append(step)
                if collect:
                    for k, v in (("state", state), ("action", raw.numpy()), ("old_logp", logp), ("old_value", value)):
                        data["upper"][k].append(v)
            delta = np.zeros(4, dtype=np.float32)
            if plan is None:
                teacher_state = np.r_[feedback, np.zeros(4, dtype=np.float32)]
            else:
                age = step % period
                reference = plan(observation=obs, history=history, subgoal=None, age=age,
                    step=step, world_low=low, world_high=high)
                velocity = plan.actor_context(age=age, step=step, horizon=args.horizon)
                if arm == "joint":
                    forecast_reference = plan.base_points[age]
                    forecast_velocity = (plan.base_points[age + 1] - plan.base_points[age]) / scale.dt_seconds
                    delta = np.r_[reference - forecast_reference, velocity - forecast_velocity].astype(np.float32)
                else:
                    forecast_reference, forecast_velocity = reference, velocity
                teacher_state = source.advice_state(feedback, obs, forecast_reference, forecast_velocity)
            state = np.r_[teacher_state, delta, clock].astype(np.float32)
            with torch.inference_mode():
                torch.manual_seed(lower_seed + step)
                distribution, correction, residual = trainer.lower_actor.components(torch.as_tensor(state).view(1, -1))
                raw = distribution.sample()[0]
                logp = distribution.log_prob(raw).sum().item()
                value = trainer.lower_value(torch.as_tensor(state).view(1, -1)).item() if collect else 0.
                innovations.append(((raw - distribution.mean[0]) / distribution.stddev[0]).numpy())
                reference_square += float(correction.double().square().sum())
                residual_square += float(residual.double().square().sum())
                reference_peak = max(reference_peak, float(correction.abs().max()))
                residual_peak = max(residual_peak, float(residual.abs().max()))
            delta_square += float(np.square(delta.astype(np.float64)).sum())
            command = source.native.joint.squash_box_action(raw.numpy(), task.action_low, task.action_high)
            obs, reward, terminated, truncated, info = task.step(command)
            if bool(terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("Stage121 native path ended before fixed horizon")
            if collect:
                for k, v in (("state", state), ("action", raw.numpy()), ("old_logp", logp), ("old_value", value)):
                    data["lower"][k].append(v)
            rewards.append(float(reward)); distances.append(float(info["tracking_distance"]))
            history.update(obs)
        batch = None
        if collect:
            lower = finish_batch(data["lower"], rewards, 1)
            upper = (finish_batch(data["upper"], [sum(rewards[s:s + period]) for s in decisions], period)
                     if decisions else empty_upper())
            batch = HierarchicalTrajectoryBatch(upper=upper, lower=lower)
            for level in (lower, upper):
                if level.size:
                    _, returns = trainer._gae(level.reward, level.done, level.duration, level.old_value)
                    np.testing.assert_allclose(returns[0], sum(rewards), atol=.002, rtol=0)
        row = {"seed": seed, "noise_seed": noise_seed, "state_arm": arm,
            "episode_return": float(sum(rewards)), "episode_length": args.horizon,
            "lower_calls": args.horizon, "upper_calls": len(decisions), "decision_steps": decisions,
            "plan_renewals": args.horizon // period if plan is not None else 0,
            "plan_fits": plan.ols_fits if plan is not None else 0,
            "reference_calls": plan.calls if plan is not None else 0,
            "reference_donor_calls": 2 * args.horizon, "lower_sample": True, "upper_sample": collect and arm == "joint",
            "reference_correction_rms": float(np.sqrt(reference_square / (2 * args.horizon))),
            "reference_correction_peak": reference_peak,
            "learned_residual_rms": float(np.sqrt(residual_square / (2 * args.horizon))),
            "learned_residual_peak": residual_peak,
            "plan_delta_rms": float(np.sqrt(delta_square / (4 * args.horizon))),
            "upper_mean_rms": float(np.sqrt(upper_mean_square / (8 * len(decisions)))) if decisions else 0.,
            "tracking_squared_error_integral": float(np.dot(distances, distances) * scale.dt_seconds)}
        return batch, row, {"innovations": np.asarray(innovations), "measurements": np.asarray(measurements)}
    finally:
        task.environment.close()


def worker_group(job):
    source_state, teacher_state, states, seed, noises, period, predictor, envelope, collect = job
    source_model, args = source.native._WORKER
    source_model.load_state_dict(source_state)
    trainer = make_trainer(source_model, teacher_state, args)
    outputs, common = [], {}
    variants = list(spec.METHODS) if collect else list(spec.ARMS)
    mapping = {"source_flat": ("initial", "flat"), "source_forecast": ("initial", "forecast"),
        "joint_blinded": ("joint", "forecast"), "upper_transfer": ("forecast", "joint")}
    for noise in noises:
        reference_audit = None
        for variant in variants:
            weights_key, arm = mapping.get(variant, (variant, variant))
            state = states[weights_key]
            if variant == "upper_transfer":
                state = {**state, "upper_actor": states["joint"]["upper_actor"]}
            load_weights(trainer, state)
            batch, row, audit = native_episode(trainer, args=args, seed=seed, noise_seed=noise,
                arm=arm, period=period, predictor=predictor, envelope=envelope, collect=collect)
            row["variant"] = variant
            if reference_audit is None:
                reference_audit = audit
            else:
                np.testing.assert_array_equal(audit["measurements"], reference_audit["measurements"])
                np.testing.assert_allclose(audit["innovations"], reference_audit["innovations"], atol=3e-5, rtol=0)
            outputs.append((variant, batch, row))
        common[noise] = "passed"
    return {"outputs": outputs, "pairing": common}


def parameter_delta(before, after):
    changed = torch.cat([(after[k] - before[k]).detach().reshape(-1) for k in before])
    return float(changed.double().square().mean().sqrt())


def update(trainer, batches, *, optimizer_seed):
    batch = concat_hierarchical_batches(batches)
    replay_error = 0.
    for level, actor in ((batch.upper, trainer.upper_actor), (batch.lower, trainer.lower_actor)):
        for start in range(0, level.size, spec.MINIBATCH):
            stop = start + spec.MINIBATCH
            with torch.no_grad():
                logp, _ = actor.log_prob_entropy(torch.as_tensor(level.state[start:stop]),
                                                torch.as_tensor(level.action[start:stop]))
            replay_error = max(replay_error, float(np.abs(logp.numpy() - level.old_logp[start:stop]).max()))
    if replay_error > 3e-5:
        raise ValueError("Stage121 execution and PPO likelihood disagree")
    before = weights(trainer)
    np.random.seed(optimizer_seed)
    torch.manual_seed(optimizer_seed)
    metrics = trainer.update(batch)
    deltas = {name: parameter_delta(before[name], getattr(trainer, name).state_dict()) for name in NETWORKS}
    return {"optimizer_seed": optimizer_seed, "old_logp_replay_max_error": replay_error, "parameter_delta_rms": deltas,
        "optimizer_steps": {k: int(v) for k, v in metrics.items() if k.endswith("optimizer_steps")},
        "losses": {k: float(metrics[k]) for k in ("upper_policy_loss", "lower_policy_loss", "upper_value_loss", "lower_value_loss")}}


def count_row(cost, row, *, collect):
    cost["training_episodes" if collect else "evaluation_episodes"] += 1
    cost["native_episodes"] += 1
    for key, row_key in (("native_steps", "episode_length"), ("native_lower_calls", "lower_calls"),
            ("native_upper_calls", "upper_calls"), ("native_donor_response_calls", "reference_donor_calls"),
            ("planning_renewals", "plan_renewals"), ("planning_fits", "plan_fits"),
            ("planning_reference_calls", "reference_calls")):
        cost[key] += row[row_key]
    cost["credit_checks"] += int(collect)


def run(root, *, preflight, output):
    output = Path(output)
    args, roles, options = spec.arguments(root, preflight=preflight), spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    source.qualify(json.loads(spec.source_result(root).read_text()), preflight=False)
    models, predictor, _, calibrations = source.load_source(root)
    cost = dict.fromkeys(spec.budget(preflight=preflight), 0)
    cost.update(source_cell_loads=1, source_clone_loads=len(spec.PERIODS), lower_checkpoint_loads=len(spec.PERIODS))
    groups = {}
    with ProcessPoolExecutor(max_workers=options["workers"], mp_context=mp.get_context("spawn"),
            initializer=source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            source_model = models[str(period)]
            source_snapshot = copy.deepcopy(source_model.state_dict())
            teacher_state = base.load_lower_state(root, period, protocol=spec)
            trainers = {method: make_trainer(source_model, teacher_state, args) for method in spec.METHODS}
            initial = weights(trainers["forecast"])
            source_state = weights(source_model)
            history = {method: [] for method in spec.METHODS}
            for iteration, roster in enumerate(roles["training_rounds"], 1):
                states = {method: weights(trainer) for method, trainer in trainers.items()}
                jobs = [(source_state, teacher_state, states, r["scenario_seed"], r["noise_seeds"], period,
                    predictor, calibrations[str(period)]["envelope"], True) for r in roster]
                batches = {method: [] for method in spec.METHODS}
                returns = {method: [] for method in spec.METHODS}
                for registered, group in zip(roster, pool.map(worker_group, jobs)):
                    if list(group["pairing"]) != registered["noise_seeds"]:
                        raise ValueError("Stage121 training noise roster changed")
                    for method, batch, row in group["outputs"]:
                        if row["seed"] != registered["scenario_seed"]:
                            raise ValueError("Stage121 training scenario changed")
                        batches[method].append(batch); returns[method].append(row["episode_return"])
                        count_row(cost, row, collect=True)
                for method, trainer in trainers.items():
                    report = update(trainer, batches[method], optimizer_seed=spec.optimizer_seed(root, period, iteration))
                    for key in spec.budget(preflight=preflight):
                        if key.endswith("optimizer_steps"):
                            cost[key] += report["optimizer_steps"][key]
                    cost["ppo_updates"] += 1
                    history[method].append({"update": iteration, "episodes": len(batches[method]),
                        "mean_training_return": float(np.mean(returns[method])), **report})
                print(f"root={root} period={period} update={iteration}/{options['updates']} joint_return={np.mean(returns['joint']):.6f}", flush=True)
            final_states = {method: weights(trainer) for method, trainer in trainers.items()}
            final_states["initial"] = initial
            evaluation = {variant: [] for variant in spec.ARMS}
            jobs = [(source_state, teacher_state, final_states, seed, [seed], period, predictor,
                calibrations[str(period)]["envelope"], False) for seed in roles["native_evaluation"]]
            for seed, group in zip(roles["native_evaluation"], pool.map(worker_group, jobs)):
                for variant, _, row in group["outputs"]:
                    if row["seed"] != seed or row["noise_seed"] != seed:
                        raise ValueError("Stage121 evaluation roster changed")
                    evaluation[variant].append(row); count_row(cost, row, collect=False)
                cost["evaluation_pair_groups"] += 1
            for method, trainer in trainers.items():
                torch.testing.assert_close(trainer.lower_actor.teacher.state_dict(), teacher_state, atol=0, rtol=0)
                torch.testing.assert_close(trainer.upper_actor.log_std, torch.full_like(trainer.upper_actor.log_std, np.log(spec.UPPER_STD)), atol=0, rtol=0)
                if not preflight:
                    path = output.parent / "final_weights" / f"period_{period}_{method}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period,
                        "method": method, "updates": options["updates"], "weights": final_states[method]}, path)
                    cost["checkpoint_writes"] += 1
            source.native.curves.support.assert_frozen(source_model, source_snapshot)
            groups[str(period)] = {"history": history, "evaluation": evaluation,
                "effects": base.paired_effects(period, evaluation, roles["native_evaluation"], protocol=spec),
                "source_and_teacher_unchanged": "passed", "final_update": options["updates"]}
    cell = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "root": root, "preflight": preflight, "seed_roles": roles, "cost": cost,
        "groups": groups, "native_trace_writes": 0}
    qualify(cell, preflight=preflight)
    write_json(output, cell)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL})
    return cell


def qualify(cell, *, preflight):
    root = cell["root"]
    if (cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["status"] != "complete"
            or cell["contract"] != spec.contract() or cell["preflight"] != preflight
            or root not in spec.roots(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight) or cell["native_trace_writes"] != 0
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage121 protocol, roster or native/optimizer budget changed")
    o, h = spec.options(preflight=preflight), spec.arguments(root, preflight=preflight).horizon
    for period, group in cell["groups"].items():
        if group["source_and_teacher_unchanged"] != "passed" or group["final_update"] != o["updates"]:
            raise ValueError("Stage121 source freeze or final update changed")
        if set(group["history"]) != set(spec.METHODS):
            raise ValueError("Stage121 training methods changed")
        effects = base.paired_effects(int(period), group["evaluation"], cell["seed_roles"]["native_evaluation"], protocol=spec)
        if group["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Stage121 evaluation effects changed")
        for method, history in group["history"].items():
            if len(history) != o["updates"] or [r["update"] for r in history] != list(range(1, o["updates"] + 1)):
                raise ValueError("Stage121 PPO update sequence changed")
            for report in history:
                if report["old_logp_replay_max_error"] > 3e-5 or report["episodes"] != o["scenarios_per_update"] * o["rollouts_per_scenario"]:
                    raise ValueError("Stage121 PPO likelihood or path budget changed")
                if report["optimizer_seed"] != spec.optimizer_seed(root, int(period), report["update"]):
                    raise ValueError("Stage121 PPO optimizer seed changed")
                if any(not np.isfinite(report["parameter_delta_rms"][n]) for n in NETWORKS) or any(not np.isfinite(v) for v in report["losses"].values()):
                    raise ValueError("Stage121 nonfinite training")
                if method != "joint" and any(report["parameter_delta_rms"][n] != 0 for n in ("upper_actor", "upper_value")):
                    raise ValueError("Stage121 baseline updated upper")
            active = NETWORKS if method == "joint" else ("lower_actor", "lower_value")
            if any(sum(r["parameter_delta_rms"][n] for r in history) <= 0 for n in active):
                raise ValueError("Stage121 active actor or critic never changed")
        for variant, rows in group["evaluation"].items():
            arm = "flat" if variant in ("source_flat", "flat") else "joint" if variant in ("joint", "upper_transfer") else "forecast"
            for row in rows:
                expected = {"variant": variant, "state_arm": arm, "episode_length": h, "lower_calls": h,
                    "upper_calls": h // int(period) if arm == "joint" else 0,
                    "decision_steps": list(range(0, h, int(period))) if arm == "joint" else [],
                    "reference_donor_calls": 2 * h, "upper_sample": False, "lower_sample": True,
                    "plan_renewals": h // int(period) if arm != "flat" else 0,
                    "plan_fits": h // int(period) - 1 if arm != "flat" else 0,
                    "reference_calls": h if arm != "flat" else 0, "noise_seed": row["seed"]}
                if any(row[k] != v for k, v in expected.items()):
                    raise ValueError("Stage121 deployed policy schedule changed")
                if row["reference_correction_peak"] > spec.REFERENCE_LIMIT + 1e-8 or row["learned_residual_peak"] > spec.RESIDUAL_LIMIT + 1e-8:
                    raise ValueError("Stage121 bounded execution correction exceeded")
    return cell


def aggregate(cells, *, preflight):
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(p): all(result["endpoints"][f"{p}/joint_minus_{baseline}"]["ci"][0] >
        (spec.MINIMUM_GAIN if baseline in ("flat", "forecast") else 0.)
        for baseline in ("flat", "forecast", "joint_blinded", "source_forecast")) for p in spec.PERIODS}
    result.update(joint_gain_gate="mechanical_only" if preflight else "supported_both_periods" if all(passed.values()) else "not_closed",
        period_joint_gain_gate=passed, performance_claim="new_joint_PPO_reference_control_development",
        stage120_gate="inconclusive_no_automatic_seed_extension_unchanged")
    return result
