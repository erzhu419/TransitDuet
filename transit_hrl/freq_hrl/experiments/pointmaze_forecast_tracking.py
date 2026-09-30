"""Separate causal plan prediction from position/velocity tracking."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import time

import numpy as np
import torch
from freq_hrl.domains.mujoco.pointmaze_regime import PointMazeRegimeDriver
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from .pointmaze_lower_learnability import FeedbackActor, feedback_gain
from .pointmaze_plan_alignment import fit_velocity
from .pointmaze_update_direction import load_pair
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_forecast_tracking_stage54_spec as spec


def forecast_features(targets):
    target = np.asarray(targets, dtype=np.float64)[-spec.LOOKBACK_STEPS:]
    velocity = np.diff(target, axis=0) / spec.DT_SECONDS
    last, age = velocity[-1], 1
    for v in velocity[-2::-1]:
        if np.linalg.norm(v - last) > spec.MOTION_TOLERANCE:
            break
        age += 1
    age /= spec.LOOKBACK_STEPS - 1
    return np.concatenate((target[-1], last, velocity[-4:].mean(axis=0), velocity[-16:].mean(axis=0),
                           fit_velocity(target), [age], age * last, np.outer(target[-1], last).reshape(-1)))


def fit_forecaster(args, seeds):
    features, labels = [], []
    for seed in seeds:
        driver = PointMazeRegimeDriver(seed=seed, horizon=args.horizon, dt_seconds=spec.DT_SECONDS, **joint._task_options(args))
        target = np.array([driver.sample(t)[0] for t in range(args.horizon + 1)])
        for t in range(1, args.horizon - spec.FORECAST_STEPS + 1):
            features.append(forecast_features(target[max(0, t + 1 - spec.LOOKBACK_STEPS):t + 1]))
            labels.append((target[t + 1:t + 1 + spec.FORECAST_STEPS].astype(np.float64) - target[t]).reshape(-1))
    x, y = np.asarray(features), np.asarray(labels)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.diag([0., *([spec.RIDGE_LAMBDA] * x.shape[1])])
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {"mean": mean, "scale": scale, "weights": weights}, {"rows": len(x), "feature_ols_fits": len(x),
        "driver_paths": len(seeds), "observations": len(seeds) * (args.horizon + 1), "ridge_solves": 1,
        "native_steps": 0, "label_displacements": len(x) * spec.FORECAST_STEPS,
        "training_displacement_mse": float(np.mean((design @ weights - y) ** 2))}


def plan_points(targets, policy, predictor, period, bounds):
    anchor = np.asarray(targets[-1], dtype=np.float32).astype(np.float64)
    if len(targets) < 2 or policy == "target_hold":
        points = np.repeat(anchor[None], period + 1, axis=0)
    elif policy.startswith("linear_"):
        age = np.arange(period + 1, dtype=np.float64)
        points = anchor + fit_velocity(targets) * age[:, None] * spec.DT_SECONDS
    else:
        x = np.r_[1., (forecast_features(targets) - predictor["mean"]) / predictor["scale"]]
        displacement = (x @ predictor["weights"]).reshape(spec.FORECAST_STEPS, 2)
        points = np.vstack((anchor, anchor + displacement[:period]))
    return np.clip(points, *bounds).astype(np.float32)


class PlanReference:
    def __init__(self, policy, predictor, period):
        self.policy, self.predictor, self.period = policy, predictor, period
        self.points, self.bounds = None, None
        self.calls, self.ols_fits, self.ridge_predictions, self.context_calls = 0, 0, 0, 0

    def __call__(self, *, observation, history, subgoal, age, step, world_low, world_high):
        self.calls += 1
        self.bounds = (np.asarray(world_low), np.asarray(world_high))
        if self.policy == "frozen":
            return subgoal.copy()
        if age == 0:
            count = min(spec.LOOKBACK_STEPS, step + 1)
            targets = history.history.reshape(-1, 6)[-count:, :2]
            self.points = plan_points(targets, self.policy, self.predictor, self.period, self.bounds)
            if step and self.policy != "target_hold":
                self.ols_fits += 1
                self.ridge_predictions += int(self.policy.startswith("ridge_"))
        return self.points[age].copy()

    def actor_context(self, *, age, step, horizon):
        self.context_calls += 1
        return (self.points[age + 1] - self.points[age]) / spec.DT_SECONDS


class VelocityFeedbackActor(FeedbackActor):
    def distribution(self, state):
        base, velocity = state[..., :-2], state[..., -2:]
        command = (base[..., 4:6] @ self.gain[:, :2].T - base[..., 2:4] @ self.gain[:, 2:].T
                   - base[..., -4:-2] + velocity @ self.gain[:, 2:].T)
        return torch.distributions.Normal(torch.atanh(command.clamp(-.95, .95)), self.log_std.exp().clamp(1e-4, 3.))


def audit_plan(raw, row, *, policy, period, predictor, bounds):
    horizon = row["episode_length"]
    expected = np.empty((horizon, 2), dtype=np.float32)
    velocity = np.zeros_like(expected)
    fits, predictions = 0, 0
    if policy == "frozen":
        expected[:] = raw["subgoal"]
    else:
        for start in range(0, horizon, period):
            stop = min(start + period, horizon)
            targets = raw["measurement"][max(0, start + 1 - spec.LOOKBACK_STEPS):start + 1, :2]
            points = plan_points(targets, policy, predictor, period, bounds)
            expected[start:stop] = points[:stop - start]
            velocity[start:stop] = (points[1:stop - start + 1] - points[:stop - start]) / spec.DT_SECONDS
            fits += int(start > 0 and policy != "target_hold")
            predictions += int(start > 0 and policy.startswith("ridge_"))
    np.testing.assert_array_equal(raw["lower_reference"], expected, err_msg="causal frozen plan changed")
    if policy.endswith("_velocity"):
        np.testing.assert_array_equal(raw["lower_actor_context"], velocity, err_msg="planned velocity input changed")
    elif "lower_actor_context" in raw:
        raise ValueError("position-only actor received velocity context")
    return {"audit_ols_fits": fits, "audit_ridge_predictions": predictions,
        "reference_target_squared_error_integral": float(np.square(expected.astype(np.float64) - raw["measurement"][:, :2]).sum() * spec.DT_SECONDS)}


def worker_rollout(job):
    weights, seed, policy, period, mode, gain, predictor, path = job
    model, args, _, _ = joint._WORKER
    model.load_state_dict(weights)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    reference, original = PlanReference(policy, predictor, period), model.lower_actor
    if policy != "frozen":
        actor = VelocityFeedbackActor if policy.endswith("_velocity") else FeedbackActor
        model.lower_actor = actor(original, gain, "waypoint")
    try:
        batch, row, raw = joint.rollout(model, args, f"fixed{period}", seed=seed, capture=True,
            lower_credit="task_option", lower_value_context_builder=clocks.context_builder("task_clock"),
            lower_reference_builder=reference,
            lower_actor_context_builder=reference.actor_context if policy.endswith("_velocity") else None,
            **spec.rollout_arguments(args.optimizer_seed, seed, mode=mode))
    finally:
        model.lower_actor = original
    if batch is not None:
        raise ValueError("Stage54 must not create RL training batches")
    clocks.audit_context(None, row, raw["lower_value_context"], clock=True)
    row.update(policy_seed=spec.policy_seed(args.optimizer_seed, seed), policy=policy, period=period, deployment_mode=mode,
        plan_ols_fits=reference.ols_fits, plan_ridge_predictions=reference.ridge_predictions,
        reference_evaluations=reference.calls, actor_context_evaluations=reference.context_calls,
        **audit_plan(raw, row, policy=policy, period=period, predictor=predictor, bounds=reference.bounds))
    np.savez_compressed(path, **raw)
    return row


COUNT_KEYS = ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls",
              "plan_ols_fits", "audit_ols_fits", "plan_ridge_predictions", "audit_ridge_predictions",
              "reference_evaluations", "actor_context_evaluations")


def train(root, *, preflight, output):
    args, opt = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles, budget = spec.seed_roles(root, preflight=preflight), spec.budget(preflight=preflight)
    source = json.loads(spec.source_result(root, "task_clock", preflight=preflight).read_text())
    model = load_pair(source, root=root, method="task_clock", preflight=preflight)[0]
    if (model.config.lower_state_dim, model.config.lower_action_dim) != (390, 2):
        raise ValueError("Stage54 requires the native lower feature layout")
    raw, started, gain = raw_directory(output), time.monotonic(), feedback_gain()
    predictor, fitting = fit_forecaster(args, roles["fitting"])
    np.savez_compressed(raw / "forecaster.npz", **predictor)
    counts, evaluation, weights = dict.fromkeys(COUNT_KEYS, 0), {}, joint.inference_weights(model)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=joint.init_worker,
                             initargs=(model.config, args, "fixed50", "task_option")) as pool:
        for period in spec.PERIODS:
            evaluation[str(period)] = {}
            for policy in spec.POLICIES:
                evaluation[str(period)][policy] = {}
                for mode in spec.MODES:
                    directory = raw / str(period) / policy / mode
                    directory.mkdir(parents=True, exist_ok=True)
                    rows = list(pool.map(worker_rollout, [(weights, seed, policy, period, mode, gain, predictor,
                        str(directory / f"episode_{seed}.npz")) for seed in roles["evaluation"]]))
                    joint.audit_trajectories(rows, args=args, method=f"fixed{period}", raw_path=directory)
                    evaluation[str(period)][policy][mode] = rows
                    for row in rows:
                        counts["primitive_steps"] += row["episode_length"]
                        for key in COUNT_KEYS[1:]:
                            counts[key] += row[key]
                print(f"evaluated {root}/period{period}/{policy}", flush=True)
    warm = spec.warmup_iterations(preflight=preflight)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "fitting": fitting,
        "inference_counts": counts, "evaluation_rows": evaluation, "native_trace_audits": budget["native_trace_audits"],
        "source_checkpoint": source["snapshots"][str(warm)]["checkpoint"], "source_checkpoint_iteration": warm,
        "computation": {"riccati_solves": 1, "actor_optimizer_steps": 0, "value_optimizer_steps": 0},
        "feedback_gain": gain.tolist(), "wall_seconds": time.monotonic() - started}
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
        raise ValueError("Stage54 root roster incomplete")
    root_rows, totals = [], dict.fromkeys(COUNT_KEYS, 0)
    for root in roots:
        cell, horizon = cells[root], spec.arguments(root, preflight=preflight).horizon
        roles = spec.seed_roles(root, preflight=preflight)
        if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
                or cell["preflight"] != preflight or cell["options"] != opt or cell["budget"] != budget
                or cell["seed_roles"] != roles or cell["native_trace_audits"] != budget["native_trace_audits"]
                or cell["source_checkpoint_iteration"] != spec.warmup_iterations(preflight=preflight)
                or cell["computation"] != {"riccati_solves": 1, "actor_optimizer_steps": 0, "value_optimizer_steps": 0}
                or set(cell["evaluation_rows"]) != {str(p) for p in spec.PERIODS}):
            raise ValueError("Stage54 result violates frozen protocol")
        fit = cell["fitting"]
        if (fit["rows"] != budget["fitting_rows"] or fit["feature_ols_fits"] != fit["rows"]
                or fit["driver_paths"] != budget["fitting_driver_paths"] or fit["observations"] != budget["fitting_observations"]
                or fit["ridge_solves"] != 1 or fit["native_steps"] != 0
                or fit["label_displacements"] != fit["rows"] * spec.FORECAST_STEPS):
            raise ValueError("Stage54 disjoint forecast fitting budget changed")
        observed, means = dict.fromkeys(COUNT_KEYS, 0), {}
        for period in spec.PERIODS:
            stage = cell["evaluation_rows"][str(period)]
            if set(stage) != set(spec.POLICIES):
                raise ValueError("Stage54 policy roster incomplete")
            means[str(period)] = {m: {} for m in spec.MODES}
            for policy, modes in stage.items():
                if set(modes) != set(spec.MODES):
                    raise ValueError("Stage54 deployment roster incomplete")
                for mode, rows in modes.items():
                    if [r["seed"] for r in rows] != roles["evaluation"]:
                        raise ValueError("Stage54 paired path roster changed")
                    for row in rows:
                        fits = horizon // period - 1 if policy not in ("frozen", "target_hold") else 0
                        predicts = fits if policy.startswith("ridge_") else 0
                        kwargs = spec.rollout_arguments(root, row["seed"], mode=mode)
                        if (row["episode_length"] != horizon or row["policy"] != policy or row["period"] != period
                                or row["method"] != f"fixed{period}" or row["deployment_mode"] != mode
                                or row["decision_steps"] != list(range(0, horizon, period))
                                or row["policy_seed"] != spec.policy_seed(root, row["seed"])
                                or row["upper_inference_calls"] != horizon // period or row["lower_inference_calls"] != horizon
                                or row["gate_inference_calls"] != 0 or row["plan_ols_fits"] != fits or row["audit_ols_fits"] != fits
                                or row["plan_ridge_predictions"] != predicts or row["audit_ridge_predictions"] != predicts
                                or row["reference_evaluations"] != horizon
                                or row["actor_context_evaluations"] != horizon * int(policy.endswith("_velocity"))
                                or row["candidate_preview_calls"] != 0
                                or any(row[k] != v for k, v in kwargs.items() if k != "sample")):
                            raise ValueError("Stage54 factor or matched-call accounting changed")
                        observed["primitive_steps"] += horizon
                        for key in COUNT_KEYS[1:]:
                            observed[key] += row[key]
                    means[str(period)][mode][policy] = {k: float(np.mean([r[k] for r in rows])) for k in spec.METRICS}
        for key in COUNT_KEYS:
            expected = 0 if key == "gate_inference_calls" else budget["total_primitive_steps" if key in ("primitive_steps", "lower_inference_calls") else key]
            if observed[key] != expected:
                raise ValueError("Stage54 total budget changed")
        if observed != cell["inference_counts"]:
            raise ValueError("Stage54 total inference accounting changed")
        for key in totals:
            totals[key] += observed[key]
        root_rows.append({"root": root, "means": means, "endpoints": spec.contrasts(means), "fitting": fit})
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": root_rows, "method_cost": totals,
        "native_trace_audits": sum(r["native_trace_audits"] for r in results), "verification_primitive_steps": 0,
        "fitting_cost": {k: sum(r["fitting"][k] for r in results) for k in
                         ("rows", "feature_ols_fits", "driver_paths", "observations", "ridge_solves", "native_steps", "label_displacements")},
        "computation": {k: sum(r["computation"][k] for r in results) for k in results[0]["computation"]}}
    if not preflight:
        endpoints = bootstrap(root_rows)
        summary["primary_endpoints"] = endpoints
        summary["development_gate"] = "passed" if all(endpoints[f"period{p}:{k}"]["effect"] == "positive"
            for p in spec.PERIODS for k in spec.EFFECTS[3:]) else "failed"
    return summary
