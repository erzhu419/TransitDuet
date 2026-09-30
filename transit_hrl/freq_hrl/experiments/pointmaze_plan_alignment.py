"""Isolate native plan anchoring and phase evolution from learning and gates."""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import time

import numpy as np
import torch
from . import pointmaze_joint_renewal as joint
from . import pointmaze_critic_clock as clocks
from .pointmaze_lower_learnability import FeedbackActor, feedback_gain
from .pointmaze_update_direction import load_pair
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_plan_alignment_stage52_spec as spec


def fit_velocity(targets):
    times = (np.arange(len(targets), dtype=np.float64) - len(targets) + 1) * spec.DT_SECONDS
    design = np.column_stack((np.ones(len(targets)), times))
    return np.linalg.lstsq(design, np.asarray(targets, dtype=np.float64), rcond=None)[0][1]


class CausalReference:
    def __init__(self, policy):
        self.policy, self.anchor, self.velocity = policy, None, np.zeros(2)
        self.fits, self.calls, self.velocity_norm_sum = 0, 0, 0.
        self.bounds = None

    def __call__(self, *, observation, history, subgoal, age, step, world_low, world_high):
        self.calls += 1
        self.bounds = (np.asarray(world_low), np.asarray(world_high))
        if self.policy in ("frozen", "waypoint"):
            return subgoal.copy()
        if self.policy == "current_target":
            return observation.task_measurement[:2].astype(np.float32)
        if age == 0:
            self.anchor = observation.task_measurement[:2].astype(np.float32).astype(np.float64)
            count = min(spec.LOOKBACK_STEPS, step + 1)
            if self.policy in ("target_curve", "reverse_curve") and count >= 2:
                targets = history.history.reshape(-1, 6)[-count:, :2]
                self.velocity = fit_velocity(targets)
                self.fits += 1
                self.velocity_norm_sum += float(np.linalg.norm(self.velocity))
            else:
                self.velocity = np.zeros(2)
        direction = -1 if self.policy == "reverse_curve" else 1
        reference = self.anchor + direction * self.velocity * age * spec.DT_SECONDS
        return np.clip(reference, world_low, world_high).astype(np.float32)


def audit_reference(raw, row, *, policy, period, bounds):
    horizon, fits = row["episode_length"], 0
    target = raw["measurement"][:, :2].astype(np.float32)
    expected = np.empty((horizon, 2), dtype=np.float32)
    if policy in ("frozen", "waypoint"):
        expected[:] = raw["subgoal"]
    elif policy == "current_target":
        expected[:] = target
    else:
        for start in range(0, horizon, period):
            stop, velocity = min(start + period, horizon), np.zeros(2)
            if policy in ("target_curve", "reverse_curve") and start:
                begin = max(0, start + 1 - spec.LOOKBACK_STEPS)
                velocity = fit_velocity(target[begin:start + 1])
                fits += 1
            if policy == "reverse_curve":
                velocity = -velocity
            age = np.arange(stop - start, dtype=np.float64)
            # Preserve the executed multiply order, including cancellation near zero.
            expected[start:stop] = np.clip(target[start].astype(np.float64)
                + velocity * age[:, None] * spec.DT_SECONDS, *bounds)
    np.testing.assert_array_equal(raw["lower_reference"], expected,
                                  err_msg="executed reference differs from the causal renewal-only plan")
    return {"audit_regression_fits": fits,
        "reference_target_squared_error_integral": float(np.square(expected.astype(np.float64) - target).sum() * spec.DT_SECONDS)}


def worker_rollout(job):
    weights, seed, policy, period, mode, gain, path = job
    model, args, _, _ = joint._WORKER
    model.load_state_dict(weights)
    torch.manual_seed(spec.policy_seed(args.optimizer_seed, seed))
    reference, original = CausalReference(policy), model.lower_actor
    if policy != "frozen":
        model.lower_actor = FeedbackActor(original, gain, "waypoint")
    try:
        batch, row, raw = joint.rollout(model, args, f"fixed{period}", seed=seed, capture=True,
            lower_credit="task_option", lower_value_context_builder=clocks.context_builder("task_clock"),
            lower_reference_builder=reference, **spec.rollout_arguments(args.optimizer_seed, seed, mode=mode))
    finally:
        model.lower_actor = original
    if batch is not None:
        raise ValueError("Stage52 must not create training batches")
    clocks.audit_context(None, row, raw["lower_value_context"], clock=True)
    row.update(policy_seed=spec.policy_seed(args.optimizer_seed, seed), policy=policy, period=period, deployment_mode=mode,
        plan_regression_fits=reference.fits, reference_evaluations=reference.calls,
        mean_plan_velocity_norm=reference.velocity_norm_sum / reference.fits if reference.fits else 0.,
        **audit_reference(raw, row, policy=policy, period=period, bounds=reference.bounds))
    np.savez_compressed(path, **raw)
    return row


def train(root, *, preflight, output):
    args, opt = spec.arguments(root, preflight=preflight), spec.options(preflight=preflight)
    roles, budget = spec.seed_roles(root, preflight=preflight), spec.budget(preflight=preflight)
    source = json.loads(spec.source_result(root, "task_clock", preflight=preflight).read_text())
    model = load_pair(source, root=root, method="task_clock", preflight=preflight)[0]
    if (model.config.lower_state_dim, model.config.lower_action_dim) != (390, 2):
        raise ValueError("Stage52 requires the registered native lower feature layout")
    raw, started, gain = raw_directory(output), time.monotonic(), feedback_gain()
    keys = ("primitive_steps", "upper_inference_calls", "lower_inference_calls", "gate_inference_calls",
            "plan_regression_fits", "audit_regression_fits", "reference_evaluations")
    counts, evaluation = dict.fromkeys(keys, 0), {}
    weights = joint.inference_weights(model)
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=joint.init_worker,
                             initargs=(model.config, args, "fixed50", "task_option")) as pool:
        for period in spec.PERIODS:
            evaluation[str(period)] = {}
            for policy in spec.POLICIES:
                evaluation[str(period)][policy] = {}
                for mode in spec.MODES:
                    directory = raw / str(period) / policy / mode
                    directory.mkdir(parents=True, exist_ok=True)
                    rows = list(pool.map(worker_rollout, [(weights, seed, policy, period, mode, gain,
                        str(directory / f"episode_{seed}.npz")) for seed in roles["evaluation"]]))
                    joint.audit_trajectories(rows, args=args, method=f"fixed{period}", raw_path=directory)
                    evaluation[str(period)][policy][mode] = rows
                    for row in rows:
                        counts["primitive_steps"] += row["episode_length"]
                        for key in keys[1:]:
                            counts[key] += row[key]
                print(f"evaluated {root}/period{period}/{policy}", flush=True)
    if any(counts[k] != budget[b] for k, b in (("primitive_steps", "total_primitive_steps"),
            ("lower_inference_calls", "total_primitive_steps"), ("upper_inference_calls", "upper_inference_calls"),
            ("plan_regression_fits", "plan_regression_fits"), ("audit_regression_fits", "audit_regression_fits"),
            ("reference_evaluations", "reference_evaluations"))) or counts["gate_inference_calls"]:
        raise ValueError("Stage52 matched-budget accounting changed")
    warm = spec.warmup_iterations(preflight=preflight)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget,
        "inference_counts": counts, "evaluation_rows": evaluation, "native_trace_audits": budget["native_trace_audits"],
        "source_checkpoint": source["snapshots"][str(warm)]["checkpoint"], "source_checkpoint_iteration": warm,
        "computation": {"riccati_solves": 1, "actor_optimizer_steps": 0, "value_optimizer_steps": 0},
        "feedback_gain": gain.tolist(), "wall_seconds": time.monotonic() - started}
    write_json(output, result)
    return result


def bootstrap_endpoints(root_rows):
    x = np.asarray([[r["endpoints"][k] for k in spec.ENDPOINTS] for r in root_rows])
    indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
    tail = .05 / (2 * spec.CI_FAMILY_SIZE)
    bounds = np.quantile(x[indices].mean(axis=1), [tail, 1 - tail], axis=0)
    return {k: {"mean": float(x[:, i].mean()), "ci": bounds[:, i].tolist(),
        "effect": "positive" if bounds[0, i] > 0 else "negative" if bounds[1, i] < 0 else "inconclusive"}
        for i, k in enumerate(spec.ENDPOINTS)}


def aggregate(results, *, preflight):
    roots, opt, budget = spec.roots(preflight=preflight), spec.options(preflight=preflight), spec.budget(preflight=preflight)
    cells = {r["root"]: r for r in results}
    if len(cells) != len(results) or set(cells) != set(roots):
        raise ValueError("Stage52 root roster incomplete")
    root_rows, totals = [], dict.fromkeys(results[0]["inference_counts"], 0)
    for root in roots:
        cell, horizon = cells[root], spec.arguments(root, preflight=preflight).horizon
        roles = spec.seed_roles(root, preflight=preflight)
        if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
                or cell["preflight"] != preflight or cell["options"] != opt or cell["budget"] != budget
                or cell["seed_roles"] != roles or cell["native_trace_audits"] != budget["native_trace_audits"]
                or cell["source_checkpoint_iteration"] != spec.warmup_iterations(preflight=preflight)
                or cell["computation"] != {"riccati_solves": 1, "actor_optimizer_steps": 0, "value_optimizer_steps": 0}
                or set(cell["evaluation_rows"]) != {str(p) for p in spec.PERIODS}):
            raise ValueError("Stage52 result violates frozen protocol")
        observed, means = dict.fromkeys(totals, 0), {}
        for period in spec.PERIODS:
            stage = cell["evaluation_rows"][str(period)]
            if set(stage) != set(spec.POLICIES):
                raise ValueError("Stage52 policy roster incomplete")
            means[str(period)] = {m: {} for m in spec.MODES}
            for policy, modes in stage.items():
                if set(modes) != set(spec.MODES):
                    raise ValueError("Stage52 deployment roster incomplete")
                for mode, rows in modes.items():
                    if [r["seed"] for r in rows] != roles["evaluation"]:
                        raise ValueError("Stage52 paired path roster changed")
                    for row in rows:
                        fits = horizon // period - 1 if policy in ("target_curve", "reverse_curve") else 0
                        kwargs = spec.rollout_arguments(root, row["seed"], mode=mode)
                        if (row["episode_length"] != horizon or row["policy"] != policy or row["period"] != period
                                or row["method"] != f"fixed{period}" or row["deployment_mode"] != mode
                                or row["decision_steps"] != list(range(0, horizon, period))
                                or row["policy_seed"] != spec.policy_seed(root, row["seed"])
                                or row["upper_inference_calls"] != horizon // period or row["lower_inference_calls"] != horizon
                                or row["gate_inference_calls"] != 0 or row["plan_regression_fits"] != fits
                                or row["audit_regression_fits"] != fits or row["reference_evaluations"] != horizon
                                or row["candidate_preview_calls"] != 0
                                or any(row[k] != v for k, v in kwargs.items() if k != "sample")):
                            raise ValueError("Stage52 sampling or equal-call accounting changed")
                        observed["primitive_steps"] += horizon
                        for key in observed:
                            if key != "primitive_steps":
                                observed[key] += row[key]
                    means[str(period)][mode][policy] = {k: float(np.mean([r[k] for r in rows])) for k in spec.METRICS}
        if observed != cell["inference_counts"]:
            raise ValueError("Stage52 total inference accounting changed")
        for key in totals:
            totals[key] += observed[key]
        root_rows.append({"root": root, "means": means, "endpoints": spec.contrasts(means),
            "curve_velocity_norms": {str(p): {k: float(np.mean([r["mean_plan_velocity_norm"] for r in
                cell["evaluation_rows"][str(p)][k]["deterministic"]])) for k in ("target_curve", "reverse_curve")} for p in spec.PERIODS}})
    summary = {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": root_rows, "method_cost": totals,
        "native_trace_audits": sum(r["native_trace_audits"] for r in results), "verification_primitive_steps": 0,
        "computation": {k: sum(r["computation"][k] for r in results) for k in results[0]["computation"]}}
    if not preflight:
        summary["primary_endpoints"] = bootstrap_endpoints(root_rows)
    return summary
