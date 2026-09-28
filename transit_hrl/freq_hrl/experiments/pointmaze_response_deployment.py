"""Deploy frozen forecast-conditioned renewal decisions on their own trajectories."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from freq_hrl.core.causal_motion import CausalMotionForecaster
from freq_hrl.core.plan_response import PlanResponseCritic
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from .pointmaze_goal_validation import _json_ready, pointmaze_goal_bounds, squash_box_action
from .pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder, _make_task
from .pointmaze_plan_validity_branching import _task_options
from .pointmaze_temporal_plan import HISTORY, causal_frame, padded_history
from . import pointmaze_forecast_response as response
from . import pointmaze_plan_hold as hold
from . import pointmaze_root_response as source_run
from . import pointmaze_separate_motion as motion
from scripts import pointmaze_root_response_stage33_spec as source_spec
from scripts import pointmaze_response_deployment_stage34_spec as spec


def load_frozen(root, *, preflight):
    args = source_spec.arguments(root, preflight=preflight)
    source = json.loads(spec.source_result(root, preflight=preflight).read_text())
    if (source["status"] != "complete" or source["protocol"]["phase"] != "response"
            or source["protocol"]["protocol_version"] != source_spec.EXPERIMENT_PROTOCOL
            or source["protocol"]["optimizer_seed"] != root
            or source["protocol"]["options"] != _json_ready(source_spec.options(root, preflight=preflight))):
        raise ValueError("deployment source differs from the frozen Stage-33 response")
    if not preflight:
        qualified = json.loads((spec.ROOT / "results" / spec.SOURCE_RUN / "root_qualification.json").read_text())
        if not qualified["qualification_passed"] or qualified["optimizer_roots"] != list(spec.OPTIMIZER_ROOTS):
            raise ValueError("full deployment requires the complete qualified Stage-33 roster")
    cell = source["cells"][0]
    controller, trained, factual = source_run.load_controller(args, Path(cell["controller_result"]))
    if trained["selected_checkpoint_iteration"] != cell["controller_selected_iteration"]:
        raise ValueError("response and controller checkpoints differ")
    paths = spec.evaluation_paths(root, preflight=preflight)
    hold.validate_paths(args, {"fit": [], "evaluation": paths},
                        {"temporal_seed_roles": source_spec.seed_roles(root, preflight=preflight)})
    raw = Path(cell["raw_server_directory"])
    models, critics = {}, {}
    for method, fitted in json.loads((raw / "motion_fits.json").read_text()).items():
        model = CausalMotionForecaster(observed_dim=6, velocity_channels=(0, 1),
            horizon_steps=motion.HORIZONS, dt_seconds=.01)
        model.fitted = {k: np.asarray(v) if k in ("feature_mean", "feature_scale", "weights") else v
                        for k, v in fitted.items()}
        models[method] = model
    for method, fitted in json.loads((raw / "response_fits.json").read_text()).items():
        critic = PlanResponseCritic(durations_seconds=np.asarray(hold.HORIZONS) * .01)
        critic.fitted = {k: np.asarray(v) if k in ("feature_mean", "feature_scale", "weights") else v
                         for k, v in fitted.items()}
        critics[method] = critic
    if set(models) != set(motion.METHODS) or set(critics) != set(response.METHODS):
        raise ValueError("deployment needs all original frozen forecast/response views")
    return args, controller, models, critics, factual, cell["controller_selected_iteration"]


def rollout(controller, models, critics, *, args, seed, method):
    scale = source_run.scale_for(args)
    task = _make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **_task_options(args))
    started = time.perf_counter()
    try:
        observation = task.reset()
        low, high = pointmaze_goal_bounds(task.environment)
        adapter = RelativeSubgoalAdapter(maximum_delta=np.full(2, args.maximum_subgoal_delta, dtype=np.float32),
                                        world_low=low, world_high=high)
        history = PointMazeRegimeFeatureBuilder(time_scale=scale, task_dim=len(observation.task_measurement))
        history.reset(observation)
        controller.reset_recurrent_inference()
        subgoal, previous_action = observation.achieved_goal.copy(), np.zeros_like(task.action_low)
        frames, last_plan, delayed = deque(maxlen=HISTORY), -1, None
        checks = spec.checks(args.horizon)
        end = checks[-1] + hold.SETTLEMENT_STEPS
        shared = {s for s in source_run.schedule_for(args, seed) if s < checks[0]}
        shared.update(range(end, args.horizon, 50))
        if method == "fixed50":
            shared = set(range(0, args.horizon, 50))
        calls, executed, decisions, sequences = [], [], [], []
        costs, rewards, targets, positions, goals, actions = [], [], [], [], [], []
        physical, achieved, measurements = [], [], []
        upper_seconds = lower_seconds = response_seconds = 0.

        def propose(kind):
            nonlocal upper_seconds
            before = time.perf_counter()
            output = controller.plan_goal(history.upper_state(observation, oracle_context=None), sample=False)
            goal = adapter.decode(np.asarray(output["action"], dtype=np.float32), observation.achieved_goal)
            upper_seconds += time.perf_counter() - before
            calls.append({"step": step, "kind": kind})
            return goal

        for step in range(args.horizon):
            names, frame = causal_frame(observation, subgoal, previous_action, step=step,
                                       last_plan=last_plan, horizon=args.horizon)
            frames.append(frame)
            physical.append(observation.physical.copy())
            achieved.append(observation.achieved_goal.copy())
            measurements.append(observation.task_measurement.copy())
            if step in checks:
                sequence = padded_history(frames)
                sequences.append(sequence)
                rate, candidate, renew = None, None, None
                if method in response.METHODS:
                    candidate = propose("candidate_preview")
                    before = time.perf_counter()
                    row = {"seed": seed, "check_step": step, "sequence": sequence,
                           "feature_names": names, "candidate_plan": candidate}
                    design = response.design([row], models, method=method, root=args.optimizer_seed)
                    rate = float(critics[method].predict_rates(design)[0, -1])
                    response_seconds += time.perf_counter() - before
                    if not np.isfinite(rate):
                        raise RuntimeError("non-finite deployed response rate")
                    renew = rate > 0
                elif method != "fixed50":
                    renew = method == "always_renew"
                decisions.append({"step": step, "renew": renew, "settled_rate": rate,
                                  "candidate_plan": candidate})
                if method != "fixed50":
                    if renew:
                        subgoal = candidate if candidate is not None else propose("immediate_plan")
                        last_plan = step
                        executed.append(step)
                    else:
                        delayed = step + hold.HOLD_STEPS
            if step in shared or step == delayed:
                subgoal = propose("delayed_plan" if step == delayed else "shared_plan")
                last_plan = step
                executed.append(step)
                if step == delayed:
                    delayed = None
            before = time.perf_counter()
            output = controller.act_conditioned(history.lower_state(observation, subgoal=subgoal), sample=False)
            action = squash_box_action(np.asarray(output["action"], dtype=np.float32), task.action_low, task.action_high)
            lower_seconds += time.perf_counter() - before
            target = observation.target.copy()
            goals.append(subgoal.copy())
            observation, reward, terminated, truncated, info = task.step(action)
            if (terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("deployed episode ended before the frozen horizon")
            costs.append(float(info["tracking_distance"]) ** 2 * scale.dt_seconds)
            rewards.append(float(reward))
            targets.append(target)
            positions.append(observation.achieved_goal.copy())
            actions.append(action.copy())
            previous_action = action.copy()
            history.update(observation)
        discarded = sum(d["renew"] is False for d in decisions) if method in response.METHODS else 0
        expected_executed = args.horizon // 50 if method == "fixed50" else len(shared) + len(checks)
        if len(executed) != expected_executed or len(calls) != len(executed) + discarded:
            raise RuntimeError("deployed planning calls omit a discarded preview or an executed plan")
        row = {"seed": seed, "method": method, "episode_length": args.horizon,
               "episode_return": float(np.sum(rewards)), "tracking_squared_error_integral": float(np.sum(costs)),
               "decisions": decisions, "upper_calls": calls, "executed_plan_steps": executed,
               "upper_inference_calls": len(calls), "executed_plan_count": len(executed),
               "candidate_preview_calls": len(checks) if method in response.METHODS else 0,
               "discarded_preview_calls": discarded, "lower_inference_calls": args.horizon,
               "response_inference_calls": len(checks) if method in response.METHODS else 0,
               "motion_inference_calls": len(checks) if method in motion.METHODS else 0,
               "upper_inference_seconds": upper_seconds, "lower_inference_seconds": lower_seconds,
               "response_inference_seconds": response_seconds, "wall_seconds": time.perf_counter() - started}
        arrays = {"sequence": np.stack(sequences), "ise": np.asarray(costs), "reward": np.asarray(rewards),
                  "target": np.stack(targets), "position": np.stack(positions),
                  "subgoal": np.stack(goals), "action": np.stack(actions),
                  "physical": np.stack(physical), "achieved": np.stack(achieved),
                  "measurement": np.stack(measurements)}
        return row, arrays, names
    finally:
        task.environment.close()


def init_worker(controller, models, critics, args):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = controller, models, critics, args


def sample_episode(case):
    controller, models, critics, args = _WORKER
    return rollout(controller, models, critics, args=args, **case)


def summarize(rows):
    metrics = {m: {k: float(np.mean([r[k] for r in rows if r["method"] == m])) for k in
                  ("episode_return", "tracking_squared_error_integral", "upper_inference_calls",
                   "executed_plan_count", "discarded_preview_calls", "upper_inference_seconds",
                   "response_inference_seconds", "wall_seconds")} for m in spec.METHODS}
    benefits = {f"ise:{m}": metrics[m]["tracking_squared_error_integral"] - metrics["history"]["tracking_squared_error_integral"]
                for m in spec.CONTROLS}
    benefits.update({f"return:{m}": metrics["history"]["episode_return"] - metrics[m]["episode_return"]
                     for m in spec.CONTROLS})
    return {"methods": metrics, "endpoint_values": benefits,
            "root_point_gate_passed": all(v > 0 for v in benefits.values())}


def run(root, *, preflight, output):
    started = time.monotonic()
    args, controller, models, critics, factual, selected = load_frozen(root, preflight=preflight)
    paths = spec.evaluation_paths(root, preflight=preflight)
    cases = [{"seed": seed, "method": method} for seed in paths for method in spec.METHODS]
    np.random.default_rng(np.random.SeedSequence([root, 34_034])).shuffle(cases)
    results = []
    with ProcessPoolExecutor(max_workers=1 if preflight else 16, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(controller, models, critics, args)) as pool:
        for future in as_completed([pool.submit(sample_episode, case) for case in cases]):
            results.append(future.result())
            if len(results) % 40 == 0 or len(results) == len(cases):
                print(f"deployed episodes {len(results)}/{len(cases)}; root {root}", flush=True)
    results.sort(key=lambda item: (item[0]["seed"], spec.METHODS.index(item[0]["method"])))
    rows = [item[0] for item in results]
    raw = source_run.raw_directory(output)
    np.savez_compressed(raw / "episodes.npz", seeds=[r["seed"] for r in rows], methods=[r["method"] for r in rows],
                        feature_names=results[0][2], **{k: np.stack([item[1][k] for item in results]) for k in results[0][1]})
    budget = spec.budget(root, preflight=preflight)
    if sum(r["episode_length"] for r in rows) != budget["evaluation_primitive_steps"]:
        raise RuntimeError("deployed episode cost differs from the frozen roster")
    cell = {"optimizer_seed": root, "source_result": str(spec.source_result(root, preflight=preflight)),
            "controller_selected_iteration": selected, "evaluation_paths": paths, "factual_replay": factual,
            "budget": budget, "rows": rows, "metrics": summarize(rows), "raw_server_directory": str(raw),
            "raw_server_bytes": (raw / "episodes.npz").stat().st_size,
            "wall_seconds": time.monotonic() - started}
    result = {"status": "complete", "protocol": {"protocol_version": spec.EXPERIMENT_PROTOCOL,
              "optimizer_seed": root, "preflight": preflight, "contract": spec.contract()}, "cells": [cell]}
    source_run.write_json(output, result)
    return result


def aggregate(results):
    by_root = {r["protocol"]["optimizer_seed"]: r for r in results}
    if len(results) != len(spec.OPTIMIZER_ROOTS) or set(by_root) != set(spec.OPTIMIZER_ROOTS):
        raise ValueError("deployment aggregation requires the complete eight-root roster")
    vectors, root_metrics = [], {}
    for root in spec.OPTIMIZER_ROOTS:
        result = by_root[root]
        if result["status"] != "complete" or result["protocol"] != {
                "protocol_version": spec.EXPERIMENT_PROTOCOL, "optimizer_seed": root,
                "preflight": False, "contract": spec.contract()}:
            raise ValueError("deployment result differs from the frozen contract")
        cell = result["cells"][0]
        expected = [(seed, m) for seed in spec.evaluation_paths(root, preflight=False) for m in spec.METHODS]
        if (len(cell["rows"]) != len(expected) or sorted((r["seed"], r["method"]) for r in cell["rows"]) != sorted(expected)
                or cell["budget"] != spec.budget(root, preflight=False)
                or any(r["episode_length"] != 1200 for r in cell["rows"])):
            raise ValueError("deployment result has missing episodes or substituted costs")
        metrics = summarize(cell["rows"])
        root_metrics[str(root)] = metrics
        vectors.append([metrics["endpoint_values"][name] for name in spec.ENDPOINTS])
    values = np.asarray(vectors)
    rng = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED))
    draws = rng.integers(0, len(values), size=(spec.BOOTSTRAP_DRAWS, len(values)))
    tail = 100 * .05 / (2 * len(spec.ENDPOINTS))
    intervals = np.percentile(values[draws].mean(axis=1), [tail, 100 - tail], axis=0)
    means = {m: {k: float(np.mean([metrics["methods"][m][k] for metrics in root_metrics.values()]))
                 for k in next(iter(root_metrics.values()))["methods"][m]} for m in spec.METHODS}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
            "optimizer_roots": list(spec.OPTIMIZER_ROOTS), "root_metrics": root_metrics,
            "root_endpoint_values": vectors, "method_means": means,
            "endpoints": {name: {"mean": float(values[:, i].mean()), "adjusted_ci95": intervals[:, i].tolist()}
                          for i, name in enumerate(spec.ENDPOINTS)},
            "deployment_gate_passed": bool(np.all(intervals[0] > 0)),
            "total_primitive_steps": sum(spec.budget(root, preflight=False)["total_primitive_steps"] for root in spec.OPTIMIZER_ROOTS)}
