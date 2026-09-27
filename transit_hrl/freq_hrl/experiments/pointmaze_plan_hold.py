"""Genuine plan holding curves and an executed, equal-call settlement endpoint."""

from __future__ import annotations

from argparse import Namespace
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from .pointmaze_budgeted_trigger import balanced_jitter_schedule, build_parser
from .pointmaze_goal_validation import (
    POINTMAZE_LOWER_ACTION_COST, _json_ready, _training_seed,
    pointmaze_goal_bounds, squash_box_action,
)
from .pointmaze_history_information import DT_SECONDS, RIDGE_ALPHA, causal_design
from .pointmaze_plan_validity_branching import _ridge_fit_predict, _task_options
from .pointmaze_plan_value_qualification import (
    PointMazeRegimeFeatureBuilder, _make_task, build_pointmaze_plan_value_model,
    pointmaze_plan_value_dimensions,
)
from .pointmaze_temporal_plan import (
    HISTORY, METHODS, PROTOCOL_VERSION as CACHE_PROTOCOL, causal_frame, padded_history,
)
from .pointmaze_timing_pair import (
    WINDOWED_PROTOCOL_VERSION, rollout_timing_schedule, timing_pair_opportunities,
)


PROTOCOL_VERSION = "pointmaze_plan_hold_stage28_v1_development"
HOLD_STEPS = 100
SETTLEMENT_STEPS = 150
HORIZONS = (10, 25, 50, HOLD_STEPS, SETTLEMENT_STEPS)


def path_roles(root, *, preflight):
    base = {208001: 3_279_000, 209011: 3_280_000, 209061: 3_281_000}[root]
    return {"fit": list(range(base + 1, base + (3 if preflight else 17))),
            "evaluation": list(range(base + 101, base + (103 if preflight else 109)))}


def cases_for_paths(root, paths, *, horizon, pairs_per_path):
    return [{"seed": seed, "check_step": b * 50 + offset} for seed in paths
            for b, offset in timing_pair_opportunities(
                seed=seed, optimizer_seed=root + 28_028, horizon=horizon,
                period_steps=50, max_offset_steps=25, check_stride_steps=5,
                pairs_per_seed=pairs_per_path, credit_window_steps=SETTLEMENT_STEPS)]


def pair_schedules(seed, check, horizon):
    if check < 50 or check + SETTLEMENT_STEPS > horizon or check % 50 not in (0, 5, 10, 15, 20):
        raise ValueError("plan-hold check is outside the registered grid")
    prefix = tuple(s for s in balanced_jitter_schedule(
        seed=seed, horizon=horizon, period_steps=50, max_offset_steps=25) if s < check)
    return prefix + (check,), prefix + (check + HOLD_STEPS,)


def load_controller(args):
    source = json.loads(args.source_result.read_text())
    original = json.loads(args.controller_result.read_text())
    for result, protocol in ((source, CACHE_PROTOCOL), (original, WINDOWED_PROTOCOL_VERSION)):
        if (result["status"] != "complete" or result["protocol"]["protocol_version"] != protocol
                or result["protocol"]["optimizer_seed"] != args.optimizer_seed or len(result["cells"]) != 1):
            raise ValueError("plan-hold source is not the registered completed root")
    cache, cell = source["cells"][0], original["cells"][0]
    checkpoint_path = Path(cache["raw_server_directory"]) / "controller.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (checkpoint["optimizer_seed"] != args.optimizer_seed
            or checkpoint["selected_iteration"] != cell["controller_selected_iteration"]
            or cache["controller_selected_iteration"] != cell["controller_selected_iteration"]):
        raise ValueError("cached controller differs from the source checkpoint")
    saved = Namespace(**checkpoint["arguments"])
    fields = ("env_id", "horizon", "iterations", "reference_hidden_dim", "learning_rate",
              "upper_period_seconds", "history_seconds", "fast_period_seconds", "maximum_subgoal_delta",
              "checkpoint_evaluation_interval", "max_offset_steps", "check_stride_steps",
              "train_seeds", "selection_seeds", "branch_fit_seeds", "trigger_eval_seeds")
    if (any(_json_ready(getattr(saved, k)) != _json_ready(getattr(args, k)) for k in fields)
            or _json_ready(_task_options(saved)) != _json_ready(_task_options(args))):
        raise ValueError("cached controller options differ from the registered replay")
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=DT_SECONDS, upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds, fast_period_seconds=args.fast_period_seconds)
    if time_scale.upper_period_steps != 50 or args.max_offset_steps != 25 or args.check_stride_steps != 5:
        raise ValueError("plan-hold timing grid changed")
    dimensions = pointmaze_plan_value_dimensions(
        env_id=args.env_id, horizon=args.horizon, time_scale=time_scale, task_options=_task_options(args))
    controller, _ = build_pointmaze_plan_value_model(
        method="hrl_regime_history", dimensions=dimensions, reference_hidden_dim=args.reference_hidden_dim,
        learning_rate=args.learning_rate, optimizer_seed=args.optimizer_seed)
    if controller.config.__dict__ != checkpoint["state_dict"]["config"]:
        raise ValueError("cached controller architecture differs from the replay model")
    controller.load_state_dict(checkpoint["state_dict"])
    factual = cell["aligned_candidate_rows"][0]
    replay = rollout_timing_schedule(
        controller, seed=factual["seed"], decision_steps=tuple(factual["decision_steps"]), capture_step=0,
        env_id=args.env_id, horizon=args.horizon, time_scale=time_scale,
        maximum_subgoal_delta=args.maximum_subgoal_delta, task_options=_task_options(args))
    errors = {k: abs(replay[k] - factual[k]) for k in ("episode_return", "tracking_squared_error_integral")}
    if any(v > 1e-8 for v in errors.values()):
        raise RuntimeError("cached controller differs from the frozen factual rollout")
    return cache, cell, controller, time_scale, checkpoint_path, {"seed": factual["seed"], "absolute_errors": errors}


def validate_paths(args, roles, cache):
    inherited = {s for role in ("train", "selection", "branch_fit", "trigger_eval")
                 for s in getattr(args, role + "_seeds")}
    inherited.update(s for paths in cache["temporal_seed_roles"].values() for s in paths)
    inherited.update(_training_seed(optimizer_seed=args.optimizer_seed, rollout_root=seed, iteration=i)
                     for seed in args.train_seeds for i in range(args.iterations))
    if (set(roles["fit"]).intersection(roles["evaluation"])
            or inherited.intersection(roles["fit"] + roles["evaluation"])):
        raise ValueError("plan-hold paths overlap inherited or fit/evaluation paths")


def rollout_window(controller, *, seed, check, schedule, args, time_scale):
    task = _make_task(env_id=args.env_id, seed=seed, horizon=args.horizon, **_task_options(args))
    try:
        observation = task.reset()
        low, high = pointmaze_goal_bounds(task.environment)
        adapter = RelativeSubgoalAdapter(
            maximum_delta=np.full(observation.achieved_goal.size, args.maximum_subgoal_delta, dtype=np.float32),
            world_low=low, world_high=high, action_cost=POINTMAZE_LOWER_ACTION_COST)
        history = PointMazeRegimeFeatureBuilder(time_scale=time_scale, task_dim=len(observation.task_measurement))
        history.reset(observation)
        controller.reset_recurrent_inference()
        subgoal, previous_action = observation.achieved_goal.copy(), np.zeros_like(task.action_low)
        last_plan, frames, costs, rewards, calls = -1, deque(maxlen=HISTORY), [], [], []
        for step in range(check + SETTLEMENT_STEPS):
            names, frame = causal_frame(observation, subgoal, previous_action, step=step,
                                       last_plan=last_plan, horizon=args.horizon)
            frames.append(frame)
            if step == check:
                sequence = padded_history(frames)
                policy_prefix = np.concatenate((history.history, frame, subgoal))
            if step in schedule:
                output = controller.plan_goal(history.upper_state(observation, oracle_context=None), sample=False)
                subgoal = adapter.decode(np.asarray(output["action"], dtype=np.float32), observation.achieved_goal)
                last_plan = step
                calls.append(step)
            output = controller.act_conditioned(history.lower_state(observation, subgoal=subgoal), sample=False)
            action = squash_box_action(np.asarray(output["action"], dtype=np.float32), task.action_low, task.action_high)
            observation, reward, terminated, truncated, info = task.step(action)
            if (terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("plan-hold window ended early")
            if step >= check:
                costs.append(float(info["tracking_distance"]) ** 2 * time_scale.dt_seconds)
                rewards.append(float(reward))
            previous_action = action.copy()
            history.update(observation)
        return {"sequence": sequence, "policy_prefix": policy_prefix, "feature_names": names,
                "step_ise": np.asarray(costs), "step_reward": np.asarray(rewards), "calls": calls,
                "primitive_steps": check + SETTLEMENT_STEPS}
    finally:
        task.environment.close()


def combine_pair(case, renew, keep):
    if (not np.array_equal(renew["sequence"], keep["sequence"])
            or not np.array_equal(renew["policy_prefix"], keep["policy_prefix"])
            or renew["feature_names"] != keep["feature_names"]):
        raise RuntimeError("plan-hold arms differ before intervention")
    check = case["check_step"]
    prefix = [s for s in renew["calls"] if s < check]
    if (keep["calls"] != prefix + [check + HOLD_STEPS] or renew["calls"] != prefix + [check]
            or any(arm["step_ise"].shape != (SETTLEMENT_STEPS,) for arm in (renew, keep))):
        raise RuntimeError("plan-hold continuation or executed call budget differs")
    end = np.asarray(HORIZONS) - 1
    ise = {name: np.cumsum(arm["step_ise"])[end] for name, arm in (("renew", renew), ("keep", keep))}
    reward = {name: np.cumsum(arm["step_reward"])[end] for name, arm in (("renew", renew), ("keep", keep))}
    counts = {name: [sum(s < check + h for s in arm["calls"]) for h in HORIZONS]
              for name, arm in (("renew", renew), ("keep", keep))}
    return {**case, "sequence": renew["sequence"], "feature_names": renew["feature_names"],
            "curve": ise["keep"] - ise["renew"], "arm_ise_curves": ise, "arm_reward_curves": reward,
            "upper_calls_at_horizons": counts, "upper_call_steps": {"renew": renew["calls"], "keep": keep["calls"]},
            "step_ise": np.stack((renew["step_ise"], keep["step_ise"])),
            "primitive_steps": renew["primitive_steps"] + keep["primitive_steps"]}


def init_worker(controller, args, time_scale):
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = controller, args, time_scale


def sample_case(case):
    controller, args, time_scale = _WORKER
    arms = [rollout_window(controller, seed=case["seed"], check=case["check_step"], schedule=s,
                           args=args, time_scale=time_scale)
            for s in pair_schedules(case["seed"], case["check_step"], args.horizon)]
    return combine_pair(case, *arms)


def fit_curves(train, query, *, root):
    if set(r["seed"] for r in train).intersection(r["seed"] for r in query):
        raise ValueError("plan-hold fit and evaluation paths overlap")
    names = train[0]["feature_names"]
    if any(r["feature_names"] != names for r in [*train, *query]):
        raise ValueError("plan-hold feature schemas differ")
    x, q = np.stack([r["sequence"] for r in train]), np.stack([r["sequence"] for r in query])
    y = np.stack([r["curve"] for r in train]) / (np.asarray(HORIZONS) * DT_SECONDS)
    predictions, fits = {}, {}
    for method in METHODS:
        design = causal_design(x, train, names, method=method, root=root)
        future = causal_design(q, query, names, method=method, root=root)
        heads = [_ridge_fit_predict(design, y[:, i], future, alpha=RIDGE_ALPHA) for i in range(len(HORIZONS))]
        predictions[method] = np.column_stack([p for p, _ in heads])
        fits[method] = {"training_rows": len(train), "scalar_linear_solves": len(heads),
                        "parameter_count": (design.shape[1] + 1) * len(heads), "alpha": RIDGE_ALPHA,
                        "feature_mean": heads[0][1]["feature_mean"], "feature_scale": heads[0][1]["feature_scale"],
                        "weights": np.column_stack([d["weights"] for _, d in heads]).tolist()}
    return predictions, fits


def summarize(rows):
    rates = np.asarray([r["curve"] for r in rows]) / (np.asarray(HORIZONS) * DT_SECONDS)
    predictions = {m: np.asarray([r["predicted_rates"][m] for r in rows]) for m in METHODS}
    predictions["zero"] = np.zeros_like(rates)
    mse_by_h = {m: np.mean((p - rates)**2, axis=0) for m, p in predictions.items()}
    gross = {m: float(e[:-1].mean()) for m, e in mse_by_h.items()}
    settled = {m: float(e[-1]) for m, e in mse_by_h.items()}
    choices = {m: (p[:, -1] > 0).astype(int) for m, p in predictions.items() if m != "zero"}
    choices.update(always_keep=np.zeros(len(rows), dtype=int), always_renew=np.ones(len(rows), dtype=int))
    gains = {m: float(np.mean((choices["history"] - a) * rates[:, -1] * SETTLEMENT_STEPS * DT_SECONDS))
             for m, a in choices.items() if m != "history"}
    return {"opportunities": len(rows), "rate_mse_by_horizon": mse_by_h, "gross_hold_rate_mse": gross,
            "equal_call_settled_rate_mse": settled, "renew_counts": {m: int(a.sum()) for m, a in choices.items()},
            "history_settled_ise_benefit_vs_control": gains,
            "prediction_gate_passed": all(e["history"] < min(e[m] for m in ("zero", *METHODS[1:]))
                                          for e in (gross, settled)),
            "decision_gate_passed": all(g > 0 for g in gains.values())}


def run_cell(args):
    if args.workers < 1:
        raise ValueError("plan-hold replay requires workers")
    torch.set_num_threads(1)
    cache, source, controller, scale, checkpoint_path, factual = load_controller(args)
    roles = path_roles(args.optimizer_seed, preflight=args.horizon == 300)
    validate_paths(args, roles, cache)
    cases = [dict(c, role=role) for role, paths in roles.items() for c in cases_for_paths(
        args.optimizer_seed, paths, horizon=args.horizon, pairs_per_path=args.pairs_per_path)]
    print(f"cached controller matched; collecting {len(cases)} true hold/renew pairs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(controller, args, scale)) as pool:
        futures = [pool.submit(sample_case, case) for case in cases]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0 or len(rows) == len(cases):
                print(f"plan-hold pairs complete: {len(rows)}/{len(cases)}", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["check_step"]))
    raw_dir = args.output.resolve().parent.with_name(args.output.parent.name + "_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir / "plan_hold_pairs.npz", sequences=np.stack([r["sequence"] for r in rows]),
                        curves=np.stack([r["curve"] for r in rows]), step_ise=np.stack([r["step_ise"] for r in rows]),
                        seeds=[r["seed"] for r in rows], check_steps=[r["check_step"] for r in rows],
                        roles=[r["role"] for r in rows], feature_names=rows[0]["feature_names"])
    train, query = ([r for r in rows if r["role"] == role] for role in ("fit", "evaluation"))
    predictions, fits = fit_curves(train, query, root=args.optimizer_seed)
    scores = [{**{k: v for k, v in r.items() if k not in ("sequence", "step_ise")},
               "predicted_rates": {m: p[i] for m, p in predictions.items()}} for i, r in enumerate(query)]
    metrics = summarize(scores)
    return {"optimizer_seed": args.optimizer_seed, "plan_hold_seed_roles": roles,
            "controller_selected_iteration": source["controller_selected_iteration"],
            "controller_checkpoint": str(checkpoint_path), "factual_replay": factual,
            "controller_reconstruction_primitive_steps": 0, "controller_updates": 0, "optimizer_updates": 0,
            "factual_replay_primitive_steps": args.horizon,
            "plan_hold_replay_primitive_steps": sum(r["primitive_steps"] for r in rows),
            "reused_controller_source_primitive_steps": sum(cache["controller_reconstruction_primitive_steps"].values()),
            "training_pairs": len(train), "evaluation_pairs": len(query), "linear_fits": len(fits),
            "scalar_linear_solves": sum(f["scalar_linear_solves"] for f in fits.values()),
            "sequence_shape": list(rows[0]["sequence"].shape), "feature_names": rows[0]["feature_names"],
            "coverage": {role: {str(s): {str(o): sum(r["seed"] == s and r["check_step"] % 50 == o for r in rows)
                                        for o in (0, 5, 10, 15, 20)} for s in paths} for role, paths in roles.items()},
            "raw_server_directory": str(raw_dir), "raw_server_bytes": (raw_dir / "plan_hold_pairs.npz").stat().st_size,
            "fits": fits, "rows": scores, "metrics": metrics,
            "path_metrics": {str(s): summarize([r for r in scores if r["seed"] == s]) for s in roles["evaluation"]},
            "development_gate_passed": metrics["prediction_gate_passed"] and metrics["decision_gate_passed"]}


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--controller-result", type=Path, required=True)
    parser.add_argument("--pairs-per-path", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    output = {"status": "dry_run" if args.dry_run else "complete", "protocol": {
        "protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
        "source_result": str(args.source_result), "controller_result": str(args.controller_result),
        "evidence_role": "fresh_path_true_plan_hold_development_only", "policy_deployment": False,
        "hold_steps": HOLD_STEPS, "settlement_steps": SETTLEMENT_STEPS, "curve_horizons_steps": HORIZONS,
        "history_steps": HISTORY, "ridge_alpha": RIDGE_ALPHA, "methods": METHODS,
        "pairs_per_path": args.pairs_per_path, "workers": args.workers,
        "seed_roles": path_roles(args.optimizer_seed, preflight=args.horizon == 300),
        "primary_endpoint": "history_ise_benefit_at_executed_equal_call_150_step_settlement",
        "gross_hold_postcheck_calls": {"renew": 1, "keep": 0},
        "settlement_postcheck_calls": {"renew": 1, "keep": 1}},
        "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
