"""Temporal supervision for matched-budget, one-check plan renewal costs."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from torch import nn

from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from .pointmaze_budgeted_trigger import balanced_jitter_schedule, build_parser
from .pointmaze_deployed_pair_diagnostic import replay_source_controller
from .pointmaze_fresh_future_diagnostic import reconstruction_steps
from .pointmaze_goal_validation import (
    POINTMAZE_LOWER_ACTION_COST, _json_ready, pointmaze_goal_bounds, squash_box_action,
)
from .pointmaze_plan_validity_branching import PointMazeRegimeFeatureBuilder, _task_options
from .pointmaze_plan_value_qualification import _make_task
from .pointmaze_timing_pair import rollout_timing_schedule, timing_pair_opportunities


PROTOCOL_VERSION = "pointmaze_temporal_plan_stage26_v1_development"
HORIZONS = (10, 25, 50)
HISTORY = 64
WIDTH = 32
METHODS = ("history", "current_repeat", "shuffled_history")


def path_roles(root, *, preflight):
    bases = {208001: 3_259_000, 209011: 3_260_000, 209061: 3_261_000}
    base = bases[root]
    return {"fit": list(range(base + 1, base + (3 if preflight else 17))),
            "evaluation": list(range(base + 101, base + (103 if preflight else 109)))}


def cases_for_paths(root, paths, *, horizon, pairs_per_path):
    return [{"seed": s, "check_step": b * 50 + offset} for s in paths
            for b, offset in timing_pair_opportunities(
                seed=s, optimizer_seed=root + 26_026, horizon=horizon, period_steps=50,
                max_offset_steps=25, check_stride_steps=5, pairs_per_seed=pairs_per_path,
                credit_window_steps=50)]


def pair_schedules(seed, check, horizon):
    schedule = list(balanced_jitter_schedule(seed=seed, horizon=horizon,
                                            period_steps=50, max_offset_steps=25))
    if not 50 <= check < horizon - 50 or check % 50 not in (0, 5, 10, 15, 20):
        raise ValueError("temporal probe is outside the matched one-check grid")
    now, wait = schedule.copy(), schedule.copy()
    now[check // 50], wait[check // 50] = check, check + 5
    return tuple(now), tuple(wait)


def causal_frame(observation, subgoal, previous_action, *, step, last_plan, horizon):
    groups = {"physical": observation.physical, "achieved": observation.achieved_goal,
              "target_error": observation.target - observation.achieved_goal,
              "waypoint_error": subgoal - observation.achieved_goal,
              "measured": observation.task_measurement, "previous_action": previous_action}
    names = [f"{name}_{i}" for name, values in groups.items() for i in range(len(values))]
    names += ["plan_age_fraction", "within_bin_fraction", "remaining_fraction",
              "budget_spent", "valid_observation"]
    values = np.concatenate((*groups.values(), [(step - last_plan) / 50, (step % 50) / 50,
                             (horizon - step) / horizon, float(last_plan // 50 == step // 50), 1.]))
    return names, values.astype(np.float32)


def padded_history(frames):
    pad = frames[0].copy()
    pad[-1] = 0.
    return np.stack([pad] * (HISTORY - len(frames)) + list(frames))


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
        last_plan, frames, costs, calls = -1, deque(maxlen=HISTORY), [], []
        for step in range(check + HORIZONS[-1]):
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
            observation, _, terminated, truncated, info = task.step(action)
            if (terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("temporal window ended early")
            if step >= check:
                costs.append(float(info["tracking_distance"]) ** 2 * time_scale.dt_seconds)
            previous_action = action.copy()
            history.update(observation)
        return {"sequence": sequence, "policy_prefix": policy_prefix, "feature_names": names,
                "step_ise": np.asarray(costs), "calls": calls, "primitive_steps": check + HORIZONS[-1]}
    finally:
        task.environment.close()


def combine_pair(case, now, wait):
    if (not np.array_equal(now["sequence"], wait["sequence"])
            or not np.array_equal(now["policy_prefix"], wait["policy_prefix"])
            or now["feature_names"] != wait["feature_names"]):
        raise RuntimeError("temporal arms differ before intervention")
    if now["step_ise"].shape != (50,) or wait["step_ise"].shape != (50,):
        raise RuntimeError("temporal credit window is incomplete")
    counts = [[sum(s < case["check_step"] + h for s in arm["calls"]) for h in HORIZONS]
              for arm in (now, wait)]
    if counts[0] != counts[1]:
        raise RuntimeError("temporal labels have unequal upper call budgets")
    curve = np.cumsum(wait["step_ise"] - now["step_ise"])[np.asarray(HORIZONS) - 1]
    return {**case, "sequence": now["sequence"], "curve": curve, "feature_names": now["feature_names"],
            "upper_calls_at_horizons": counts[0], "primitive_steps": now["primitive_steps"] + wait["primitive_steps"]}


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


def history_view(x, rows, method, *, root):
    if method == "history":
        return x.copy()
    if method == "current_repeat":
        return np.repeat(x[:, -1:, :], x.shape[1], axis=1)
    if method != "shuffled_history":
        raise ValueError("unknown temporal representation")
    out = x.copy()
    for i, row in enumerate(rows):
        order = np.random.default_rng(np.random.SeedSequence(
            [root, row["seed"], row["check_step"], 26_028])).permutation(x.shape[1] - 1)
        out[i, :-1] = x[i, order]
    return out


class PlanCurveNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.encoder = nn.GRU(features, WIDTH, batch_first=True)
        self.head = nn.Linear(WIDTH, len(HORIZONS))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        _, hidden = self.encoder(x)
        return self.head(hidden[-1])


def fit_temporal(train, query, *, root, epochs):
    if set(r["seed"] for r in train).intersection(r["seed"] for r in query):
        raise ValueError("temporal training and evaluation paths overlap")
    if any(r["feature_names"] != train[0]["feature_names"] for r in [*train, *query]):
        raise ValueError("temporal feature schema differs")
    torch.set_num_threads(1)
    x, q = np.stack([r["sequence"] for r in train]), np.stack([r["sequence"] for r in query])
    mean, scale = x.mean(axis=(0, 1)), x.std(axis=(0, 1))
    scale = np.where(scale > 1e-8, scale, 1.)
    rates = np.stack([r["curve"] for r in train]) / (np.asarray(HORIZONS) * .01)
    target_scale = np.maximum(np.sqrt(np.mean(rates ** 2, axis=0)), 1e-8)
    target = torch.tensor(rates / target_scale, dtype=torch.float32)
    predictions, diagnostics, states = {}, {}, {}
    seed = int(np.random.SeedSequence([root, 26_027]).generate_state(1)[0])
    for method in METHODS:
        torch.manual_seed(seed)
        model = PlanCurveNet(x.shape[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        state = torch.tensor(history_view((x - mean) / scale, train, method, root=root), dtype=torch.float32)
        future = torch.tensor(history_view((q - mean) / scale, query, method, root=root), dtype=torch.float32)
        rng = np.random.default_rng(seed)
        updates = 0
        for _ in range(epochs):
            for indices in np.array_split(rng.permutation(len(train)), (len(train) + 63) // 64):
                optimizer.zero_grad()
                loss = torch.mean((model(state[indices]) - target[indices]) ** 2)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                updates += 1
        model.eval()
        with torch.no_grad():
            predictions[method] = model(future).numpy() * target_scale
        if not np.isfinite(predictions[method]).all():
            raise RuntimeError("non-finite temporal curve prediction")
        diagnostics[method] = {"epochs": epochs, "optimizer_steps": updates, "training_rows": len(train),
                               "parameter_count": sum(p.numel() for p in model.parameters()), "initialization_seed": seed}
        states[method] = model.state_dict()
    return predictions, {"feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
                         "target_scale": target_scale.tolist(), "fits": diagnostics}, states


def summarize(rows):
    rates = np.asarray([r["curve"] for r in rows]) / (np.asarray(HORIZONS) * .01)
    predictions = {m: np.asarray([r["predicted_rates"][m] for r in rows]) for m in METHODS}
    errors = {m: float(np.mean((p - rates) ** 2)) for m, p in predictions.items()}
    errors["zero"] = float(np.mean(rates ** 2))
    choices = {m: p[:, -1] > 0 for m, p in predictions.items()}
    choices.update(always_wait=np.zeros(len(rows), dtype=bool), always_now=np.ones(len(rows), dtype=bool))
    gains = {m: float(np.mean((choices["history"].astype(int) - d.astype(int)) * rates[:, -1] * .5))
             for m, d in choices.items() if m != "history"}
    return {"opportunities": len(rows), "curve_rate_mse": errors,
            "now_counts": {m: int(a.sum()) for m, a in choices.items()}, "history_ise_benefit_vs_control": gains,
            "prediction_gate_passed": errors["history"] < min(errors[m] for m in ("zero", *METHODS[1:])),
            "decision_gate_passed": all(g > 0 for g in gains.values())}


def run_cell(args):
    if args.workers < 1 or args.curve_epochs < 1:
        raise ValueError("temporal screen requires workers and positive epochs")
    roles = path_roles(args.optimizer_seed, preflight=args.horizon == 300)
    old_roles = [s for role in ("train", "selection", "branch_fit", "trigger_eval") for s in getattr(args, role + "_seeds")]
    if set(old_roles).intersection(roles["fit"] + roles["evaluation"]):
        raise ValueError("temporal sample paths overlap inherited paths")
    cases = [dict(r, role=role) for role, seeds in roles.items() for r in cases_for_paths(
        args.optimizer_seed, seeds, horizon=args.horizon, pairs_per_path=args.pairs_per_path)]
    raw_dir = args.output.resolve().parent.with_name(args.output.parent.name + "_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("reconstructing frozen controller once; temporal raw cache remains on server", flush=True)
    source, controller, time_scale = replay_source_controller(args)
    torch.save({"state_dict": controller.state_dict(), "optimizer_seed": args.optimizer_seed,
                "selected_iteration": source["controller_selected_iteration"], "arguments": vars(args)}, raw_dir / "controller.pt")
    factual = source["aligned_candidate_rows"][0]
    replay = rollout_timing_schedule(controller, seed=factual["seed"], decision_steps=tuple(factual["decision_steps"]),
                                    capture_step=0, env_id=args.env_id, horizon=args.horizon, time_scale=time_scale,
                                    maximum_subgoal_delta=args.maximum_subgoal_delta, task_options=_task_options(args))
    if any(abs(replay[k] - factual[k]) > 1e-8 for k in ("episode_return", "tracking_squared_error_integral")):
        raise RuntimeError("reconstructed controller differs from frozen factual rollout")
    print(f"controller matched; collecting {len(cases)} temporal pairs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(controller, args, time_scale)) as pool:
        futures = [pool.submit(sample_case, case) for case in cases]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0 or len(rows) == len(cases):
                print(f"temporal pairs complete: {len(rows)}/{len(cases)}", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["check_step"]))
    np.savez_compressed(raw_dir / "temporal_pairs.npz", sequences=np.stack([r["sequence"] for r in rows]),
                        curves=np.stack([r["curve"] for r in rows]), seeds=[r["seed"] for r in rows],
                        check_steps=[r["check_step"] for r in rows], roles=[r["role"] for r in rows],
                        feature_names=rows[0]["feature_names"])
    train, query = ([r for r in rows if r["role"] == role] for role in ("fit", "evaluation"))
    print("raw pairs saved; fitting matched temporal representations", flush=True)
    predictions, diagnostics, states = fit_temporal(train, query, root=args.optimizer_seed, epochs=args.curve_epochs)
    torch.save({"models": states, **diagnostics, "feature_names": rows[0]["feature_names"]}, raw_dir / "curve_models.pt")
    scores = [{"seed": r["seed"], "check_step": r["check_step"], "curve": r["curve"],
               "predicted_rates": {m: p[i] for m, p in predictions.items()}} for i, r in enumerate(query)]
    metrics = summarize(scores)
    return {"optimizer_seed": args.optimizer_seed, "branch_fit_seeds": args.branch_fit_seeds,
            "trigger_eval_seeds": args.trigger_eval_seeds, "temporal_seed_roles": roles,
            "controller_selected_iteration": source["controller_selected_iteration"],
            "controller_reconstruction_primitive_steps": reconstruction_steps(args),
            "factual_replay_primitive_steps": args.horizon,
            "temporal_replay_primitive_steps": sum(r["primitive_steps"] for r in rows),
            "training_pairs": len(train), "evaluation_pairs": len(query),
            "sequence_shape": list(rows[0]["sequence"].shape), "feature_names": rows[0]["feature_names"],
            "coverage": {role: {str(s): {str(o): sum(r["seed"] == s and r["check_step"] % 50 == o for r in rows)
                                        for o in (0, 5, 10, 15, 20)} for s in seeds} for role, seeds in roles.items()},
            "raw_server_directory": str(raw_dir),
            "raw_server_bytes": {p.name: p.stat().st_size for p in raw_dir.iterdir() if p.is_file()},
            "fits": diagnostics, "rows": scores, "metrics": metrics,
            "path_metrics": {str(s): summarize([r for r in scores if r["seed"] == s]) for s in roles["evaluation"]},
            "development_gate_passed": metrics["prediction_gate_passed"] and metrics["decision_gate_passed"]}


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--pairs-per-path", type=int, default=20)
    parser.add_argument("--curve-epochs", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    output = {"status": "dry_run" if args.dry_run else "complete", "protocol": {
        "protocol_version": PROTOCOL_VERSION, "optimizer_seed": args.optimizer_seed,
        "source_result": str(args.source_result), "candidate": "history", "controls": METHODS[1:],
        "curve_horizons_steps": HORIZONS, "history_steps": HISTORY, "gru_width": WIDTH,
        "pairs_per_path": args.pairs_per_path, "curve_epochs": args.curve_epochs, "workers": args.workers,
        "seed_roles": path_roles(args.optimizer_seed, preflight=args.horizon == 300),
        "evidence_role": "fresh_path_temporal_plan_development_only", "policy_deployment": False},
        "cells": [] if args.dry_run else [run_cell(args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
