"""Short-window policy improvement with out-of-path continuation values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from freq_hrl.rl.dual_actor_critic import ValueNet
from .pointmaze_adaptive_pair_diagnostic import rollout_intervention
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_deployed_pair_diagnostic import replay_source_controller
from .pointmaze_goal_validation import _json_ready
from .pointmaze_onecheck_advantage import (
    advantage_matrix, episode_row, predict_advantage, require_factual_match, select_checks,
)
from .pointmaze_plan_validity_branching import _ridge_fit_predict


PROTOCOL_VERSION = "pointmaze_continuation_credit_stage16_v1_development"
VALUE_EPOCHS = 64
VALUE_WIDTH = 64
RIDGE_ALPHA = 100.0


def fit_continuation(train, query, *, seed, epochs=VALUE_EPOCHS):
    """Fit V_pi on other paths; queries contain states, never fitting targets."""
    x = np.asarray([r["features"] for r in train], dtype=np.float64)
    y = np.asarray([r["cost_to_go"] for r in train], dtype=np.float64)
    groups = np.asarray([r["seed"] for r in train])
    names = train[0]["feature_names"]
    if any(r["feature_names"] != names for r in [*train, *query]):
        raise ValueError("continuation feature schema differs")
    if set(groups).intersection(r["seed"] for r in query):
        raise ValueError("continuation fit and query paths overlap")
    # Each exogenous path has equal mass, regardless of how many suffix states it supplies.
    weights = np.asarray([1.0 / np.sum(groups == s) for s in groups])
    weights /= weights.sum()
    mean = np.sum(weights[:, None] * x, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (x - mean) ** 2, axis=0))
    scale[scale < 1e-8] = 1.0
    remaining_index = names.index("remaining_fraction")
    remaining = x[:, remaining_index]
    rate = float(np.sum(weights * remaining * y) / np.sum(weights * remaining ** 2))
    target_scale = max(float(np.sqrt(np.sum(weights * y ** 2))), 1e-8)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = ValueNet(state_dim=x.shape[1], hidden_dim=VALUE_WIDTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    state = torch.tensor((x - mean) / scale, dtype=torch.float32)
    time_left = torch.tensor(remaining, dtype=torch.float32)
    target = torch.tensor((y - rate * remaining) / target_scale, dtype=torch.float32)
    weight = torch.tensor(weights, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.sum(weight * (time_left * model(state) - target) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    q = np.asarray([r["features"] for r in query], dtype=np.float64)
    with torch.no_grad():
        residual = model(torch.tensor((q - mean) / scale, dtype=torch.float32)).numpy()
    prediction = q[:, remaining_index] * (rate + target_scale * residual)
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("non-finite continuation estimate")
    return prediction, {
        "training_paths": sorted(map(int, set(groups))), "training_rows": len(train),
        "query_rows": len(query), "epochs": epochs, "hidden_dim": VALUE_WIDTH,
        "optimizer_steps": epochs, "seed": seed,
        "baseline_cost_rate": rate, "target_scale": target_scale,
        "last_training_normalized_loss": float(loss.detach()),
    }


def crossfit_continuations(traces, pairs, *, root, fit_seeds, eval_seeds, epochs=VALUE_EPOCHS):
    paths = set(fit_seeds)
    if (len(paths) < 2 or paths.intersection(eval_seeds)
            or {r["seed"] for r in traces} != paths or {r["seed"] for r in pairs} != paths):
        raise ValueError("continuation training has wrong seed roles")
    rows, folds = [], []
    for held_out in sorted(paths):
        train = [r for r in traces if r["seed"] != held_out]
        held_pairs = [r for r in pairs if r["seed"] == held_out]
        query = [{"seed": held_out, **r[f"{arm}_endpoint"]}
                 for r in held_pairs for arm in ("now", "wait")]
        prediction, diagnostics = fit_continuation(
            train, query, seed=int(np.random.SeedSequence([root, held_out, 16_016])
                                   .generate_state(1)[0]), epochs=epochs,
        )
        folds.append({"held_out_path": held_out, **diagnostics})
        for index, row in enumerate(held_pairs):
            now, wait = prediction[2 * index:2 * index + 2]
            difference = float(wait - now)
            rows.append({
                **row, "predicted_now_tail": float(now), "predicted_wait_tail": float(wait),
                "predicted_tail_advantage": difference,
                "bootstrap_advantage": row["window_ise_advantage"] + difference,
            })
    truth = np.asarray([r["actual_tail_advantage"] for r in rows])
    estimate = np.asarray([r["predicted_tail_advantage"] for r in rows])
    mse = float(np.mean((truth - estimate) ** 2))
    null_mse = float(np.mean(truth ** 2))
    return rows, {"folds": folds, "value_trace_rows": len(traces),
                  "continuation_contrast_mse": mse, "zero_contrast_mse": null_mse,
                  "contrast_gate_passed": mse < null_mse}


def fit_trigger(rows, *, target):
    matrix, names = advantage_matrix(rows)
    _, model = _ridge_fit_predict(matrix, np.asarray([r[target] for r in rows]),
                                  matrix, alpha=RIDGE_ALPHA)
    return {"model": model, "feature_names": names, "threshold": 0.0, "target": target}


def run_cell(args):
    source, controller, time_scale = replay_source_controller(args)
    period = time_scale.upper_period_steps
    reference_predictor = source["trigger_predictor"]
    options = dict(predictor=reference_predictor, args=args, time_scale=time_scale)
    pairs, traces = [], []
    for seed in args.branch_fit_seeds:
        reference = rollout_intervention(
            controller, seed=seed, intervention_step=None, arm=None,
            collect_value_trace=True, **options,
        )
        traces.extend({"seed": seed, **r} for r in reference["value_trace"])
        for step in select_checks(
            root=args.optimizer_seed, seed=seed, schedule=reference["decision_steps"],
            period=period, stride=args.check_stride_steps,
            deadline=args.max_offset_steps, count=args.pairs_per_seed,
        ):
            now, wait = [rollout_intervention(
                controller, seed=seed, intervention_step=step, arm=arm,
                collect_value_trace=True, **options,
            ) for arm in ("now", "wait_one_check")]
            if any(not np.array_equal(now[key], wait[key]) for key in ("prefix", "features")):
                raise RuntimeError("continuation pair causal prefixes differ")
            factual = now if now["score"] >= reference_predictor["threshold"] else wait
            require_factual_match(factual, reference)
            endpoints = [r["value_trace"][0] for r in (now, wait)]
            if any(r["step"] != step + period for r in endpoints):
                raise RuntimeError("continuation state is not at the window boundary")
            full = wait["tracking_squared_error_integral"] - now["tracking_squared_error_integral"]
            window = wait["window_ise"] - now["window_ise"]
            tail = endpoints[1]["cost_to_go"] - endpoints[0]["cost_to_go"]
            if not np.isclose(full, window + tail, atol=1e-10, rtol=1e-10):
                raise RuntimeError("window and continuation credit do not reconstruct full advantage")
            for arm in (now, wait):
                traces.extend({"seed": seed, **r} for r in arm["value_trace"])
            pairs.append({
                "seed": seed, "check_step": step,
                "feature_names": list(now["feature_names"]),
                "causal_features": now["features"].tolist(),
                "offset_fraction": (step % period) / args.max_offset_steps,
                "remaining_fraction": (args.horizon - step) / args.horizon,
                "full_ise_advantage": full, "window_ise_advantage": window,
                "actual_tail_advantage": tail,
                "now_endpoint": endpoints[0], "wait_endpoint": endpoints[1],
                "prefix_max_abs_difference": 0.0,
                "now_call_step": now["decision_steps"][step // period],
                "wait_call_step": wait["decision_steps"][step // period],
                "now_upper_call_count": len(now["decision_steps"]),
                "wait_upper_call_count": len(wait["decision_steps"]),
            })
    rows, critic_diagnostics = crossfit_continuations(
        traces, pairs, root=args.optimizer_seed, fit_seeds=args.branch_fit_seeds,
        eval_seeds=args.trigger_eval_seeds,
    )
    predictors = {key: fit_trigger(rows, target=target) for key, target in (
        ("short_only", "window_ise_advantage"), ("bootstrap", "bootstrap_advantage"),
    )}
    evaluations = {key: [] for key in predictors}
    reference_rows = []
    for saved in source["aligned_candidate_rows"]:
        seed = saved["seed"]
        reference = rollout_intervention(
            controller, seed=seed, intervention_step=None, arm=None, **options,
        )
        require_factual_match(reference, saved)
        reference_rows.append(episode_row(seed, reference))
        for key, fitted in predictors.items():
            candidate = rollout_intervention(
                controller, seed=seed, intervention_step=None, arm=None,
                predictor={"threshold": 0.0}, args=args, time_scale=time_scale,
                score_fn=lambda n, v, o, r: predict_advantage(fitted, n, v, o, r),
            )
            evaluations[key].append(episode_row(seed, candidate))
    means = {key: float(np.mean([r["tracking_squared_error_integral"] for r in values]))
             for key, values in {**evaluations, "stage12": reference_rows,
                                 "fixed": source["fixed_replay_rows"]}.items()}
    return {
        "optimizer_seed": args.optimizer_seed, "horizon": args.horizon,
        "selected_checkpoint_iteration": source["controller_selected_iteration"],
        "replayed_controller_training_iterations": args.iterations,
        "inherited_branch_fit_primitive_steps": source["branch_fit_primitive_steps_replayed"],
        "branch_fit_seeds": args.branch_fit_seeds, "trigger_eval_seeds": args.trigger_eval_seeds,
        "pairs_per_seed": args.pairs_per_seed,
        "branch_fit_rows": rows, "critic_diagnostics": critic_diagnostics,
        "predictors": predictors, "fixed_replay_rows": source["fixed_replay_rows"],
        "reference_rows": reference_rows, "evaluation_rows": evaluations, "mean_ise": means,
        "additional_primitive_steps": {
            "reference_fit_trajectories": len(args.branch_fit_seeds) * args.horizon,
            "paired_fit_replay": len(rows) * 2 * args.horizon,
            "held_out_reference_replay": len(reference_rows) * args.horizon,
            **{f"held_out_{key}_evaluation": len(values) * args.horizon
               for key, values in evaluations.items()},
        },
    }


def main(argv=None):
    parser = build_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--pairs-per-seed", type=int, default=12)
    args = parser.parse_args(argv)
    output = {
        "status": "dry_run" if args.dry_run else "complete",
        "protocol": {
            "protocol_version": PROTOCOL_VERSION, "source_result": str(args.source_result),
            "optimizer_seed": args.optimizer_seed, "pairs_per_seed": args.pairs_per_seed,
            "value_epochs": VALUE_EPOCHS, "value_hidden_dim": VALUE_WIDTH,
            "ridge_alpha": RIDGE_ALPHA, "threshold": 0.0,
            "evidence_role": "crossfitted_continuation_development_only",
        },
        "cells": [] if args.dry_run else [run_cell(args)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
