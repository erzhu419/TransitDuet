"""One fitted policy-improvement step using adaptive one-check timing credit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pointmaze_adaptive_pair_diagnostic import rollout_intervention
from .pointmaze_budgeted_trigger import build_parser
from .pointmaze_compact_plan_validity import (
    _base_matrix, _causal_validity_interactions, _grouped_ridge_fit_predict,
)
from .pointmaze_deployed_pair_diagnostic import replay_source_controller
from .pointmaze_goal_validation import _json_ready


PROTOCOL_VERSION = "pointmaze_onecheck_advantage_stage15_v1_development"


def select_checks(*, root, seed, schedule, period, stride, deadline, count):
    bins = np.arange(1, len(schedule) - 1)
    if not 1 <= count <= len(bins):
        raise ValueError("one-check pair count exceeds eligible bins")
    rng = np.random.default_rng(np.random.SeedSequence([root, seed, 15_015]))
    selected = sorted(map(int, rng.choice(bins, size=count, replace=False)))
    return tuple(
        index * period + int(rng.integers(
            0, min(schedule[index] % period, deadline - stride) // stride + 1,
        )) * stride
        for index in selected
    )


def advantage_matrix(rows):
    names, values = _base_matrix(rows)
    matrix, feature_names = _causal_validity_interactions(names, values)
    clocks = np.asarray([[r["offset_fraction"], r["remaining_fraction"]]
                         for r in rows], dtype=np.float64)
    return np.column_stack((matrix, clocks)), [*feature_names, "offset_fraction", "remaining_fraction"]


def predict_advantage(fitted, names, values, offset_fraction, remaining_fraction):
    matrix, feature_names = advantage_matrix([{
        "feature_names": names, "causal_features": values,
        "offset_fraction": offset_fraction, "remaining_fraction": remaining_fraction,
    }])
    if feature_names != fitted["feature_names"]:
        raise ValueError("one-check predictor feature order changed")
    model = fitted["model"]
    x = (matrix[0] - np.asarray(model["feature_mean"])) / np.asarray(model["feature_scale"])
    weights = np.asarray(model["weights"])
    score = float(weights[0] + x @ weights[1:])
    if not np.isfinite(score):
        raise RuntimeError("non-finite one-check advantage")
    return score


def fit_advantage(rows, *, alpha_grid, fit_seeds, eval_seeds):
    observed = {r["seed"] for r in rows}
    if observed != set(fit_seeds) or observed.intersection(eval_seeds):
        raise ValueError("one-check training includes wrong seed roles")
    matrix, names = advantage_matrix(rows)
    labels = np.asarray([r["renew_ise_advantage"] for r in rows])
    _, model = _grouped_ridge_fit_predict(
        matrix, labels, np.asarray([r["seed"] for r in rows]), matrix,
        alpha_grid=alpha_grid,
    )
    return {"model": model, "feature_names": names, "threshold": 0.0,
            "target": "full_episode_ise_wait_one_check_minus_now"}


def require_factual_match(actual, reference):
    if (
        actual["decision_steps"] != reference["decision_steps"]
        or abs(actual["episode_return"] - reference["episode_return"]) > 1e-8
        or abs(actual["tracking_squared_error_integral"]
               - reference["tracking_squared_error_integral"]) > 1e-8
    ):
        raise RuntimeError("one-check factual controller replay differs")


def episode_row(seed, row):
    return {"seed": seed, **{k: row[k] for k in (
        "episode_return", "tracking_squared_error_integral", "decision_steps",
    )}}


def run_cell(args):
    source, controller, time_scale = replay_source_controller(args)
    period = time_scale.upper_period_steps
    reference_predictor = source["trigger_predictor"]
    rollout_options = dict(predictor=reference_predictor, args=args, time_scale=time_scale)
    rows = []
    for seed in args.branch_fit_seeds:
        reference = rollout_intervention(
            controller, seed=seed, intervention_step=None, arm=None, **rollout_options,
        )
        for step in select_checks(
            root=args.optimizer_seed, seed=seed, schedule=reference["decision_steps"],
            period=period, stride=args.check_stride_steps,
            deadline=args.max_offset_steps, count=args.pairs_per_seed,
        ):
            now, wait = [rollout_intervention(
                controller, seed=seed, intervention_step=step, arm=arm,
                **rollout_options,
            ) for arm in ("now", "wait_one_check")]
            if (
                not np.array_equal(now["prefix"], wait["prefix"])
                or not np.array_equal(now["features"], wait["features"])
            ):
                raise RuntimeError("one-check training pair prefix differs")
            factual = now if now["score"] >= reference_predictor["threshold"] else wait
            require_factual_match(factual, reference)
            rows.append({
                "seed": seed, "check_step": step,
                "feature_names": list(now["feature_names"]),
                "causal_features": now["features"].tolist(),
                "offset_fraction": (step % period) / args.max_offset_steps,
                "remaining_fraction": (args.horizon - step) / args.horizon,
                "renew_ise_advantage": (wait["tracking_squared_error_integral"]
                                        - now["tracking_squared_error_integral"]),
                "window_ise_advantage": wait["window_ise"] - now["window_ise"],
                "reference_score": now["score"],
                "prefix_max_abs_difference": 0.0,
                "now_call_step": now["decision_steps"][step // period],
                "wait_call_step": wait["decision_steps"][step // period],
                "now_upper_call_count": len(now["decision_steps"]),
                "wait_upper_call_count": len(wait["decision_steps"]),
            })
    fitted = fit_advantage(rows, alpha_grid=args.ridge_alpha_grid,
                           fit_seeds=args.branch_fit_seeds, eval_seeds=args.trigger_eval_seeds)

    def score_fn(names, values, offset, remaining):
        return predict_advantage(fitted, names, values, offset, remaining)

    candidates, reference_rows = [], []
    for saved in source["aligned_candidate_rows"]:
        seed = saved["seed"]
        reference = rollout_intervention(
            controller, seed=seed, intervention_step=None, arm=None, **rollout_options,
        )
        require_factual_match(reference, saved)
        candidate = rollout_intervention(
            controller, seed=seed, intervention_step=None, arm=None,
            predictor={"threshold": 0.0}, args=args, time_scale=time_scale,
            score_fn=score_fn,
        )
        reference_rows.append(episode_row(seed, reference))
        candidates.append(episode_row(seed, candidate))
    return {
        "optimizer_seed": args.optimizer_seed,
        "selected_checkpoint_iteration": source["controller_selected_iteration"],
        "replayed_controller_training_iterations": args.iterations,
        "horizon": args.horizon,
        "inherited_branch_fit_primitive_steps": source["branch_fit_primitive_steps_replayed"],
        "branch_fit_seeds": args.branch_fit_seeds,
        "trigger_eval_seeds": args.trigger_eval_seeds,
        "pairs_per_seed": args.pairs_per_seed,
        "branch_fit_rows": rows,
        "predictor": fitted,
        "fixed_replay_rows": source["fixed_replay_rows"],
        "reference_rows": reference_rows,
        "candidate_rows": candidates,
        "additional_primitive_steps": {
            "reference_fit_trajectories": len(args.branch_fit_seeds) * args.horizon,
            "paired_fit_replay": len(rows) * 2 * args.horizon,
            "held_out_reference_replay": len(reference_rows) * args.horizon,
            "held_out_candidate_evaluation": len(candidates) * args.horizon,
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
            "protocol_version": PROTOCOL_VERSION,
            "source_result": str(args.source_result),
            "optimizer_seed": args.optimizer_seed,
            "pairs_per_seed": args.pairs_per_seed,
            "threshold": 0.0,
            "evidence_role": "one_step_policy_improvement_development_only",
        },
        "cells": [] if args.dry_run else [run_cell(args)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
