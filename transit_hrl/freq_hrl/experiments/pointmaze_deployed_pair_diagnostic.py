"""Paired timing interventions at states visited by the deployed PointMaze trigger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract

from .pointmaze_budgeted_trigger import (
    _predict_renewal_advantage,
    balanced_jitter_schedule,
    build_parser as build_stage9_parser,
)
from .pointmaze_goal_validation import _json_ready
from .pointmaze_plan_validity_branching import _task_options
from .pointmaze_plan_value_qualification import train_pointmaze_plan_value_cell
from .pointmaze_timing_pair import (
    WINDOWED_PROTOCOL_VERSION,
    rollout_timing_schedule,
)


PROTOCOL_VERSION = "pointmaze_deployed_pair_stage13_v1_diagnostic"
SOURCE_RUN = "pointmaze_timing_pair_stage12_v1_{}_20260927_r1"


def select_bins(
    *, root: int, seed: int, decision_steps: list[int], period: int,
    deadline: int, pairs_per_class: int,
) -> tuple[int, ...]:
    if pairs_per_class < 1:
        raise ValueError("pairs per class must be positive")
    eligible = range(1, len(decision_steps) - 1)
    early = [index for index in eligible
             if decision_steps[index] - index * period < deadline]
    late = [index for index in eligible
            if decision_steps[index] - index * period == deadline]
    if len(early) + len(late) != len(eligible):
        raise ValueError("source schedule has a call outside the legal grid")
    rng = np.random.default_rng(np.random.SeedSequence([root, seed, 13_013]))
    selected = [
        *rng.choice(early, size=min(pairs_per_class, len(early)), replace=False),
        *rng.choice(late, size=min(pairs_per_class, len(late)), replace=False),
    ]
    return tuple(sorted(int(index) for index in selected))


def replay_source_controller(args: argparse.Namespace) -> tuple:
    source = json.loads(args.source_result.read_text(encoding="utf-8"))
    if (
        source.get("status") != "complete"
        or source["protocol"]["protocol_version"] != WINDOWED_PROTOCOL_VERSION
        or source["protocol"]["optimizer_seed"] != args.optimizer_seed
        or source["protocol"]["credit_window_steps"] != 50
    ):
        raise ValueError("source is not the completed Stage-12 cell")
    cell = source["cells"][0]
    for role in ("train", "selection", "branch_fit", "trigger_eval"):
        if list(getattr(args, f"{role}_seeds")) != cell[f"{role}_seeds"]:
            raise ValueError(f"{role} seed role differs from Stage-12")
    if (
        source["protocol"]["horizon"] != args.horizon
        or source["protocol"]["iterations"] != args.iterations
        or source["protocol"]["task_options"] != _json_ready(_task_options(args))
    ):
        raise ValueError("controller training options differ from Stage-12")
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=0.01,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
    )
    period = time_scale.upper_period_steps
    if period != 50 or args.max_offset_steps != 25:
        raise ValueError("diagnostic timing protocol changed")

    def training_schedule(seed: int) -> tuple[int, ...]:
        return balanced_jitter_schedule(
            seed=seed, horizon=args.horizon, period_steps=period,
            max_offset_steps=args.max_offset_steps,
        )

    payload, controller = train_pointmaze_plan_value_cell(
        method="hrl_regime_history",
        env_id=args.env_id,
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        eval_seeds=(*args.branch_fit_seeds, *args.trigger_eval_seeds),
        iterations=args.iterations,
        horizon=args.horizon,
        optimizer_seed=args.optimizer_seed,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
        maximum_subgoal_delta=args.maximum_subgoal_delta,
        reference_hidden_dim=args.reference_hidden_dim,
        learning_rate=args.learning_rate,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        waypoint_perturbation=0.25,
        event_window_seconds=1.0,
        task_options=_task_options(args),
        diagnostic_schedules=("fixed",),
        training_decision_steps_fn=training_schedule,
        training_schedule_name="balanced_jitter",
    )
    if payload["selected_checkpoint_iteration"] != cell["controller_selected_iteration"]:
        raise RuntimeError("retrained controller selected a different checkpoint")
    return cell, controller, time_scale


def run_cell(args: argparse.Namespace) -> dict:
    cell, controller, time_scale = replay_source_controller(args)
    period = time_scale.upper_period_steps
    predictor = cell["trigger_predictor"]
    threshold = predictor["threshold"]
    rows = []
    for candidate in cell["aligned_candidate_rows"]:
        seed = candidate["seed"]
        decision_steps = list(candidate["decision_steps"])
        for bin_index in select_bins(
            root=args.optimizer_seed, seed=seed, decision_steps=decision_steps,
            period=period, deadline=args.max_offset_steps,
            pairs_per_class=args.pairs_per_class,
        ):
            offset = decision_steps[bin_index] - bin_index * period
            factual_early = offset < args.max_offset_steps
            check_step = bin_index * period + (offset if factual_early else 0)
            alternate_steps = decision_steps.copy()
            alternate_steps[bin_index] = bin_index * period + (
                args.max_offset_steps if factual_early else 0
            )
            common = dict(
                seed=seed, capture_step=check_step, env_id=args.env_id,
                horizon=args.horizon, time_scale=time_scale,
                maximum_subgoal_delta=args.maximum_subgoal_delta,
                task_options=_task_options(args), credit_window_steps=period,
            )
            factual = rollout_timing_schedule(
                controller, decision_steps=tuple(decision_steps), **common,
            )
            alternate = rollout_timing_schedule(
                controller, decision_steps=tuple(alternate_steps), **common,
            )
            prefix_difference = float(np.max(np.abs(
                factual["prefix_snapshot"] - alternate["prefix_snapshot"]
            )))
            if (
                abs(factual["tracking_squared_error_integral"]
                    - candidate["tracking_squared_error_integral"]) > 1e-8
                or abs(factual["episode_return"] - candidate["episode_return"]) > 1e-8
                or prefix_difference > 1e-10
                or factual["feature_names"] != alternate["feature_names"]
                or not np.array_equal(
                    factual["causal_features"], alternate["causal_features"]
                )
            ):
                raise RuntimeError("deployed paired replay is not causally matched")
            score = _predict_renewal_advantage(
                feature_names=factual["feature_names"],
                features=factual["causal_features"],
                predictor="causal_validity_interactions", fitted=predictor,
            )
            if (score >= threshold) != factual_early:
                raise RuntimeError("saved deployed decision disagrees with fitted score")
            early = factual if factual_early else alternate
            late = alternate if factual_early else factual
            rows.append({
                "seed": seed,
                "bin_index": bin_index,
                "check_step": check_step,
                "factual_offset": offset,
                "factual_early": factual_early,
                "predicted_ise_advantage": score,
                "threshold": threshold,
                "early_minus_late_ise_advantage_50_steps": (
                    late["credit_window_squared_error_integral"]
                    - early["credit_window_squared_error_integral"]
                ),
                "early_minus_late_ise_advantage_full_episode": (
                    late["tracking_squared_error_integral"]
                    - early["tracking_squared_error_integral"]
                ),
                "prefix_max_abs_difference": prefix_difference,
                "pair_primitive_steps_replayed": 2 * args.horizon,
            })
    return {
        "optimizer_seed": args.optimizer_seed,
        "selected_checkpoint_iteration": cell["controller_selected_iteration"],
        "paths": len(cell["aligned_candidate_rows"]),
        "pairs": len(rows),
        "pair_primitive_steps_replayed": sum(
            row["pair_primitive_steps_replayed"] for row in rows
        ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_stage9_parser()
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--pairs-per-class", type=int, default=2)
    args = parser.parse_args(argv)
    output = {
        "status": "dry_run" if args.dry_run else "complete",
        "protocol": {
            "protocol_version": PROTOCOL_VERSION,
            "source_protocol": WINDOWED_PROTOCOL_VERSION,
            "source_result": str(args.source_result),
            "optimizer_seed": args.optimizer_seed,
            "pairs_per_class": args.pairs_per_class,
            "continuation": "factual_deployed_static_schedule_after_one_bin_flip",
            "evidence_role": "mechanism_diagnostic_only",
        },
        "cells": [] if args.dry_run else [run_cell(args)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
