#!/usr/bin/env python3
"""Root-level closed-loop analysis for the frozen Stage-9 development matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.pointmaze_budgeted_trigger import (  # noqa: E402
    POINTMAZE_BUDGETED_TRIGGER_ALGORITHM_PATH,
    POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION,
    PREDICTOR_MODES,
    TRIGGER_MODES,
)
from freq_hrl.experiments.pointmaze_plan_validity_branching import (  # noqa: E402
    BRANCH_CATEGORIES,
)
from scripts.analyze_pointmaze_plan_validity_stage8b import _interval  # noqa: E402
from scripts import pointmaze_budgeted_trigger_stage9_spec as spec  # noqa: E402


SEED_FIELDS = {
    "train": "train_seeds",
    "selection": "selection_seeds",
    "branch_fit": "branch_fit_seeds",
    "trigger_eval": "trigger_eval_seeds",
}
PROTOCOL_OPTIONS = {
    "env_id": "environment_id",
    "iterations": "iterations",
    "horizon": "horizon",
    "upper_period_seconds": "upper_period_seconds",
    "history_seconds": "history_seconds",
    "fast_period_seconds": "fast_period_seconds",
    "maximum_subgoal_delta": "maximum_subgoal_delta",
    "reference_hidden_dim": "reference_hidden_dim",
    "learning_rate": "learning_rate",
    "checkpoint_evaluation_interval": "checkpoint_evaluation_interval",
    "branch_window_seconds": "branch_window_seconds",
    "max_events_per_class": "max_events_per_class",
    "ridge_alpha_grid": "ridge_alpha_grid",
    "threshold_quantile": "threshold_quantile",
    "max_offset_steps": "max_offset_steps",
    "check_stride_steps": "check_stride_steps",
    "methods": "methods",
}
TASK_OPTIONS = {
    "regime_dwell_seconds",
    "target_speed_modes",
    "force_pulse_amplitude",
    "force_pulse_duration_seconds",
    "force_pulse_gap_seconds",
    "distractor_amplitude",
    "distractor_dwell_seconds",
}


def _validate_cell(
    payload: dict[str, Any],
    *,
    preflight: bool,
) -> tuple[int, dict[str, float]]:
    protocol = payload["protocol"]
    cell = payload["cells"][0]
    root = int(cell["optimizer_seed"])
    options = spec.cell_options(root, preflight=preflight)
    if (
        payload.get("status") != "complete"
        or protocol.get("protocol_version") != spec.PROTOCOL
        or protocol.get("algorithm_path")
        != POINTMAZE_BUDGETED_TRIGGER_ALGORITHM_PATH
        or protocol.get("optimizer_seed") != root
        or cell.get("protocol_version") != spec.PROTOCOL
        or cell.get("algorithm_path")
        != POINTMAZE_BUDGETED_TRIGGER_ALGORITHM_PATH
        or cell.get("evidence_role")
        != "budgeted_closed_loop_trigger_development"
        or cell.get("training_schedule") != "balanced_jitter"
        or cell.get("controller_training_schedule") != "balanced_jitter"
    ):
        raise ValueError(f"Stage-9 result contract changed: {root}")
    for key, field in PROTOCOL_OPTIONS.items():
        expected = list(options[key]) if isinstance(options[key], tuple) else options[key]
        if protocol.get(field) != expected:
            raise ValueError(f"Stage-9 frozen option changed: {(root, key)}")
    for key in TASK_OPTIONS:
        expected = list(options[key]) if isinstance(options[key], tuple) else options[key]
        if protocol["task_options"].get(key) != expected:
            raise ValueError(f"Stage-9 task option changed: {(root, key)}")
    if any(
        tuple(map(int, cell.get(field, []))) != options[role]
        or tuple(map(int, protocol.get(field, []))) != options[role]
        for role, field in SEED_FIELDS.items()
    ):
        raise ValueError(f"Stage-9 seed roles changed: {root}")
    if any(
        cell.get("runtime_versions", {}).get(name) != version
        for name, version in spec.RUNTIME_EXPECTATIONS.items()
    ):
        raise ValueError(f"Stage-9 runtime changed: {root}")
    if (
        cell["config"]["device"] != "cpu"
        or cell["capacity"]["actual_parameter_count"] != 267018
        or cell["gradient_updates_train"] <= 0
        or cell["trigger_training"]
        != "branch_supervised_ridge_with_grouped_fit_cv"
    ):
        raise ValueError(f"Stage-9 controller training is invalid: {root}")

    fit = cell["branch_fit_rows"]
    fit_seeds = set(options["branch_fit"])
    expected_per_class = int(options["max_events_per_class"])
    counts = Counter((int(row["seed"]), row["category"]) for row in fit)
    if (
        len(fit) != len(fit_seeds) * len(BRANCH_CATEGORIES) * expected_per_class
        or set(counts.values()) != {expected_per_class}
        or set(counts) != {
            (seed, category)
            for seed in fit_seeds for category in BRANCH_CATEGORIES
        }
        or any(
            not row["protocol_valid"]
            or row["prefix_max_abs_difference"] != 0
            or row["feature_max_abs_difference"] != 0
            or row["privileged_regime_context_present"]
            or "oracle_regime_context" in row
            or row["keep_upper_calls_at_opportunity"] != 0
            or row["renew_upper_calls_at_opportunity"] != 1
            or row["downstream_upper_call_count_per_branch"] != 0
            or not row["lower_controller_remains_closed_loop"]
            for row in fit
        )
        or sum(
            row["keep_primitive_steps_replayed"]
            + row["renew_primitive_steps_replayed"]
            for row in fit
        ) != cell["branch_fit_primitive_steps_replayed"]
    ):
        raise ValueError(f"Stage-9 branch supervision is invalid: {root}")
    predictors = cell["trigger_predictors"]
    if set(predictors) != set(PREDICTOR_MODES):
        raise ValueError(f"Stage-9 predictor suite changed: {root}")
    for name in PREDICTOR_MODES:
        item = predictors[name]
        model = item["model"]
        count = 39 if name == "causal_validity_interactions" else 170
        if (
            len(item["feature_names"]) != count
            or model["group_count"] != len(fit_seeds)
            or model["alpha_grid"] != list(spec.RIDGE_ALPHA_GRID)
            or model["alpha"] not in spec.RIDGE_ALPHA_GRID
            or item["threshold_quantile"] != spec.THRESHOLD_QUANTILE
            or item["threshold_source"]
            != "branch_fit_leave_one_path_out_predictions"
            or not np.isfinite(float(item["threshold"]))
            or any(
                token in feature.lower()
                for feature in item["feature_names"]
                for token in ("regime", "oracle", "distractor", "event")
            )
        ):
            raise ValueError(f"Stage-9 fitted predictor is invalid: {(root, name)}")

    eval_seeds = set(options["trigger_eval"])
    rows = cell["trigger_evaluation_rows"]
    by_key = {(int(row["seed"]), row["mode"]): row for row in rows}
    expected = {(seed, mode) for seed in eval_seeds for mode in TRIGGER_MODES}
    if len(by_key) != len(rows) or set(by_key) != expected:
        raise ValueError(f"Stage-9 paired evaluation is incomplete: {root}")
    period = int(round(spec.UPPER_PERIOD_SECONDS / 0.01))
    horizon = int(options["horizon"])
    for (seed, mode), row in by_key.items():
        decisions = row["decision_steps"]
        durations = np.diff([*decisions, horizon])
        if (
            not row["protocol_valid"]
            or row["has_privileged_regime_input"]
            or not row["planner_called_only_on_decision"]
            or row["upper_decision_count"] != horizon // period
            or row["fixed_budget_upper_decision_count"] != horizon // period
            or row["episode_length"] != horizon
            or decisions[0] != 0
            or [step // period for step in decisions]
            != list(range(horizon // period))
            or np.min(durations) < period - spec.MAX_OFFSET_STEPS
            or np.max(durations) > period + spec.MAX_OFFSET_STEPS
            or int(np.sum(durations)) != horizon
            or not np.isfinite(float(row["episode_return"]))
            or not np.isfinite(float(row["tracking_squared_error_integral"]))
            or (mode in PREDICTOR_MODES and row["trigger_score_checks"] < 1)
        ):
            raise ValueError(f"Stage-9 episode accounting is invalid: {(root, seed, mode)}")

    trained = {int(row["seed"]): row for row in cell["canonical_evaluation_rows"]}
    untrained = {int(row["seed"]): row for row in cell["untrained_evaluation_rows"]}
    if not eval_seeds.issubset(trained) or not eval_seeds.issubset(untrained):
        raise ValueError(f"Stage-9 controller evaluation is incomplete: {root}")
    learning = float(np.mean([
        untrained[seed]["tracking_squared_error_integral"]
        - trained[seed]["tracking_squared_error_integral"]
        for seed in eval_seeds
    ]))
    fixed = "fixed"
    candidate = "causal_validity_interactions"
    current = "current_compact_quadratic"
    jitter = "balanced_jitter"
    return root, {
        "controller_learning_gain": learning,
        "candidate_minus_fixed_episode_tracking_ise": float(np.mean([
            by_key[(seed, fixed)]["tracking_squared_error_integral"]
            - by_key[(seed, candidate)]["tracking_squared_error_integral"]
            for seed in eval_seeds
        ])),
        "candidate_minus_jitter_episode_tracking_ise": float(np.mean([
            by_key[(seed, jitter)]["tracking_squared_error_integral"]
            - by_key[(seed, candidate)]["tracking_squared_error_integral"]
            for seed in eval_seeds
        ])),
        "candidate_minus_current_episode_tracking_ise": float(np.mean([
            by_key[(seed, current)]["tracking_squared_error_integral"]
            - by_key[(seed, candidate)]["tracking_squared_error_integral"]
            for seed in eval_seeds
        ])),
        "candidate_minus_fixed_episode_return": float(np.mean([
            by_key[(seed, candidate)]["episode_return"]
            - by_key[(seed, fixed)]["episode_return"]
            for seed in eval_seeds
        ])),
        "candidate_early_calls_per_episode": float(np.mean([
            by_key[(seed, candidate)]["trigger_early_calls"]
            for seed in eval_seeds
        ])),
    }


def analyze(paths: Iterable[Path]) -> dict[str, Any]:
    inputs = tuple(paths)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    if not payloads or any(len(p.get("cells", [])) != 1 for p in payloads):
        raise ValueError("Stage-9 requires one complete cell per result")
    roots = [int(p["cells"][0]["optimizer_seed"]) for p in payloads]
    if set(roots) == set(spec.PREFLIGHT_OPTIMIZER_SEEDS):
        expected_roots = spec.PREFLIGHT_OPTIMIZER_SEEDS
        preflight = True
    elif set(roots) == set(spec.OPTIMIZER_SEEDS):
        expected_roots = spec.OPTIMIZER_SEEDS
        preflight = False
    else:
        raise ValueError("Stage-9 registered root matrix is incomplete")
    if len(roots) != len(expected_roots):
        raise ValueError("Stage-9 optimizer roots are duplicated")
    summaries = dict(_validate_cell(p, preflight=preflight) for p in payloads)
    intervals = {
        name: _interval((row[name] for row in summaries.values()), confidence=0.95)
        for name in next(iter(summaries.values()))
    }
    checks = {
        "controller_learned": intervals["controller_learning_gain"]["status"] == "supported",
        "candidate_beats_fixed_ise": intervals["candidate_minus_fixed_episode_tracking_ise"]["status"] == "supported",
        "candidate_beats_random_ise": intervals["candidate_minus_jitter_episode_tracking_ise"]["status"] == "supported",
        "candidate_beats_current_only_ise": intervals["candidate_minus_current_episode_tracking_ise"]["status"] == "supported",
        "candidate_return_beats_fixed": intervals["candidate_minus_fixed_episode_return"]["status"] == "supported",
    }
    return {
        "analysis_version": "pointmaze_budgeted_trigger_stage9_analysis_v1",
        "protocol_version": POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION,
        "matrix": "preflight" if preflight else "formal_development",
        "optimizer_root_count": len(summaries),
        "statistical_unit": "optimizer_seed_root",
        "root_summaries": {str(root): summaries[root] for root in sorted(summaries)},
        "intervals": intervals,
        "qualification_checks": checks,
        "decision": (
            "stage9_development_gate_passed"
            if all(checks.values()) else "stage9_development_gate_failed"
        ),
        "claim_boundary": (
            "PointMaze development only; independent confirmation and domain "
            "transfer remain required"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# PointMaze Stage-9 Budgeted Trigger",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Quantity | Root mean [95% CI] | Status |",
        "|---|---:|---|",
    ]
    for name, item in result["intervals"].items():
        lines.append(
            f"| {name} | {item['mean']:.6f} "
            f"[{item['ci_lower']:.6f}, {item['ci_upper']:.6f}] "
            f"| {item['status']} |"
        )
    lines.extend(("", result["claim_boundary"] + ".", ""))
    (args.output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
