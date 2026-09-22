#!/usr/bin/env python3
"""Root-paired analysis for Stage-8 plan-value task qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.pointmaze_plan_value_qualification import (  # noqa: E402
    POINTMAZE_PLAN_VALUE_ALGORITHM_PATH,
    POINTMAZE_PLAN_VALUE_METHODS,
    POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
    POINTMAZE_PLAN_VALUE_SCHEDULES,
)


METRICS = {
    "tracking_squared_error_integral": False,
    "episode_return": True,
    "tracking_success_rate": True,
    "event_post_tracking_mse": False,
    "event_recovery_seconds_mean": False,
}
CONTRASTS = {
    "plan_refresh_vs_stale": (
        ("hrl_regime_history", "fixed"),
        ("hrl_regime_history", "stale_plan"),
    ),
    "plan_integrity_vs_perturbed": (
        ("hrl_regime_history", "fixed"),
        ("hrl_regime_history", "fixed_waypoint_perturbed"),
    ),
    "current_regime_information": (
        ("hrl_regime_oracle_context", "fixed"),
        ("hrl_regime_history", "fixed"),
    ),
    "oracle_timing_same_budget": (
        ("hrl_regime_oracle_context", "oracle_event_delay_000ms"),
        ("hrl_regime_oracle_context", "fixed"),
    ),
    "oracle_timing_100ms_delay_cost": (
        ("hrl_regime_oracle_context", "oracle_event_delay_000ms"),
        ("hrl_regime_oracle_context", "oracle_event_delay_100ms"),
    ),
    "oracle_timing_250ms_delay_cost": (
        ("hrl_regime_oracle_context", "oracle_event_delay_000ms"),
        ("hrl_regime_oracle_context", "oracle_event_delay_250ms"),
    ),
    "oracle_timing_500ms_delay_cost": (
        ("hrl_regime_oracle_context", "oracle_event_delay_000ms"),
        ("hrl_regime_oracle_context", "oracle_event_delay_500ms"),
    ),
    "combined_oracle_reference": (
        ("hrl_regime_oracle_context", "oracle_event_delay_000ms"),
        ("hrl_regime_history", "fixed"),
    ),
}


def _interval(
    values: Iterable[float],
    *,
    confidence: float,
    classify: bool,
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("Stage-8 analysis requires finite root values")
    mean = float(np.mean(array))
    if array.size < 2:
        lower, upper = float("-inf"), float("inf")
    else:
        standard_error = float(stats.sem(array))
        if standard_error <= 1e-15:
            lower = upper = mean
        else:
            critical = float(stats.t.ppf(
                0.5 + float(confidence) / 2.0,
                df=array.size - 1,
            ))
            lower = mean - critical * standard_error
            upper = mean + critical * standard_error
    result = {
        "n": int(array.size),
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": float(confidence),
    }
    if classify:
        result["status"] = (
            "supported"
            if lower > 0.0
            else "contradicted" if upper < 0.0 else "inconclusive"
        )
    return result


def _cell_identity(cell: dict[str, Any]) -> tuple[str, int]:
    method = str(cell.get("policy", ""))
    root = int(cell.get("optimizer_seed", -1))
    if method not in POINTMAZE_PLAN_VALUE_METHODS or root < 0:
        raise ValueError("Stage-8 cell has an invalid identity")
    return method, root


def _validate_cells(cells: list[dict[str, Any]]) -> tuple[int, ...]:
    if not cells:
        raise ValueError("Stage-8 analysis requires cells")
    identities: set[tuple[str, int]] = set()
    runtime_contracts: set[str] = set()
    role_contracts: dict[int, tuple[tuple[int, ...], ...]] = {}
    budgets: dict[int, int] = {}
    for cell in cells:
        method, root = _cell_identity(cell)
        if (method, root) in identities:
            raise ValueError(f"duplicate Stage-8 cell: {(method, root)}")
        identities.add((method, root))
        if (
            cell.get("protocol_version")
            != POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION
            or cell.get("algorithm_path")
            != POINTMAZE_PLAN_VALUE_ALGORITHM_PATH
            or cell.get("evidence_role")
            != "task_qualification_development"
            or cell.get("belief_training")
            != "disabled_not_yet_authorized"
            or cell.get("trigger_training")
            != "disabled_not_yet_authorized"
        ):
            raise ValueError(f"Stage-8 cell contract mismatch: {(method, root)}")
        dimensions = cell.get("dimensions", {})
        expected_upper = (
            int(dimensions.get("oracle_upper", -1))
            if method == "hrl_regime_oracle_context"
            else int(dimensions.get("base_upper", -1))
        )
        config = cell.get("config", {})
        if (
            expected_upper < 1
            or int(config.get("upper_state_dim", -1)) != expected_upper
            or int(config.get("lower_state_dim", -1))
            != int(dimensions.get("lower", -2))
        ):
            raise ValueError(f"Stage-8 state shape changed: {(method, root)}")
        capacity = cell.get("capacity", {})
        budget = int(capacity.get("reference_parameter_budget", -1))
        actual = int(capacity.get("actual_parameter_count", -1))
        ratio = float(capacity.get("parameter_budget_ratio", float("nan")))
        if budget < 1 or actual < 1 or not np.isfinite(ratio) or abs(ratio - 1.0) > 0.08:
            raise ValueError(f"Stage-8 capacity mismatch: {(method, root)}")
        previous_budget = budgets.setdefault(root, budget)
        if previous_budget != budget:
            raise ValueError(f"Stage-8 methods have unequal budget: root {root}")
        runtime = cell.get("runtime_versions")
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError(f"Stage-8 runtime missing: {(method, root)}")
        runtime_contracts.add(json.dumps(runtime, sort_keys=True))
        roles = tuple(
            tuple(map(int, cell.get(name, [])))
            for name in ("train_seeds", "selection_seeds", "eval_seeds")
        )
        if any(not values for values in roles):
            raise ValueError(f"Stage-8 seed role missing: {(method, root)}")
        previous_roles = role_contracts.setdefault(root, roles)
        if roles != previous_roles:
            raise ValueError(f"Stage-8 methods are not seed paired: root {root}")
    if len(runtime_contracts) != 1:
        raise ValueError("Stage-8 runtime versions differ")
    roots = tuple(sorted({root for _, root in identities}))
    expected = {
        (method, root)
        for root in roots
        for method in POINTMAZE_PLAN_VALUE_METHODS
    }
    if identities != expected:
        raise ValueError("Stage-8 method/root matrix is incomplete")
    return roots


def _index_rows(
    cells: Iterable[dict[str, Any]],
    *,
    field: str,
    schedules: tuple[str, ...],
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for cell in cells:
        method, root = _cell_identity(cell)
        rows = cell.get(field)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Stage-8 cell is missing {field}: {(method, root)}")
        expected_seeds = tuple(map(int, cell["eval_seeds"]))
        expected = {
            (schedule, seed)
            for schedule in schedules
            for seed in expected_seeds
        }
        observed: set[tuple[str, int]] = set()
        for row in rows:
            schedule = str(row.get("schedule_mode", ""))
            seed = int(row.get("seed", -1))
            identity = (schedule, seed)
            if identity not in expected or identity in observed:
                raise ValueError(
                    f"Stage-8 row matrix mismatch: {(method, root, identity)}"
                )
            observed.add(identity)
            key = (method, schedule, root, seed)
            if (
                int(row.get("training_replicate_seed", -1)) != root
                or row.get("method") != method
                or row.get("protocol_version")
                != POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION
                or row.get("algorithm_path")
                != POINTMAZE_PLAN_VALUE_ALGORITHM_PATH
                or float(row.get("protocol_valid", 0.0)) != 1.0
                or bool(row.get("policy_has_future_regime_access", True))
                or bool(row.get("regime_label_visible_to_candidate", True))
                or bool(row.get("external_future_visible_to_actor", True))
                or not bool(row.get("external_stream_action_independent", False))
            ):
                raise ValueError(f"Stage-8 row contract invalid: {key}")
            expected_context = method == "hrl_regime_oracle_context"
            expected_oracle_schedule = schedule.startswith("oracle_event_")
            if (
                bool(row.get("policy_has_current_regime_access"))
                != expected_context
                or bool(row.get("schedule_has_regime_event_access"))
                != expected_oracle_schedule
                or bool(row.get("schedule_has_future_regime_access"))
                != expected_oracle_schedule
            ):
                raise ValueError(f"Stage-8 oracle boundary invalid: {key}")
            budget_matched = float(row.get("planning_budget_matched", -1.0))
            if schedule == "stale_plan":
                if budget_matched != 0.0 or int(row["upper_decision_count"]) != 1:
                    raise ValueError(f"Stage-8 stale-plan contract invalid: {key}")
            elif (
                budget_matched != 1.0
                or int(row["upper_decision_count"])
                != int(row["fixed_budget_upper_decision_count"])
            ):
                raise ValueError(f"Stage-8 budget mismatch: {key}")
            if not all(np.isfinite(float(row[metric])) for metric in METRICS):
                raise ValueError(f"Stage-8 metric is not finite: {key}")
            indexed[key] = row
        if observed != expected:
            raise ValueError(f"Stage-8 row matrix is incomplete: {(method, root)}")
    return indexed


def _validate_pairing(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    schedules: tuple[str, ...],
) -> None:
    path_keys = (
        "target_start_vertex",
        "target_speed_modes_world_per_second",
        "regime_change_steps",
        "force_pulse_start_steps",
        "distractor_change_steps",
    )
    roots_and_seeds = {
        (root, seed) for _, _, root, seed in indexed
    }
    for root, seed in roots_and_seeds:
        reference = indexed[
            (POINTMAZE_PLAN_VALUE_METHODS[0], schedules[0], root, seed)
        ]
        path = {key: reference.get(key) for key in path_keys}
        for method in POINTMAZE_PLAN_VALUE_METHODS:
            for schedule in schedules:
                row = indexed[(method, schedule, root, seed)]
                if {key: row.get(key) for key in path_keys} != path:
                    raise ValueError(
                        f"Stage-8 exogenous path is not paired: {(root, seed)}"
                    )


def _root_means(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    method: str,
    schedule: str,
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for (row_method, row_schedule, root, _), row in indexed.items():
        if row_method == method and row_schedule == schedule:
            grouped.setdefault(root, []).append(float(row[metric]))
    return {root: float(np.mean(values)) for root, values in grouped.items()}


def _paired_effect(
    candidate: dict[int, float],
    baseline: dict[int, float],
    *,
    higher_is_better: bool,
    confidence: float,
) -> dict[str, Any]:
    if set(candidate) != set(baseline):
        raise ValueError("Stage-8 contrast is not optimizer-root paired")
    sign = 1.0 if higher_is_better else -1.0
    result = _interval(
        (
            sign * (candidate[root] - baseline[root])
            for root in sorted(candidate)
        ),
        confidence=confidence,
        classify=True,
    )
    result["mean_improvement"] = result.pop("mean")
    return result


def analyze_stage8(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    items = list(cells)
    roots = _validate_cells(items)
    rows = _index_rows(
        items,
        field="evaluation_rows",
        schedules=POINTMAZE_PLAN_VALUE_SCHEDULES,
    )
    _validate_pairing(rows, schedules=POINTMAZE_PLAN_VALUE_SCHEDULES)
    untrained = _index_rows(
        items,
        field="untrained_evaluation_rows",
        schedules=("fixed",),
    )
    _validate_pairing(untrained, schedules=("fixed",))

    root_means = {
        method: {
            schedule: {
                metric: _root_means(
                    rows,
                    method=method,
                    schedule=schedule,
                    metric=metric,
                )
                for metric in METRICS
            }
            for schedule in POINTMAZE_PLAN_VALUE_SCHEDULES
        }
        for method in POINTMAZE_PLAN_VALUE_METHODS
    }
    absolute = {
        method: {
            schedule: {
                metric: _interval(
                    values.values(), confidence=confidence, classify=False
                )
                for metric, values in metrics.items()
            }
            for schedule, metrics in schedules.items()
        }
        for method, schedules in root_means.items()
    }
    contrasts: dict[str, Any] = {}
    for name, (candidate, baseline) in CONTRASTS.items():
        candidate_method, candidate_schedule = candidate
        baseline_method, baseline_schedule = baseline
        contrasts[name] = {
            "candidate": {
                "method": candidate_method,
                "schedule": candidate_schedule,
            },
            "baseline": {
                "method": baseline_method,
                "schedule": baseline_schedule,
            },
            **{
                metric: _paired_effect(
                    root_means[candidate_method][candidate_schedule][metric],
                    root_means[baseline_method][baseline_schedule][metric],
                    higher_is_better=higher_is_better,
                    confidence=confidence,
                )
                for metric, higher_is_better in METRICS.items()
            },
        }

    learning_gain: dict[str, Any] = {}
    for method in POINTMAZE_PLAN_VALUE_METHODS:
        final = root_means[method]["fixed"]
        initial = {
            metric: _root_means(
                untrained,
                method=method,
                schedule="fixed",
                metric=metric,
            )
            for metric in METRICS
        }
        learning_gain[method] = {
            metric: _paired_effect(
                final[metric],
                initial[metric],
                higher_is_better=higher_is_better,
                confidence=confidence,
            )
            for metric, higher_is_better in METRICS.items()
        }

    endpoint = "tracking_squared_error_integral"
    checks = {
        "history_controller_learned": (
            learning_gain["hrl_regime_history"][endpoint]["status"]
            == "supported"
        ),
        "plan_refresh_has_value": (
            contrasts["plan_refresh_vs_stale"][endpoint]["status"]
            == "supported"
        ),
        "plan_content_has_value": (
            contrasts["plan_integrity_vs_perturbed"][endpoint]["status"]
            == "supported"
        ),
        "current_regime_information_has_value": (
            contrasts["current_regime_information"][endpoint]["status"]
            == "supported"
        ),
        "same_budget_event_timing_has_value": (
            contrasts["oracle_timing_same_budget"][endpoint]["status"]
            == "supported"
        ),
        "quarter_second_delay_has_cost": (
            contrasts["oracle_timing_250ms_delay_cost"][endpoint]["status"]
            == "supported"
        ),
    }
    distinguishability = _interval(
        (
            float(np.mean([
                float(row[
                    "causal_distinguishability_delay_seconds_max"
                ])
                for (method, schedule, row_root, _), row in rows.items()
                if method == "hrl_regime_history"
                and schedule == "fixed"
                and row_root == root
            ]))
            for root in roots
        ),
        confidence=confidence,
        classify=False,
    )
    checks["causal_observation_precedes_quarter_second_cost"] = bool(
        distinguishability["ci_upper"] < 0.25
        and checks["quarter_second_delay_has_cost"]
    )
    return {
        "analysis_version": "pointmaze_plan_value_stage8_analysis_v1",
        "protocol_version": POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "evaluation_row_count": len(rows),
        "untrained_evaluation_row_count": len(untrained),
        "independent_training_replicate_count": len(roots),
        "statistical_unit": "optimizer_seed_root",
        "primary_endpoint": endpoint,
        "absolute": absolute,
        "learning_gain_vs_untrained": learning_gain,
        "contrasts": contrasts,
        "qualification_checks": checks,
        "causal_distinguishability_delay_seconds_max": distinguishability,
        "stage9_authorized": bool(all(checks.values())),
        "decision": (
            "task_qualified_for_stage9"
            if all(checks.values())
            else "stage9_not_authorized"
        ),
        "root_means": {
            method: {
                schedule: {
                    metric: {
                        str(root): value for root, value in values.items()
                    }
                    for metric, values in metrics.items()
                }
                for schedule, metrics in schedules.items()
            }
            for method, schedules in root_means.items()
        },
    }


def load_cells(paths: Iterable[Path]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete Stage-8 result: {path}")
        protocol = payload.get("protocol", {})
        if (
            protocol.get("protocol_version")
            != POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION
            or protocol.get("algorithm_path")
            != POINTMAZE_PLAN_VALUE_ALGORITHM_PATH
        ):
            raise ValueError(f"wrong Stage-8 protocol: {path}")
        payload_cells = payload.get("cells", [])
        if len(payload_cells) != 1:
            raise ValueError(f"Stage-8 result must contain one cell: {path}")
        cells.extend(payload_cells)
    return cells


def render_report(analysis: dict[str, Any]) -> str:
    endpoint = analysis["primary_endpoint"]
    lines = [
        "# PointMaze Plan-Value Stage-8 Qualification",
        "",
        f"Protocol: `{analysis['protocol_version']}`",
        (
            "Independent optimizer roots: "
            f"{analysis['independent_training_replicate_count']}"
        ),
        f"Decision: **{analysis['decision']}**",
        "",
        "| Qualification contrast | ISE improvement [95% CI] | Status |",
        "|---|---:|---|",
    ]
    for name, row in analysis["contrasts"].items():
        effect = row[endpoint]
        lines.append(
            f"| {name} | {effect['mean_improvement']:.4f} "
            f"[{effect['ci_lower']:.4f}, {effect['ci_upper']:.4f}] "
            f"| {effect['status']} |"
        )
    lines.extend([
        "",
        (
            "Oracle schedules are privileged task-qualification references, "
            "not deployable candidates. Non-stale schedule contrasts preserve "
            "the fixed upper-call budget."
        ),
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_stage8(
        load_cells(args.inputs), confidence=args.confidence
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(analysis), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
