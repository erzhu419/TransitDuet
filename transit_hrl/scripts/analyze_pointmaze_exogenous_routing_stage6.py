#!/usr/bin/env python3
"""Root-paired analysis for PointMaze external-frequency attribution."""

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

from freq_hrl.experiments.pointmaze_exogenous_routing_attribution import (  # noqa: E402
    POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
    POINTMAZE_EXOGENOUS_ROUTING_METHOD_SPECS,
    POINTMAZE_EXOGENOUS_ROUTING_METHODS,
    POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
    POINTMAZE_EXOGENOUS_ROUTING_SHAPE_CONTRACT,
)


MIN_ROUTED_TRACKING_SUCCESS = 0.50
METRICS = {
    "tracking_success_rate": True,
    "episode_return": True,
    "tracking_rmse": False,
    "final_tracking_distance": False,
}
CONTRASTS = {
    "flat_filtered_vs_history": (
        "flat_exogenous_filtered", "flat_exogenous_history"
    ),
    "flat_multiscale_vs_history": (
        "flat_exogenous_multiscale_all", "flat_exogenous_history"
    ),
    "hrl_filtered_vs_history": (
        "hrl_exogenous_filtered", "hrl_exogenous_history"
    ),
    "hrl_multiscale_vs_history": (
        "hrl_exogenous_multiscale_all", "hrl_exogenous_history"
    ),
    "routed_vs_history": (
        "hrl_exogenous_multiscale_routed", "hrl_exogenous_history"
    ),
    "routed_vs_filtered": (
        "hrl_exogenous_multiscale_routed", "hrl_exogenous_filtered"
    ),
    "routed_vs_all_band": (
        "hrl_exogenous_multiscale_routed", "hrl_exogenous_multiscale_all"
    ),
    "routed_vs_swapped": (
        "hrl_exogenous_multiscale_routed", "hrl_exogenous_multiscale_swapped"
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
        raise ValueError("Stage-6 analysis requires finite root values")
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
    if method not in POINTMAZE_EXOGENOUS_ROUTING_METHODS or root < 0:
        raise ValueError("Stage-6 cell has an invalid identity")
    return method, root


def _validate_cells(
    cells: list[dict[str, Any]],
    *,
    methods: tuple[str, ...] = POINTMAZE_EXOGENOUS_ROUTING_METHODS,
) -> tuple[int, ...]:
    if not cells:
        raise ValueError("Stage-6 analysis requires cells")
    identities: set[tuple[str, int]] = set()
    runtime_contracts: set[str] = set()
    role_contracts: dict[int, tuple[tuple[int, ...], ...]] = {}
    capacity_contracts: dict[str, tuple[int, int, int]] = {}
    for cell in cells:
        method, root = _cell_identity(cell)
        if (method, root) in identities:
            raise ValueError(f"duplicate Stage-6 cell: {(method, root)}")
        identities.add((method, root))
        architecture, representation = POINTMAZE_EXOGENOUS_ROUTING_METHOD_SPECS[
            method
        ]
        if (
            cell.get("protocol_version")
            != POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION
            or cell.get("algorithm_path")
            != POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH
            or cell.get("architecture") != architecture
            or cell.get("representation") != representation
            or cell.get("routing_shape_contract")
            != POINTMAZE_EXOGENOUS_ROUTING_SHAPE_CONTRACT
        ):
            raise ValueError(f"Stage-6 cell contract mismatch: {(method, root)}")
        if any(cell.get(name) != "disabled" for name in (
            "projector", "promotion", "leakage_loss", "responsibility_gauge"
        )):
            raise ValueError(f"legacy mechanism enabled: {(method, root)}")
        dimensions = cell.get("dimensions", {})
        if tuple(dimensions.get(name) for name in ("flat", "upper", "lower")) != (
            134, 134, 134
        ):
            raise ValueError(f"Stage-6 state shape changed: {(method, root)}")
        capacity = cell.get("capacity", {})
        signature = (
            int(capacity.get("reference_parameter_budget", -1)),
            int(capacity.get("actual_parameter_count", -1)),
            int(capacity.get("hidden_dim", -1)),
        )
        previous_capacity = capacity_contracts.setdefault(architecture, signature)
        if signature != previous_capacity or min(signature) < 1:
            raise ValueError(f"Stage-6 capacity mismatch: {(method, root)}")
        runtime = cell.get("runtime_versions")
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError(f"Stage-6 runtime missing: {(method, root)}")
        runtime_contracts.add(json.dumps(runtime, sort_keys=True))
        roles = tuple(
            tuple(map(int, cell.get(name, [])))
            for name in ("train_seeds", "selection_seeds", "eval_seeds")
        )
        if any(not values for values in roles):
            raise ValueError(f"Stage-6 seed role missing: {(method, root)}")
        previous_roles = role_contracts.setdefault(root, roles)
        if roles != previous_roles:
            raise ValueError(f"Stage-6 methods are not seed paired: root {root}")
    if len(runtime_contracts) != 1:
        raise ValueError("Stage-6 runtime versions differ")
    roots = tuple(sorted({root for _, root in identities}))
    expected = {
        (method, root)
        for root in roots
        for method in methods
    }
    if identities != expected:
        raise ValueError("Stage-6 method/root matrix is incomplete")
    return roots


def _index_rows(
    cells: Iterable[dict[str, Any]],
    *,
    field: str,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for cell in cells:
        method, root = _cell_identity(cell)
        rows = cell.get(field)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Stage-6 cell is missing {field}: {(method, root)}")
        expected_seeds = tuple(map(int, cell["eval_seeds"]))
        if tuple(int(row["seed"]) for row in rows) != expected_seeds:
            raise ValueError(f"Stage-6 evaluation seeds changed: {(method, root)}")
        for row in rows:
            key = (method, root, int(row["seed"]))
            if key in indexed:
                raise ValueError(f"duplicate Stage-6 row: {key}")
            if (
                int(row.get("training_replicate_seed", -1)) != root
                or row.get("method") != method
                or row.get("protocol_version")
                != POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION
                or row.get("algorithm_path")
                != POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH
                or float(row.get("protocol_valid", 0.0)) != 1.0
            ):
                raise ValueError(f"Stage-6 row identity invalid: {key}")
            if (
                not bool(row.get("external_stream_action_independent", False))
                or not bool(row.get("external_stream_visible_before_action", False))
                or not bool(row.get(
                    "current_physical_state_visible_to_both_levels", False
                ))
                or bool(row.get("external_future_visible_to_actor", True))
            ):
                raise ValueError(f"Stage-6 causal contract failed: {key}")
            if not all(np.isfinite(float(row[metric])) for metric in METRICS):
                raise ValueError(f"Stage-6 metric is not finite: {key}")
            indexed[key] = row
    return indexed


def _validate_row_pairing(
    indexed: dict[tuple[str, int, int], dict[str, Any]],
    *,
    methods: tuple[str, ...] = POINTMAZE_EXOGENOUS_ROUTING_METHODS,
) -> None:
    identities = {
        method: {
            (root, seed)
            for row_method, root, seed in indexed
            if row_method == method
        }
        for method in methods
    }
    if any(not values for values in identities.values()):
        raise ValueError("Stage-6 analysis requires every method")
    if len({frozenset(values) for values in identities.values()}) != 1:
        raise ValueError("Stage-6 rows are not method/root/seed paired")
    exogenous_keys = (
        "target_start_vertex",
        "target_initial_direction",
        "target_route",
        "target_round_trip_period_seconds",
        "force_x_period_seconds",
        "force_y_period_seconds",
        "force_x_rms",
        "force_y_rms",
    )
    reference_method = methods[0]
    for root, seed in identities[reference_method]:
        reference = {
            name: indexed[(reference_method, root, seed)].get(name)
            for name in exogenous_keys
        }
        for method in methods[1:]:
            candidate = {
                name: indexed[(method, root, seed)].get(name)
                for name in exogenous_keys
            }
            if candidate != reference:
                raise ValueError(
                    f"Stage-6 exogenous path is not paired: {(root, seed)}"
                )


def _root_means(
    indexed: dict[tuple[str, int, int], dict[str, Any]],
    *,
    method: str,
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for (row_method, root, _), row in indexed.items():
        if row_method == method:
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
        raise ValueError("Stage-6 contrast is not optimizer-root paired")
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


def _interaction_effect(
    root_means: dict[str, dict[str, dict[int, float]]],
    *,
    metric: str,
    higher_is_better: bool,
    confidence: float,
) -> dict[str, Any]:
    methods = (
        "hrl_exogenous_multiscale_all",
        "hrl_exogenous_history",
        "flat_exogenous_multiscale_all",
        "flat_exogenous_history",
    )
    roots = set(root_means[methods[0]][metric])
    if any(set(root_means[method][metric]) != roots for method in methods):
        raise ValueError("Stage-6 interaction is not optimizer-root paired")
    sign = 1.0 if higher_is_better else -1.0
    result = _interval(
        (
            sign * (
                root_means[methods[0]][metric][root]
                - root_means[methods[1]][metric][root]
                - root_means[methods[2]][metric][root]
                + root_means[methods[3]][metric][root]
            )
            for root in sorted(roots)
        ),
        confidence=confidence,
        classify=True,
    )
    result["mean_interaction_improvement"] = result.pop("mean")
    return result


def analyze_stage6(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    items = list(cells)
    roots = _validate_cells(items)
    final_rows = _index_rows(items, field="evaluation_rows")
    initial_rows = _index_rows(items, field="untrained_evaluation_rows")
    _validate_row_pairing(final_rows)
    _validate_row_pairing(initial_rows)
    if set(final_rows) != set(initial_rows):
        raise ValueError("Stage-6 trained and untrained rows are not paired")

    final_means = {
        method: {
            metric: _root_means(final_rows, method=method, metric=metric)
            for metric in METRICS
        }
        for method in POINTMAZE_EXOGENOUS_ROUTING_METHODS
    }
    initial_means = {
        method: {
            metric: _root_means(initial_rows, method=method, metric=metric)
            for metric in METRICS
        }
        for method in POINTMAZE_EXOGENOUS_ROUTING_METHODS
    }
    absolute = {
        method: {
            metric: _interval(
                values.values(), confidence=confidence, classify=False
            )
            for metric, values in final_means[method].items()
        }
        for method in POINTMAZE_EXOGENOUS_ROUTING_METHODS
    }
    learning_gain = {
        method: {
            metric: _paired_effect(
                final_means[method][metric],
                initial_means[method][metric],
                higher_is_better=higher_is_better,
                confidence=confidence,
            )
            for metric, higher_is_better in METRICS.items()
        }
        for method in POINTMAZE_EXOGENOUS_ROUTING_METHODS
    }
    contrasts = {
        name: {
            "candidate": candidate,
            "baseline": baseline,
            **{
                metric: _paired_effect(
                    final_means[candidate][metric],
                    final_means[baseline][metric],
                    higher_is_better=higher_is_better,
                    confidence=confidence,
                )
                for metric, higher_is_better in METRICS.items()
            },
        }
        for name, (candidate, baseline) in CONTRASTS.items()
    }
    interaction = {
        metric: _interaction_effect(
            final_means,
            metric=metric,
            higher_is_better=higher_is_better,
            confidence=confidence,
        )
        for metric, higher_is_better in METRICS.items()
    }
    routed = "hrl_exogenous_multiscale_routed"
    checks = {
        "routed_absolute_success_ci_lower": bool(
            absolute[routed]["tracking_success_rate"]["ci_lower"]
            >= MIN_ROUTED_TRACKING_SUCCESS
        ),
        "routed_final_vs_untrained_success": (
            learning_gain[routed]["tracking_success_rate"]["status"]
            == "supported"
        ),
        "routed_final_vs_untrained_return": (
            learning_gain[routed]["episode_return"]["status"] == "supported"
        ),
        "routed_vs_history_success": (
            contrasts["routed_vs_history"]["tracking_success_rate"]["status"]
            == "supported"
        ),
        "routed_vs_causal_filter_success": (
            contrasts["routed_vs_filtered"]["tracking_success_rate"]["status"]
            == "supported"
        ),
        "routed_vs_all_band_success": (
            contrasts["routed_vs_all_band"]["tracking_success_rate"]["status"]
            == "supported"
        ),
        "routed_vs_swapped_success": (
            contrasts["routed_vs_swapped"]["tracking_success_rate"]["status"]
            == "supported"
        ),
        "hierarchy_x_multiscale_success_interaction": (
            interaction["tracking_success_rate"]["status"] == "supported"
        ),
    }
    status = "supported" if all(checks.values()) else "not_supported"
    return {
        "analysis_version": "pointmaze_exogenous_routing_stage6_analysis_v1",
        "protocol_version": POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "evaluation_row_count": len(final_rows),
        "untrained_evaluation_row_count": len(initial_rows),
        "independent_training_replicate_count": len(roots),
        "statistical_unit": "optimizer_seed_root",
        "minimum_routed_tracking_success": MIN_ROUTED_TRACKING_SUCCESS,
        "absolute": absolute,
        "learning_gain_vs_untrained": learning_gain,
        "contrasts": contrasts,
        "hierarchy_x_multiscale_interaction": interaction,
        "claim_gate_checks": checks,
        "freq_hrl_routing_claim_status": status,
        "component_status": {
            "flat_multiscale_representation": contrasts[
                "flat_multiscale_vs_history"
            ]["tracking_success_rate"]["status"],
            "hrl_multiscale_representation": contrasts[
                "hrl_multiscale_vs_history"
            ]["tracking_success_rate"]["status"],
            "hierarchy_specific_multiscale_increment": interaction[
                "tracking_success_rate"
            ]["status"],
            "selective_routing_vs_all_band": contrasts[
                "routed_vs_all_band"
            ]["tracking_success_rate"]["status"],
            "correct_vs_swapped_routing": contrasts[
                "routed_vs_swapped"
            ]["tracking_success_rate"]["status"],
        },
        "root_means": {
            "final": {
                method: {
                    metric: {str(root): value for root, value in values.items()}
                    for metric, values in metrics.items()
                }
                for method, metrics in final_means.items()
            },
            "untrained": {
                method: {
                    metric: {str(root): value for root, value in values.items()}
                    for metric, values in metrics.items()
                }
                for method, metrics in initial_means.items()
            },
        },
    }


def load_cells(paths: Iterable[Path]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete Stage-6 result: {path}")
        protocol = payload.get("protocol", {})
        if (
            protocol.get("protocol_version")
            != POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION
            or protocol.get("algorithm_path")
            != POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH
        ):
            raise ValueError(f"wrong Stage-6 protocol: {path}")
        payload_cells = payload.get("cells", [])
        if len(payload_cells) != 1:
            raise ValueError(f"Stage-6 result must contain one cell: {path}")
        cells.extend(payload_cells)
    return cells


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# PointMaze Exogenous Frequency-Routing Stage-6 Analysis",
        "",
        f"Protocol: `{analysis['protocol_version']}`",
        (
            "Independent optimizer roots: "
            f"{analysis['independent_training_replicate_count']}"
        ),
        (
            "Strict Freq-HRL routing claim: "
            f"**{analysis['freq_hrl_routing_claim_status']}**"
        ),
        "",
        "| Method | Tracking success mean [95% CI] | Return | RMSE |",
        "|---|---:|---:|---:|",
    ]
    for method in POINTMAZE_EXOGENOUS_ROUTING_METHODS:
        row = analysis["absolute"][method]
        success = row["tracking_success_rate"]
        lines.append(
            f"| {method} | {success['mean']:.3f} "
            f"[{success['ci_lower']:.3f}, {success['ci_upper']:.3f}] "
            f"| {row['episode_return']['mean']:.3f} "
            f"| {row['tracking_rmse']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "| Root-paired contrast | Success improvement [95% CI] | Status |",
        "|---|---:|---|",
    ])
    for name, row in analysis["contrasts"].items():
        success = row["tracking_success_rate"]
        lines.append(
            f"| {name} | {success['mean_improvement']:.3f} "
            f"[{success['ci_lower']:.3f}, {success['ci_upper']:.3f}] "
            f"| {success['status']} |"
        )
    interaction = analysis["hierarchy_x_multiscale_interaction"][
        "tracking_success_rate"
    ]
    lines.extend([
        "",
        (
            "Hierarchy x multiscale success interaction: "
            f"{interaction['mean_interaction_improvement']:.3f} "
            f"[{interaction['ci_lower']:.3f}, {interaction['ci_upper']:.3f}], "
            f"**{interaction['status']}**."
        ),
        "",
        (
            "The optimizer root is the statistical unit; held-out episodes "
            "are averaged within root. Failed gate components remain failed."
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
    analysis = analyze_stage6(
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
