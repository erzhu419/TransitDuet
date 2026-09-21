"""Root-paired analysis for the separate-exogenous PointMaze substrate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from .pointmaze_exogenous_validation import (
    POINTMAZE_EXOGENOUS_ALGORITHM_PATH,
    POINTMAZE_EXOGENOUS_METHODS,
    POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
)


EXOGENOUS_HRL_MIN_TRACKING_SUCCESS_RATE = 0.50
METRICS = {
    "tracking_success_rate": True,
    "episode_return": True,
    "tracking_rmse": False,
    "final_tracking_distance": False,
}


def _interval(
    values: np.ndarray,
    *,
    confidence: float,
    classify: bool,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("external PointMaze analysis requires finite values")
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


def _index_rows(
    cells: Iterable[dict[str, Any]],
    *,
    field: str,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for cell in cells:
        method = str(cell.get("policy", ""))
        root = int(cell.get("optimizer_seed", -1))
        if method not in POINTMAZE_EXOGENOUS_METHODS or root < 0:
            raise ValueError("external PointMaze cell has an invalid identity")
        rows = cell.get(field)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"external PointMaze cell is missing {field}")
        for row in rows:
            row_root = int(row.get("training_replicate_seed", root))
            key = (method, row_root, int(row["seed"]))
            if key in indexed:
                raise ValueError(f"duplicate external PointMaze row: {key}")
            if float(row.get("protocol_valid", 0.0)) != 1.0:
                raise ValueError(f"invalid external PointMaze row: {key}")
            if str(row.get("algorithm_path")) != POINTMAZE_EXOGENOUS_ALGORITHM_PATH:
                raise ValueError(f"wrong external PointMaze algorithm path: {key}")
            if str(row.get("method")) != method:
                raise ValueError(f"external PointMaze row identity mismatch: {key}")
            if (
                not bool(row.get("external_stream_action_independent", False))
                or not bool(row.get("external_stream_visible_before_action", False))
                or not bool(row.get(
                    "current_physical_state_visible_to_both_levels", False
                ))
                or bool(row.get("external_future_visible_to_actor", True))
            ):
                raise ValueError(f"external PointMaze causal contract failed: {key}")
            indexed[key] = row
    if not indexed:
        raise ValueError("external PointMaze analysis requires rows")
    return indexed


def _validate_pairing(
    indexed: dict[tuple[str, int, int], dict[str, Any]],
) -> set[int]:
    identities = {
        method: {
            (root, seed)
            for row_method, root, seed in indexed
            if row_method == method
        }
        for method in POINTMAZE_EXOGENOUS_METHODS
    }
    if any(not values for values in identities.values()):
        raise ValueError("external PointMaze analysis requires both methods")
    if len({frozenset(values) for values in identities.values()}) != 1:
        raise ValueError("external PointMaze methods must be root/seed paired")
    return {root for root, _ in next(iter(identities.values()))}


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
    if not grouped:
        raise ValueError(f"missing external PointMaze rows for {method}")
    return {
        root: float(np.mean(values)) for root, values in grouped.items()
    }


def _paired_comparison(
    candidate: dict[int, float],
    baseline: dict[int, float],
    *,
    higher_is_better: bool,
    confidence: float,
) -> dict[str, Any]:
    if set(candidate) != set(baseline):
        raise ValueError("external PointMaze comparison is not root paired")
    sign = 1.0 if higher_is_better else -1.0
    differences = np.asarray([
        sign * (candidate[root] - baseline[root])
        for root in sorted(candidate)
    ])
    result = _interval(differences, confidence=confidence, classify=True)
    result["mean_improvement"] = result.pop("mean")
    return result


def analyze_pointmaze_exogenous_cells(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    items = list(cells)
    protocols = {str(cell.get("protocol_version", "")) for cell in items}
    if protocols != {POINTMAZE_EXOGENOUS_PROTOCOL_VERSION}:
        raise ValueError("external PointMaze cells use a wrong protocol")
    runtimes = [cell.get("runtime_versions") for cell in items]
    if any(not isinstance(runtime, dict) or not runtime for runtime in runtimes):
        raise ValueError("external PointMaze cells must record runtime versions")
    serialized = {json.dumps(runtime, sort_keys=True) for runtime in runtimes}
    if len(serialized) != 1:
        raise ValueError("external PointMaze runtime versions differ")

    final_rows = _index_rows(items, field="evaluation_rows")
    initial_rows = _index_rows(items, field="untrained_evaluation_rows")
    roots = _validate_pairing(final_rows)
    _validate_pairing(initial_rows)
    if set(final_rows) != set(initial_rows):
        raise ValueError("trained and untrained external rows are not paired")

    absolute: dict[str, Any] = {}
    learning_gain: dict[str, Any] = {}
    for method in POINTMAZE_EXOGENOUS_METHODS:
        absolute[method] = {
            metric: _interval(
                np.asarray(list(_root_means(
                    final_rows,
                    method=method,
                    metric=metric,
                ).values())),
                confidence=confidence,
                classify=False,
            )
            for metric in METRICS
        }
        learning_gain[method] = {
            metric: _paired_comparison(
                _root_means(final_rows, method=method, metric=metric),
                _root_means(initial_rows, method=method, metric=metric),
                higher_is_better=higher_is_better,
                confidence=confidence,
            )
            for metric, higher_is_better in METRICS.items()
        }

    hrl_vs_flat = {
        "candidate": "hrl_exogenous_history",
        "baseline": "flat_exogenous_history",
        **{
            metric: _paired_comparison(
                _root_means(
                    final_rows,
                    method="hrl_exogenous_history",
                    metric=metric,
                ),
                _root_means(
                    final_rows,
                    method="flat_exogenous_history",
                    metric=metric,
                ),
                higher_is_better=higher_is_better,
                confidence=confidence,
            )
            for metric, higher_is_better in METRICS.items()
        },
    }
    hrl_absolute = absolute["hrl_exogenous_history"]["tracking_success_rate"]
    hrl_gain = learning_gain["hrl_exogenous_history"]
    absolute_gate = bool(
        float(hrl_absolute["ci_lower"])
        >= EXOGENOUS_HRL_MIN_TRACKING_SUCCESS_RATE
    )
    learning_gate = bool(
        hrl_gain["tracking_success_rate"]["status"] == "supported"
        and hrl_gain["episode_return"]["status"] == "supported"
    )
    gate = "supported" if absolute_gate and learning_gate else "not_supported"
    return {
        "analysis_version": "pointmaze_exogenous_control_stage5_analysis_v1",
        "protocol_version": POINTMAZE_EXOGENOUS_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "evaluation_row_count": len(final_rows),
        "untrained_evaluation_row_count": len(initial_rows),
        "independent_training_replicate_count": len(roots),
        "runtime_versions": dict(runtimes[0]),
        "ordinary_hrl_min_tracking_success_rate": (
            EXOGENOUS_HRL_MIN_TRACKING_SUCCESS_RATE
        ),
        "absolute": absolute,
        "learning_gain_vs_untrained": learning_gain,
        "hrl_vs_flat": hrl_vs_flat,
        "absolute_learning_gate_passed": absolute_gate,
        "paired_learning_gain_gate_passed": learning_gate,
        "exogenous_hrl_learning_status": gate,
        "frequency_routing_admission_status": (
            "admitted" if gate == "supported" else "blocked"
        ),
    }


def load_pointmaze_exogenous_cells(paths: Iterable[Path]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete external PointMaze result: {path}")
        protocol = payload.get("protocol", {})
        if protocol.get("protocol_version") != POINTMAZE_EXOGENOUS_PROTOCOL_VERSION:
            raise ValueError(f"wrong external PointMaze protocol in {path}")
        cells.extend(payload.get("cells", []))
    return cells


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# PointMaze Exogenous-Control Stage-5 Analysis",
        "",
        f"Protocol: `{analysis['protocol_version']}`",
        f"Evaluation rows: {analysis['evaluation_row_count']}",
        (
            "Independent training replicates: "
            f"{analysis['independent_training_replicate_count']}"
        ),
        (
            "Exogenous ordinary-HRL gate: "
            f"**{analysis['exogenous_hrl_learning_status']}**"
        ),
        (
            "Frequency-routing admission: "
            f"**{analysis['frequency_routing_admission_status']}**"
        ),
        "",
        "| Method | Tracking success mean [95% CI] | Return | Tracking RMSE |",
        "|---|---:|---:|---:|",
    ]
    for method in POINTMAZE_EXOGENOUS_METHODS:
        row = analysis["absolute"][method]
        success = row["tracking_success_rate"]
        lines.append(
            f"| {method} | {success['mean']:.3f} "
            f"[{success['ci_lower']:.3f}, {success['ci_upper']:.3f}] "
            f"| {row['episode_return']['mean']:.3f} "
            f"| {row['tracking_rmse']['mean']:.3f} |"
        )
    hrl_gain = analysis["learning_gain_vs_untrained"]["hrl_exogenous_history"]
    lines.extend([
        "",
        (
            "HRL final-minus-untrained tracking-success status: "
            f"**{hrl_gain['tracking_success_rate']['status']}**."
        ),
        (
            "HRL final-minus-untrained return status: "
            f"**{hrl_gain['episode_return']['status']}**."
        ),
        (
            "The optimizer root is the statistical unit; held-out episodes "
            "are averaged within root."
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
    analysis = analyze_pointmaze_exogenous_cells(
        load_pointmaze_exogenous_cells(args.inputs),
        confidence=args.confidence,
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
