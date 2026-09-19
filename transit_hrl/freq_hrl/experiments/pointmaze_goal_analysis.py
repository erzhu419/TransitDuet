"""Paired analysis for the PointMaze ordinary-HRL stage-2 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from .pointmaze_goal_validation import (
    POINTMAZE_GOAL_PROTOCOL_VERSION,
    POINTMAZE_METHODS,
)


ORDINARY_HRL_MIN_SUCCESS_RATE = 0.50


def _interval(
    values: np.ndarray,
    *,
    confidence: float,
    improvement: bool,
) -> dict[str, float | int | str]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("PointMaze analysis values must be finite and non-empty")
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
    result: dict[str, float | int | str] = {
        "n": int(array.size),
        "mean": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": float(confidence),
    }
    if improvement:
        result["status"] = (
            "supported"
            if lower > 0.0
            else "contradicted" if upper < 0.0 else "inconclusive"
        )
    return result


def _index_rows(
    cells: Iterable[dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for cell in cells:
        method = str(cell["policy"])
        replicate = int(cell.get("optimizer_seed", -1))
        if method not in POINTMAZE_METHODS or replicate < 0:
            raise ValueError("PointMaze cell has an invalid identity")
        for row in cell.get("evaluation_rows", []):
            row_replicate = int(row.get("training_replicate_seed", replicate))
            key = (method, row_replicate, int(row["seed"]))
            if key in indexed:
                raise ValueError(f"duplicate PointMaze evaluation row: {key}")
            if float(row.get("protocol_valid", 0.0)) != 1.0:
                raise ValueError(f"invalid PointMaze evaluation row: {key}")
            if str(row.get("algorithm_path")) != "goal_conditioned_hrl_mainline":
                raise ValueError(f"wrong PointMaze algorithm path: {key}")
            indexed[key] = row
    if not indexed:
        raise ValueError("PointMaze analysis requires evaluation rows")
    return indexed


def _root_means(
    indexed: dict[tuple[str, int, int], dict[str, Any]],
    *,
    method: str,
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = {}
    for (row_method, replicate, _), row in indexed.items():
        if row_method == method:
            grouped.setdefault(replicate, []).append(float(row[metric]))
    if not grouped:
        raise ValueError(f"PointMaze method has no rows: {method}")
    return {
        replicate: float(np.mean(values))
        for replicate, values in grouped.items()
    }


def _paired_improvement(
    indexed: dict[tuple[str, int, int], dict[str, Any]],
    *,
    metric: str,
    higher_is_better: bool,
    confidence: float,
) -> dict[str, float | int | str]:
    candidate = _root_means(
        indexed, method="hrl_goal_ppo", metric=metric
    )
    baseline = _root_means(
        indexed, method="flat_goal_ppo", metric=metric
    )
    if set(candidate) != set(baseline):
        raise ValueError("PointMaze optimizer roots are not paired")
    sign = 1.0 if higher_is_better else -1.0
    differences = np.asarray([
        sign * (candidate[root] - baseline[root])
        for root in sorted(candidate)
    ])
    result = _interval(
        differences,
        confidence=confidence,
        improvement=True,
    )
    result["mean_improvement"] = result.pop("mean")
    return result


def analyze_pointmaze_cells(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    items = list(cells)
    runtime_payloads = [cell.get("runtime_versions") for cell in items]
    if any(not isinstance(payload, dict) or not payload for payload in runtime_payloads):
        raise ValueError("PointMaze cells must record runtime versions")
    serialized_runtimes = {
        json.dumps(payload, sort_keys=True)
        for payload in runtime_payloads
    }
    if len(serialized_runtimes) != 1:
        raise ValueError("PointMaze cells use inconsistent runtime versions")
    runtime_versions = dict(runtime_payloads[0])
    indexed = _index_rows(items)
    available = {method for method, _, _ in indexed}
    if available != set(POINTMAZE_METHODS):
        raise ValueError("PointMaze analysis requires both registered methods")
    replicate_sets = {
        method: {replicate for row_method, replicate, _ in indexed if row_method == method}
        for method in POINTMAZE_METHODS
    }
    if replicate_sets[POINTMAZE_METHODS[0]] != replicate_sets[POINTMAZE_METHODS[1]]:
        raise ValueError("PointMaze methods do not share optimizer roots")
    for replicate in replicate_sets[POINTMAZE_METHODS[0]]:
        seed_sets = {
            method: {
                seed
                for row_method, row_replicate, seed in indexed
                if row_method == method and row_replicate == replicate
            }
            for method in POINTMAZE_METHODS
        }
        if seed_sets[POINTMAZE_METHODS[0]] != seed_sets[POINTMAZE_METHODS[1]]:
            raise ValueError(
                "PointMaze methods do not share held-out seeds within root"
            )
    absolute: dict[str, Any] = {}
    for method in POINTMAZE_METHODS:
        absolute[method] = {
            metric: _interval(
                np.asarray(list(_root_means(
                    indexed, method=method, metric=metric
                ).values())),
                confidence=confidence,
                improvement=False,
            )
            for metric in (
                "success",
                "episode_return",
                "final_goal_distance",
            )
        }
    comparison = {
        "candidate": "hrl_goal_ppo",
        "baseline": "flat_goal_ppo",
        "success": _paired_improvement(
            indexed,
            metric="success",
            higher_is_better=True,
            confidence=confidence,
        ),
        "episode_return": _paired_improvement(
            indexed,
            metric="episode_return",
            higher_is_better=True,
            confidence=confidence,
        ),
        "final_goal_distance": _paired_improvement(
            indexed,
            metric="final_goal_distance",
            higher_is_better=False,
            confidence=confidence,
        ),
    }
    comparison_statuses = {
        str(comparison[metric]["status"])
        for metric in ("success", "episode_return", "final_goal_distance")
    }
    comparison["joint_status"] = (
        "supported"
        if comparison_statuses == {"supported"}
        else "contradicted"
        if comparison_statuses == {"contradicted"}
        else "mixed"
        if "supported" in comparison_statuses or "contradicted" in comparison_statuses
        else "inconclusive"
    )
    hrl_success = absolute["hrl_goal_ppo"]["success"]
    ordinary_hrl_status = (
        "supported"
        if float(hrl_success["ci_lower"]) >= ORDINARY_HRL_MIN_SUCCESS_RATE
        else "not_supported"
    )
    return {
        "analysis_version": "pointmaze_goal_control_stage2_analysis_v1",
        "protocol_version": POINTMAZE_GOAL_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "evaluation_row_count": len(indexed),
        "independent_training_replicate_count": len(
            replicate_sets[POINTMAZE_METHODS[0]]
        ),
        "ordinary_hrl_min_success_rate": ORDINARY_HRL_MIN_SUCCESS_RATE,
        "runtime_versions": runtime_versions,
        "absolute": absolute,
        "hrl_vs_flat": comparison,
        "ordinary_hrl_learning_status": ordinary_hrl_status,
        "multiscale_admission_status": (
            "admitted" if ordinary_hrl_status == "supported" else "blocked"
        ),
    }


def load_pointmaze_cells(paths: Iterable[Path]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        if protocol.get("protocol_version") != POINTMAZE_GOAL_PROTOCOL_VERSION:
            raise ValueError(f"wrong PointMaze protocol version in {path}")
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete PointMaze result: {path}")
        cells.extend(payload.get("cells", []))
    return cells


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# PointMaze Goal-Control Stage-2 Analysis",
        "",
        f"Protocol: `{analysis['protocol_version']}`",
        f"Evaluation rows: {analysis['evaluation_row_count']}",
        f"Independent training replicates: {analysis['independent_training_replicate_count']}",
        f"Ordinary HRL learning gate: **{analysis['ordinary_hrl_learning_status']}**",
        f"Multiscale admission: **{analysis['multiscale_admission_status']}**",
        "Runtime: " + ", ".join(
            f"{name}={value}"
            for name, value in analysis["runtime_versions"].items()
        ),
        "",
        "| Method | Success mean [95% CI] | Return mean | Final distance mean |",
        "|---|---:|---:|---:|",
    ]
    for method in POINTMAZE_METHODS:
        row = analysis["absolute"][method]
        success = row["success"]
        lines.append(
            f"| {method} | {success['mean']:.3f} "
            f"[{success['ci_lower']:.3f}, {success['ci_upper']:.3f}] "
            f"| {row['episode_return']['mean']:.3f} "
            f"| {row['final_goal_distance']['mean']:.3f} |"
        )
    comparison = analysis["hrl_vs_flat"]
    lines.extend([
        "",
        f"HRL versus flat paired status: **{comparison['joint_status']}**.",
        "The statistical unit is the independent optimizer root; evaluation episodes are averaged within root.",
        "Multiscale mechanisms remain blocked unless the ordinary HRL success-rate CI clears the frozen gate.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PointMaze stage-2 results.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = load_pointmaze_cells(args.inputs)
    analysis = analyze_pointmaze_cells(cells, confidence=args.confidence)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(analysis),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
