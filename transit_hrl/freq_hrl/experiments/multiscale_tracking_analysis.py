"""Paired analysis for the multiscale goal-control stage-1 factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from .multiscale_tracking_validation import (
    MULTISCALE_GOAL_PROTOCOL_VERSION,
    STAGE1_CORE_METHODS,
)


PRIMARY_SCENARIOS = (
    "clean",
    "slow_target_fast_force",
    "slow_signal_fast_observation_noise",
)
BOUNDARY_SCENARIO = "band_swap"


def _paired_interval(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("paired differences must be finite and non-empty")
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
    if lower > 0.0:
        status = "supported"
    elif upper < 0.0:
        status = "contradicted"
    else:
        status = "inconclusive"
    return {
        "n": int(array.size),
        "mean_improvement": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": float(confidence),
        "status": status,
    }


def _index_rows(cells: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for cell in cells:
        method = str(cell["policy"])
        scenario = str(cell["scenario"])
        for row in cell.get("evaluation_rows", []):
            key = (scenario, method, int(row["seed"]))
            if key in indexed:
                raise ValueError(f"duplicate evaluation row: {key}")
            if float(row.get("protocol_valid", 0.0)) != 1.0:
                raise ValueError(f"invalid evaluation row: {key}")
            if str(row.get("algorithm_path")) != "multiscale_goal_hrl_mainline":
                raise ValueError(f"wrong algorithm path in row: {key}")
            indexed[key] = row
    if not indexed:
        raise ValueError("analysis requires evaluation rows")
    return indexed


def _paired_values(
    indexed: dict[tuple[str, str, int], dict[str, Any]],
    *,
    scenario: str,
    candidate: str,
    baseline: str,
    metric: str,
    higher_is_better: bool,
) -> np.ndarray:
    candidate_rows = {
        seed: row
        for (row_scenario, method, seed), row in indexed.items()
        if row_scenario == scenario and method == candidate
    }
    baseline_rows = {
        seed: row
        for (row_scenario, method, seed), row in indexed.items()
        if row_scenario == scenario and method == baseline
    }
    if not candidate_rows or set(candidate_rows) != set(baseline_rows):
        raise ValueError(
            f"unpaired rows for {scenario}: {candidate} versus {baseline}"
        )
    sign = 1.0 if higher_is_better else -1.0
    return np.asarray([
        sign * (
            float(candidate_rows[seed][metric])
            - float(baseline_rows[seed][metric])
        )
        for seed in sorted(candidate_rows)
    ], dtype=np.float64)


def _comparison(
    indexed: dict[tuple[str, str, int], dict[str, Any]],
    *,
    scenario: str,
    candidate: str,
    baseline: str,
    confidence: float,
) -> dict[str, Any]:
    reward = _paired_interval(_paired_values(
        indexed,
        scenario=scenario,
        candidate=candidate,
        baseline=baseline,
        metric="episode_return",
        higher_is_better=True,
    ), confidence=confidence)
    tracking = _paired_interval(_paired_values(
        indexed,
        scenario=scenario,
        candidate=candidate,
        baseline=baseline,
        metric="tracking_rmse",
        higher_is_better=False,
    ), confidence=confidence)
    statuses = {str(reward["status"]), str(tracking["status"])}
    if statuses == {"supported"}:
        joint = "supported"
    elif statuses == {"contradicted"}:
        joint = "contradicted"
    elif "supported" in statuses or "contradicted" in statuses:
        joint = "mixed"
    else:
        joint = "inconclusive"
    return {
        "candidate": candidate,
        "baseline": baseline,
        "episode_return": reward,
        "tracking_rmse": tracking,
        "joint_status": joint,
    }


def _interaction(
    indexed: dict[tuple[str, str, int], dict[str, Any]],
    *,
    scenario: str,
    metric: str,
    higher_is_better: bool,
    confidence: float,
) -> dict[str, Any]:
    methods = tuple(STAGE1_CORE_METHODS)
    rows_by_method: dict[str, dict[int, dict[str, Any]]] = {
        method: {
            seed: row
            for (row_scenario, row_method, seed), row in indexed.items()
            if row_scenario == scenario and row_method == method
        }
        for method in methods
    }
    seed_sets = [set(rows) for rows in rows_by_method.values()]
    if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError(f"factorial rows are not paired for {scenario}")
    sign = 1.0 if higher_is_better else -1.0
    values = []
    for seed in sorted(seed_sets[0]):
        flat_gain = (
            float(rows_by_method["flat_multiscale"][seed][metric])
            - float(rows_by_method["flat_history"][seed][metric])
        )
        hrl_gain = (
            float(rows_by_method["hrl_multiscale"][seed][metric])
            - float(rows_by_method["hrl_history"][seed][metric])
        )
        values.append(sign * (hrl_gain - flat_gain))
    return _paired_interval(np.asarray(values), confidence=confidence)


def load_stage1_cells(paths: Iterable[Path]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        if protocol.get("protocol_version") != MULTISCALE_GOAL_PROTOCOL_VERSION:
            raise ValueError(f"wrong protocol version in {path}")
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete stage-1 result: {path}")
        cells.extend(payload.get("cells", []))
    return cells


def analyze_stage1_cells(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    items = list(cells)
    indexed = _index_rows(items)
    available = {
        (scenario, method)
        for scenario, method, _ in indexed
    }
    required = {
        (scenario, method)
        for scenario in (*PRIMARY_SCENARIOS, BOUNDARY_SCENARIO)
        for method in STAGE1_CORE_METHODS
    }
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError(f"stage-1 factorial is incomplete: {missing}")
    scenario_results: dict[str, Any] = {}
    for scenario in (*PRIMARY_SCENARIOS, BOUNDARY_SCENARIO):
        scenario_results[scenario] = {
            "representation_flat": _comparison(
                indexed,
                scenario=scenario,
                candidate="flat_multiscale",
                baseline="flat_history",
                confidence=confidence,
            ),
            "hierarchy_history": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_history",
                baseline="flat_history",
                confidence=confidence,
            ),
            "multiscale_hrl_vs_history_hrl": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale",
                baseline="hrl_history",
                confidence=confidence,
            ),
            "multiscale_hrl_vs_flat_multiscale": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale",
                baseline="flat_multiscale",
                confidence=confidence,
            ),
            "factorial_interaction": {
                "episode_return": _interaction(
                    indexed,
                    scenario=scenario,
                    metric="episode_return",
                    higher_is_better=True,
                    confidence=confidence,
                ),
                "tracking_rmse": _interaction(
                    indexed,
                    scenario=scenario,
                    metric="tracking_rmse",
                    higher_is_better=False,
                    confidence=confidence,
                ),
            },
        }
        if (scenario, "flat_causal_filter") in available:
            scenario_results[scenario]["multiscale_vs_causal_filter"] = _comparison(
                indexed,
                scenario=scenario,
                candidate="flat_multiscale",
                baseline="flat_causal_filter",
                confidence=confidence,
            )
    primary_joint = [
        scenario_results[scenario]["multiscale_hrl_vs_flat_multiscale"][
            "joint_status"
        ]
        for scenario in PRIMARY_SCENARIOS
    ]
    return {
        "analysis_version": "multiscale_goal_control_stage1_analysis_v1",
        "protocol_version": MULTISCALE_GOAL_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "evaluation_row_count": len(indexed),
        "scenarios": scenario_results,
        "mainline_hrl_increment_status": (
            "supported"
            if all(status == "supported" for status in primary_joint)
            else "not_supported"
        ),
        "band_swap_is_boundary_not_success_gate": True,
    }


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# Multiscale Goal-Control Stage-1 Analysis",
        "",
        f"Protocol: `{analysis['protocol_version']}`",
        f"Evaluation rows: {analysis['evaluation_row_count']}",
        f"Mainline HRL increment: **{analysis['mainline_hrl_increment_status']}**",
        "",
        "| Scenario | Flat representation | HRL on history | Multiscale HRL vs HRL | Multiscale HRL vs flat multiscale |",
        "|---|---|---|---|---|",
    ]
    for scenario, result in analysis["scenarios"].items():
        lines.append(
            f"| {scenario} | {result['representation_flat']['joint_status']} "
            f"| {result['hierarchy_history']['joint_status']} "
            f"| {result['multiscale_hrl_vs_history_hrl']['joint_status']} "
            f"| {result['multiscale_hrl_vs_flat_multiscale']['joint_status']} |"
        )
    lines.extend([
        "",
        "Positive differences mean improvement: higher episode return or lower tracking RMSE.",
        "The band-swap row defines the assumption boundary and is not included in the success gate.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze paired stage-1 results.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = load_stage1_cells(args.inputs)
    analysis = analyze_stage1_cells(cells, confidence=args.confidence)
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
