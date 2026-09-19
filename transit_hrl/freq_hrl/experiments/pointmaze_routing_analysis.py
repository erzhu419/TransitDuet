"""Root-paired analysis for PointMaze Stage-4 routing attribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from .pointmaze_routing_attribution import (
    POINTMAZE_ROUTING_ALGORITHM_PATH,
    POINTMAZE_ROUTING_METHODS,
    POINTMAZE_ROUTING_PROTOCOL_VERSION,
    POINTMAZE_ROUTING_SCENARIOS,
)


PRIMARY_STRESS_SCENARIOS = (
    "fast_observation_noise",
    "slow_drift_fast_action",
)
CLEAN_NONINFERIORITY_MARGIN = 0.10
METRICS = {
    "success": True,
    "episode_return": True,
    "final_goal_distance": False,
}


def _interval(
    values: np.ndarray,
    *,
    confidence: float,
    classify: bool,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("routing analysis values must be finite and non-empty")
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
    result: dict[str, Any] = {
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
) -> tuple[dict[tuple[str, str, int, int], dict[str, Any]], dict[str, str]]:
    indexed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    runtimes: list[dict[str, str]] = []
    for cell in cells:
        method = str(cell.get("policy", ""))
        scenario = str(cell.get("scenario", ""))
        root = int(cell.get("optimizer_seed", -1))
        if method not in POINTMAZE_ROUTING_METHODS:
            raise ValueError(f"invalid routing method: {method}")
        if scenario not in POINTMAZE_ROUTING_SCENARIOS or root < 0:
            raise ValueError("invalid routing scenario or optimizer root")
        runtime = cell.get("runtime_versions")
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError("routing cells must record runtime versions")
        runtimes.append({str(key): str(value) for key, value in runtime.items()})
        for row in cell.get("evaluation_rows", []):
            row_root = int(row.get("training_replicate_seed", root))
            key = (scenario, method, row_root, int(row["seed"]))
            if key in indexed:
                raise ValueError(f"duplicate routing evaluation row: {key}")
            if float(row.get("protocol_valid", 0.0)) != 1.0:
                raise ValueError(f"invalid routing evaluation row: {key}")
            if str(row.get("algorithm_path")) != POINTMAZE_ROUTING_ALGORITHM_PATH:
                raise ValueError(f"wrong routing algorithm path: {key}")
            if str(row.get("scenario")) != scenario or str(row.get("method")) != method:
                raise ValueError(f"routing row identity mismatch: {key}")
            indexed[key] = row
    if not indexed:
        raise ValueError("routing analysis requires evaluation rows")
    serialized = {json.dumps(runtime, sort_keys=True) for runtime in runtimes}
    if len(serialized) != 1:
        raise ValueError("routing cells use inconsistent runtime versions")
    return indexed, runtimes[0]


def _grouped(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    scenario: str,
    method: str,
) -> dict[int, dict[int, dict[str, Any]]]:
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for (row_scenario, row_method, root, seed), row in indexed.items():
        if row_scenario == scenario and row_method == method:
            grouped.setdefault(root, {})[seed] = row
    if not grouped:
        raise ValueError(f"missing routing rows for {scenario}/{method}")
    return grouped


def _validate_pairing(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
) -> set[int]:
    reference_roots: set[int] | None = None
    reference_seeds: dict[int, set[int]] | None = None
    for scenario in POINTMAZE_ROUTING_SCENARIOS:
        for method in POINTMAZE_ROUTING_METHODS:
            grouped = _grouped(indexed, scenario=scenario, method=method)
            roots = set(grouped)
            seeds = {root: set(rows) for root, rows in grouped.items()}
            if reference_roots is None:
                reference_roots, reference_seeds = roots, seeds
            elif roots != reference_roots or seeds != reference_seeds:
                raise ValueError("routing methods and scenarios must be root paired")
    assert reference_roots is not None
    return reference_roots


def _root_means(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    scenario: str,
    method: str,
    metric: str,
) -> dict[int, float]:
    grouped = _grouped(indexed, scenario=scenario, method=method)
    return {
        root: float(np.mean([float(row[metric]) for row in rows.values()]))
        for root, rows in grouped.items()
    }


def _comparison(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    scenario: str,
    candidate: str,
    baseline: str,
    confidence: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {"candidate": candidate, "baseline": baseline}
    for metric, higher_is_better in METRICS.items():
        candidate_means = _root_means(
            indexed, scenario=scenario, method=candidate, metric=metric
        )
        baseline_means = _root_means(
            indexed, scenario=scenario, method=baseline, metric=metric
        )
        if set(candidate_means) != set(baseline_means):
            raise ValueError("routing comparison is not root paired")
        sign = 1.0 if higher_is_better else -1.0
        values = np.asarray([
            sign * (candidate_means[root] - baseline_means[root])
            for root in sorted(candidate_means)
        ])
        interval = _interval(values, confidence=confidence, classify=True)
        interval["mean_improvement"] = interval.pop("mean")
        result[metric] = interval
    return result


def analyze_pointmaze_routing_cells(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    items = list(cells)
    protocols = {str(cell.get("protocol_version", "")) for cell in items}
    if protocols != {POINTMAZE_ROUTING_PROTOCOL_VERSION}:
        raise ValueError("routing cells use a wrong or mixed protocol version")
    indexed, runtime_versions = _index_rows(items)
    roots = _validate_pairing(indexed)
    scenarios: dict[str, Any] = {}
    for scenario in POINTMAZE_ROUTING_SCENARIOS:
        absolute = {
            method: {
                metric: _interval(
                    np.asarray(list(_root_means(
                        indexed,
                        scenario=scenario,
                        method=method,
                        metric=metric,
                    ).values())),
                    confidence=confidence,
                    classify=False,
                )
                for metric in METRICS
            }
            for method in POINTMAZE_ROUTING_METHODS
        }
        scenarios[scenario] = {
            "absolute": absolute,
            "routed_vs_history": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale_routed",
                baseline="hrl_history",
                confidence=confidence,
            ),
            "routed_vs_filter": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale_routed",
                baseline="hrl_causal_filter",
                confidence=confidence,
            ),
            "routed_vs_all": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale_routed",
                baseline="hrl_multiscale_all",
                confidence=confidence,
            ),
            "routed_vs_swapped": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale_routed",
                baseline="hrl_multiscale_swapped",
                confidence=confidence,
            ),
            "all_vs_history": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale_all",
                baseline="hrl_history",
                confidence=confidence,
            ),
        }
    clean = scenarios["clean"]["routed_vs_all"]["success"]
    clean_noninferiority = {
        **clean,
        "margin": CLEAN_NONINFERIORITY_MARGIN,
        "status": (
            "supported"
            if float(clean["ci_lower"]) >= -CLEAN_NONINFERIORITY_MARGIN
            else "not_supported"
        ),
    }
    stress_gate = {
        scenario: {
            "routed_vs_all": scenarios[scenario]["routed_vs_all"]["success"]["status"],
            "routed_vs_swapped": scenarios[scenario]["routed_vs_swapped"]["success"]["status"],
        }
        for scenario in PRIMARY_STRESS_SCENARIOS
    }
    observation_filter = scenarios["fast_observation_noise"][
        "routed_vs_filter"
    ]["success"]["status"]
    mainline_status = (
        "supported"
        if all(
            values["routed_vs_all"] == "supported"
            and values["routed_vs_swapped"] == "supported"
            for values in stress_gate.values()
        )
        and observation_filter == "supported"
        and clean_noninferiority["status"] == "supported"
        else "not_supported"
    )
    return {
        "analysis_version": "pointmaze_frequency_routing_stage4_analysis_v1",
        "protocol_version": POINTMAZE_ROUTING_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "evaluation_row_count": len(indexed),
        "independent_training_replicate_count": len(roots),
        "runtime_versions": runtime_versions,
        "primary_stress_scenarios": list(PRIMARY_STRESS_SCENARIOS),
        "scenarios": scenarios,
        "clean_success_noninferiority": clean_noninferiority,
        "routing_attribution_status": mainline_status,
        "claim_gate": {
            "primary_stress": stress_gate,
            "observation_routed_vs_filter": observation_filter,
            "clean_routed_vs_all_noninferiority": clean_noninferiority["status"],
        },
        "claim_boundary": (
            "Selective routing is supported only if it beats all-band and "
            "swapped routing in both primary stresses, beats causal filtering "
            "under observation noise, and is clean-noninferior to all-band HRL."
        ),
    }


def load_pointmaze_routing_cells(paths: Iterable[Path]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"routing result is not complete: {path}")
        cells.extend(payload.get("cells", []))
    return cells


def _format_interval(interval: dict[str, Any]) -> str:
    return (
        f"{float(interval['mean']):.3f} "
        f"[{float(interval['ci_lower']):.3f}, {float(interval['ci_upper']):.3f}]"
    )


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# PointMaze Stage-4 Frequency-Routing Attribution",
        "",
        f"Protocol: `{analysis['protocol_version']}`",
        f"Evaluation rows: {analysis['evaluation_row_count']}",
        f"Independent optimizer roots: {analysis['independent_training_replicate_count']}",
        f"Routing attribution gate: **{analysis['routing_attribution_status']}**",
        "",
    ]
    comparisons = (
        ("Routed vs history", "routed_vs_history"),
        ("Routed vs causal filter", "routed_vs_filter"),
        ("Routed vs all bands", "routed_vs_all"),
        ("Routed vs swapped", "routed_vs_swapped"),
        ("All bands vs history", "all_vs_history"),
    )
    for scenario, result in analysis["scenarios"].items():
        lines.extend([
            f"## {scenario}",
            "",
            "| Method | Success [95% CI] | Return | Final distance |",
            "|---|---:|---:|---:|",
        ])
        for method in POINTMAZE_ROUTING_METHODS:
            absolute = result["absolute"][method]
            lines.append(
                f"| {method} | {_format_interval(absolute['success'])} | "
                f"{absolute['episode_return']['mean']:.3f} | "
                f"{absolute['final_goal_distance']['mean']:.3f} |"
            )
        lines.extend(["", "| Contrast | Success | Return | Final distance |", "|---|---|---|---|"])
        for label, key in comparisons:
            comparison = result[key]
            lines.append(
                f"| {label} | {comparison['success']['status']} "
                f"({comparison['success']['mean_improvement']:+.3f}) | "
                f"{comparison['episode_return']['status']} "
                f"({comparison['episode_return']['mean_improvement']:+.3f}) | "
                f"{comparison['final_goal_distance']['status']} "
                f"({comparison['final_goal_distance']['mean_improvement']:+.3f}) |"
            )
        lines.append("")
    gate = analysis["claim_gate"]
    lines.extend(["## Claim gate", ""])
    for scenario, values in gate["primary_stress"].items():
        lines.append(f"- {scenario} routed vs all: **{values['routed_vs_all']}**")
        lines.append(f"- {scenario} routed vs swapped: **{values['routed_vs_swapped']}**")
    lines.extend([
        f"- Observation routed vs causal filter: **{gate['observation_routed_vs_filter']}**",
        f"- Clean routed vs all noninferiority: **{gate['clean_routed_vs_all_noninferiority']}**",
        "",
        analysis["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze PointMaze Stage-4 routing attribution."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = analyze_pointmaze_routing_cells(
        load_pointmaze_routing_cells(args.inputs),
        confidence=args.confidence,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_report(analysis), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
