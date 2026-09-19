"""Root-paired analysis for the PointMaze stage-3 multiscale factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats

from .pointmaze_multiscale_validation import (
    POINTMAZE_MULTISCALE_ALGORITHM_PATH,
    POINTMAZE_MULTISCALE_CORE_METHODS,
    POINTMAZE_MULTISCALE_METHODS,
    POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
    POINTMAZE_MULTISCALE_SCENARIOS,
)


PRIMARY_STRESS_SCENARIOS = (
    "fast_observation_noise",
    "slow_drift_fast_action",
)
SECONDARY_STRESS_SCENARIO = "persistent_action_shift"
CLEAN_SUCCESS_NONINFERIORITY_MARGIN = 0.10
METRICS = {
    "success": True,
    "episode_return": True,
    "final_goal_distance": False,
}


def _interval(
    values: np.ndarray,
    *,
    confidence: float,
    classify_improvement: bool,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("stage-3 analysis values must be finite and non-empty")
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
    if classify_improvement:
        result["status"] = (
            "supported"
            if lower > 0.0
            else "contradicted" if upper < 0.0 else "inconclusive"
        )
    return result


def _index_rows(
    cells: Iterable[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, int, int], dict[str, Any]],
    dict[str, str],
]:
    indexed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    runtime_payloads: list[dict[str, str]] = []
    for cell in cells:
        method = str(cell.get("policy", ""))
        scenario = str(cell.get("scenario", ""))
        replicate = int(cell.get("optimizer_seed", -1))
        if method not in POINTMAZE_MULTISCALE_METHODS:
            raise ValueError(f"invalid stage-3 method: {method}")
        if scenario not in POINTMAZE_MULTISCALE_SCENARIOS or replicate < 0:
            raise ValueError("invalid stage-3 scenario or optimizer root")
        runtime = cell.get("runtime_versions")
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError("stage-3 cells must record runtime versions")
        runtime_payloads.append({str(key): str(value) for key, value in runtime.items()})
        for row in cell.get("evaluation_rows", []):
            row_replicate = int(row.get("training_replicate_seed", replicate))
            key = (scenario, method, row_replicate, int(row["seed"]))
            if key in indexed:
                raise ValueError(f"duplicate stage-3 evaluation row: {key}")
            if float(row.get("protocol_valid", 0.0)) != 1.0:
                raise ValueError(f"invalid stage-3 evaluation row: {key}")
            if str(row.get("algorithm_path")) != POINTMAZE_MULTISCALE_ALGORITHM_PATH:
                raise ValueError(f"wrong stage-3 algorithm path: {key}")
            if str(row.get("scenario")) != scenario or str(row.get("method")) != method:
                raise ValueError(f"stage-3 row identity mismatch: {key}")
            indexed[key] = row
    if not indexed:
        raise ValueError("stage-3 analysis requires evaluation rows")
    serialized = {json.dumps(runtime, sort_keys=True) for runtime in runtime_payloads}
    if len(serialized) != 1:
        raise ValueError("stage-3 cells use inconsistent runtime versions")
    return indexed, runtime_payloads[0]


def _grouped(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    scenario: str,
    method: str,
) -> dict[int, dict[int, dict[str, Any]]]:
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for (row_scenario, row_method, replicate, seed), row in indexed.items():
        if row_scenario == scenario and row_method == method:
            grouped.setdefault(replicate, {})[seed] = row
    if not grouped:
        raise ValueError(f"missing stage-3 rows for {scenario}/{method}")
    return grouped


def _validate_factorial_pairing(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
) -> set[int]:
    reference_roots: set[int] | None = None
    reference_seeds: dict[int, set[int]] | None = None
    for scenario in POINTMAZE_MULTISCALE_SCENARIOS:
        for method in POINTMAZE_MULTISCALE_METHODS:
            rows = _grouped(indexed, scenario=scenario, method=method)
            roots = set(rows)
            seed_sets = {root: set(items) for root, items in rows.items()}
            if reference_roots is None:
                reference_roots = roots
                reference_seeds = seed_sets
            elif roots != reference_roots or seed_sets != reference_seeds:
                raise ValueError(
                    "stage-3 methods and scenarios must share optimizer roots "
                    "and held-out seeds"
                )
    assert reference_roots is not None
    return reference_roots


def _root_means(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    scenario: str,
    method: str,
    metric: str,
) -> dict[int, float]:
    rows = _grouped(indexed, scenario=scenario, method=method)
    return {
        root: float(np.mean([float(row[metric]) for row in seeds.values()]))
        for root, seeds in rows.items()
    }


def _paired_values(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    scenario: str,
    candidate: str,
    baseline: str,
    metric: str,
    higher_is_better: bool,
) -> np.ndarray:
    candidate_means = _root_means(
        indexed, scenario=scenario, method=candidate, metric=metric
    )
    baseline_means = _root_means(
        indexed, scenario=scenario, method=baseline, metric=metric
    )
    if set(candidate_means) != set(baseline_means):
        raise ValueError("stage-3 comparison is not root paired")
    sign = 1.0 if higher_is_better else -1.0
    return np.asarray([
        sign * (candidate_means[root] - baseline_means[root])
        for root in sorted(candidate_means)
    ])


def _comparison(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    scenario: str,
    candidate: str,
    baseline: str,
    confidence: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate": candidate,
        "baseline": baseline,
    }
    for metric, higher_is_better in METRICS.items():
        interval = _interval(
            _paired_values(
                indexed,
                scenario=scenario,
                candidate=candidate,
                baseline=baseline,
                metric=metric,
                higher_is_better=higher_is_better,
            ),
            confidence=confidence,
            classify_improvement=True,
        )
        interval["mean_improvement"] = interval.pop("mean")
        result[metric] = interval
    statuses = {str(result[metric]["status"]) for metric in METRICS}
    result["joint_status"] = (
        "supported"
        if statuses == {"supported"}
        else "contradicted"
        if statuses == {"contradicted"}
        else "mixed"
        if "supported" in statuses or "contradicted" in statuses
        else "inconclusive"
    )
    return result


def _interaction_values(
    indexed: dict[tuple[str, str, int, int], dict[str, Any]],
    *,
    scenario: str,
    metric: str,
    higher_is_better: bool,
) -> np.ndarray:
    means = {
        method: _root_means(
            indexed,
            scenario=scenario,
            method=method,
            metric=metric,
        )
        for method in POINTMAZE_MULTISCALE_CORE_METHODS
    }
    roots = set(means[POINTMAZE_MULTISCALE_CORE_METHODS[0]])
    if any(set(values) != roots for values in means.values()):
        raise ValueError("stage-3 interaction is not root paired")
    sign = 1.0 if higher_is_better else -1.0
    return np.asarray([
        sign * (
            (means["hrl_multiscale"][root] - means["hrl_history"][root])
            - (means["flat_multiscale"][root] - means["flat_history"][root])
        )
        for root in sorted(roots)
    ])


def analyze_pointmaze_multiscale_cells(
    cells: Iterable[dict[str, Any]],
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    items = list(cells)
    protocols = {
        str(cell.get("protocol_version", "")) for cell in items
    }
    if protocols != {POINTMAZE_MULTISCALE_PROTOCOL_VERSION}:
        raise ValueError("stage-3 cells use a wrong or mixed protocol version")
    indexed, runtime_versions = _index_rows(items)
    roots = _validate_factorial_pairing(indexed)
    scenarios: dict[str, Any] = {}
    for scenario in POINTMAZE_MULTISCALE_SCENARIOS:
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
                    classify_improvement=False,
                )
                for metric in METRICS
            }
            for method in POINTMAZE_MULTISCALE_METHODS
        }
        interaction: dict[str, Any] = {}
        for metric, higher_is_better in METRICS.items():
            interval = _interval(
                _interaction_values(
                    indexed,
                    scenario=scenario,
                    metric=metric,
                    higher_is_better=higher_is_better,
                ),
                confidence=confidence,
                classify_improvement=True,
            )
            interval["mean_improvement"] = interval.pop("mean")
            interaction[metric] = interval
        scenarios[scenario] = {
            "absolute": absolute,
            "flat_representation": _comparison(
                indexed,
                scenario=scenario,
                candidate="flat_multiscale",
                baseline="flat_history",
                confidence=confidence,
            ),
            "hierarchy_on_history": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_history",
                baseline="flat_history",
                confidence=confidence,
            ),
            "freq_routing_increment": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale",
                baseline="hrl_history",
                confidence=confidence,
            ),
            "freq_hrl_vs_flat_multiscale": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale",
                baseline="flat_multiscale",
                confidence=confidence,
            ),
            "flat_multiscale_vs_causal_filter": _comparison(
                indexed,
                scenario=scenario,
                candidate="flat_multiscale",
                baseline="flat_causal_filter",
                confidence=confidence,
            ),
            "freq_hrl_vs_causal_filter": _comparison(
                indexed,
                scenario=scenario,
                candidate="hrl_multiscale",
                baseline="flat_causal_filter",
                confidence=confidence,
            ),
            "factorial_interaction": interaction,
        }
    clean_success = scenarios["clean"]["freq_routing_increment"]["success"]
    clean_noninferiority = {
        **clean_success,
        "margin": CLEAN_SUCCESS_NONINFERIORITY_MARGIN,
        "status": (
            "supported"
            if float(clean_success["ci_lower"])
            >= -CLEAN_SUCCESS_NONINFERIORITY_MARGIN
            else "not_supported"
        ),
    }
    stress_gates = {
        scenario: {
            "freq_routing_success": str(
                scenarios[scenario]["freq_routing_increment"]["success"]["status"]
            ),
            "factorial_interaction_success": str(
                scenarios[scenario]["factorial_interaction"]["success"]["status"]
            ),
        }
        for scenario in PRIMARY_STRESS_SCENARIOS
    }
    observation_filter_status = str(
        scenarios["fast_observation_noise"]["freq_hrl_vs_causal_filter"][
            "success"
        ]["status"]
    )
    mainline_status = (
        "supported"
        if all(
            gate["freq_routing_success"] == "supported"
            and gate["factorial_interaction_success"] == "supported"
            for gate in stress_gates.values()
        )
        and observation_filter_status == "supported"
        and clean_noninferiority["status"] == "supported"
        else "not_supported"
    )
    return {
        "analysis_version": "pointmaze_multiscale_goal_stage3_analysis_v2",
        "protocol_version": POINTMAZE_MULTISCALE_PROTOCOL_VERSION,
        "confidence": float(confidence),
        "cell_count": len(items),
        "evaluation_row_count": len(indexed),
        "independent_training_replicate_count": len(roots),
        "runtime_versions": runtime_versions,
        "primary_stress_scenarios": list(PRIMARY_STRESS_SCENARIOS),
        "secondary_stress_scenario": SECONDARY_STRESS_SCENARIO,
        "clean_success_noninferiority": clean_noninferiority,
        "scenarios": scenarios,
        "freq_hrl_mainline_status": mainline_status,
        "claim_gate": {
            "primary_stress": stress_gates,
            "observation_noise_vs_causal_filter_success": observation_filter_status,
            "clean_success_noninferiority": clean_noninferiority["status"],
        },
        "claim_boundary": (
            "Cross-stress Freq-HRL requires positive success increments and "
            "hierarchy-by-multiscale interactions in both registered primary "
            "stress families, superiority to the causal-filter control under "
            "observation noise, and clean success noninferiority."
        ),
    }


def load_pointmaze_multiscale_cells(
    paths: Iterable[Path],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = payload.get("protocol", {})
        if protocol.get("protocol_version") != POINTMAZE_MULTISCALE_PROTOCOL_VERSION:
            raise ValueError(f"wrong PointMaze stage-3 protocol in {path}")
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete PointMaze stage-3 result: {path}")
        cells.extend(payload.get("cells", []))
    return cells


def _format_interval(interval: dict[str, Any]) -> str:
    return (
        f"{float(interval['mean']):.3f} "
        f"[{float(interval['ci_lower']):.3f}, "
        f"{float(interval['ci_upper']):.3f}]"
    )


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# PointMaze Multiscale Stage-3 Analysis",
        "",
        f"Protocol: `{analysis['protocol_version']}`",
        f"Evaluation rows: {analysis['evaluation_row_count']}",
        f"Independent optimizer roots: {analysis['independent_training_replicate_count']}",
        f"Freq-HRL mainline gate: **{analysis['freq_hrl_mainline_status']}**",
        "",
    ]
    for scenario, result in analysis["scenarios"].items():
        lines.extend([
            f"## {scenario}",
            "",
            "| Method | Success [95% CI] | Return | Final distance |",
            "|---|---:|---:|---:|",
        ])
        for method in POINTMAZE_MULTISCALE_METHODS:
            absolute = result["absolute"][method]
            lines.append(
                f"| {method} | {_format_interval(absolute['success'])} "
                f"| {absolute['episode_return']['mean']:.3f} "
                f"| {absolute['final_goal_distance']['mean']:.3f} |"
            )
        lines.extend([
            "",
            "| Contrast | Success | Return | Final distance |",
            "|---|---|---|---|",
        ])
        for label, key in (
            ("Flat representation", "flat_representation"),
            ("Hierarchy on raw history", "hierarchy_on_history"),
            ("Freq routing increment", "freq_routing_increment"),
            ("Freq-HRL vs flat multiscale", "freq_hrl_vs_flat_multiscale"),
            ("Flat multiscale vs causal filter", "flat_multiscale_vs_causal_filter"),
            ("Freq-HRL vs causal filter", "freq_hrl_vs_causal_filter"),
        ):
            comparison = result[key]
            lines.append(
                f"| {label} | {comparison['success']['status']} "
                f"({comparison['success']['mean_improvement']:+.3f}) "
                f"| {comparison['episode_return']['status']} "
                f"({comparison['episode_return']['mean_improvement']:+.3f}) "
                f"| {comparison['final_goal_distance']['status']} "
                f"({comparison['final_goal_distance']['mean_improvement']:+.3f}) |"
            )
        interaction = result["factorial_interaction"]
        lines.append(
            "| Factorial interaction | "
            f"{interaction['success']['status']} "
            f"({interaction['success']['mean_improvement']:+.3f}) | "
            f"{interaction['episode_return']['status']} "
            f"({interaction['episode_return']['mean_improvement']:+.3f}) | "
            f"{interaction['final_goal_distance']['status']} "
            f"({interaction['final_goal_distance']['mean_improvement']:+.3f}) |"
        )
        lines.append("")
    gate = analysis["claim_gate"]
    lines.extend([
        "## Claim gate",
        "",
    ])
    for scenario, scenario_gate in gate["primary_stress"].items():
        lines.extend([
            f"- {scenario} Freq routing success: "
            f"**{scenario_gate['freq_routing_success']}**",
            f"- {scenario} factorial interaction: "
            f"**{scenario_gate['factorial_interaction_success']}**",
        ])
    lines.extend([
        "- Observation-noise Freq-HRL versus causal filter: "
        f"**{gate['observation_noise_vs_causal_filter_success']}**",
        f"- Clean success noninferiority: **{gate['clean_success_noninferiority']}**",
        "",
        analysis["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the PointMaze multiscale stage-3 factorial."
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cells = load_pointmaze_multiscale_cells(args.inputs)
    analysis = analyze_pointmaze_multiscale_cells(
        cells, confidence=args.confidence
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
