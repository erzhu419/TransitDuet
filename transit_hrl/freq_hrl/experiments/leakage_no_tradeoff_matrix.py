"""Leakage/no-tradeoff evidence matrix across Freq-HRL domains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.statistics import claim_status, noninferiority_status, paired_delta_stats


DEFAULT_RESULT_PATHS = {
    "trading_constraint": Path("transit_hrl/results/trading_lower_lf_constraint_validation/summary.json"),
    "trading_ppo_primal_dual": Path("transit_hrl/results/trading_ppo_primal_dual_leakage/summary.json"),
    "transit_real_surrogate": Path("transit_hrl/results/transit_real_demand_control/summary.json"),
    "transit_ppo_primal_dual": Path("transit_hrl/results/transit_ppo_primal_dual_leakage/summary.json"),
    "native_real_demand_service_response_v7": Path("transit_hrl/results/transit_native_real_demand_service_response_v7_48pair_merged/summary.json"),
    "native_real_demand_throughput_safe_wait_v6": Path("transit_hrl/results/transit_native_real_demand_throughput_safe_wait_v6_48pair_merged/summary.json"),
    "native_real_demand_alighting_throughput_v5": Path("transit_hrl/results/transit_native_real_demand_alighting_throughput_v5_24pair_merged/summary.json"),
    "native_real_demand_alighting_wait_v4": Path("transit_hrl/results/transit_native_real_demand_alighting_wait_v4_24pair_merged/summary.json"),
    "native_real_demand_alighting_safe_v2": Path("transit_hrl/results/transit_native_real_demand_alighting_safe_v2_24pair_merged/summary.json"),
    "native_real_demand": Path("transit_hrl/results/transit_native_real_demand_control/summary.json"),
}

ACCEPTED_PERFORMANCE_STATUSES = {
    "supported",
    "positive_mixed",
    "noninferiority_supported",
    "summary_only_positive",
    "summary_only_noharm",
}
DRIFT_METRICS = {"LowerLFDrift", "RawLowerLFDriftAbs", "LowerLFDriftAbs", "UpperHFPower"}
CORE_PERFORMANCE_METRICS = {
    "control_objective",
    "control_score",
    "reward_mean",
    "ep_reward",
    "total_return",
    "sharpe",
    "FocusScore",
    "wait_proxy",
    "avg_wait_min",
    "native_avg_board_wait_min",
    "native_boarded_pax",
    "native_alighted_pax",
    "native_completed_throughput_pax",
    "headway_cv",
    "max_drawdown",
}
NATIVE_REAL_DEMAND_REQUIRED_PERF = {
    "control_score",
    "ep_reward",
    "avg_wait_min",
    "native_avg_board_wait_min",
    "native_alighted_pax",
    "native_completed_throughput_pax",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _metric_row(domain: str, check: str, stats: dict[str, Any], status: str, source_path: Path) -> dict[str, Any]:
    return {
        "domain": str(domain),
        "check": str(check),
        "metric": str(stats.get("metric", "")),
        "treatment": str(stats.get("treatment", "")),
        "control": str(stats.get("control", "")),
        "direction": str(stats.get("direction", "")),
        "n_common": int(stats.get("n_common", 0)),
        "delta_mean": float(stats.get("delta_mean", 0.0)),
        "delta_ci95_low": float(stats.get("delta_ci95_low", 0.0)),
        "delta_ci95_high": float(stats.get("delta_ci95_high", 0.0)),
        "improvement_mean": float(stats.get("improvement_mean", 0.0)),
        "improvement_ci95_low": float(stats.get("improvement_ci95_low", 0.0)),
        "improvement_ci95_high": float(stats.get("improvement_ci95_high", 0.0)),
        "win_rate": float(stats.get("win_rate", 0.0)),
        "status": str(status),
        "source_path": str(source_path),
    }


def _paired(
    rows: list[dict[str, Any]],
    *,
    domain: str,
    pair_keys: tuple[str, ...],
    treatment: str,
    control: str,
    metric: str,
    lower_is_better: bool,
    min_pairs: int,
    source_path: Path,
    noninferiority_margin: float | None = None,
) -> dict[str, Any]:
    stats = paired_delta_stats(
        rows,
        variant_key="variant" if "variant" in rows[0] else "baseline",
        pair_keys=pair_keys,
        metric=metric,
        treatment=treatment,
        control=control,
        lower_is_better=lower_is_better,
    )
    if noninferiority_margin is None:
        status = claim_status(stats, min_pairs=int(min_pairs))
    else:
        status = noninferiority_status(
            stats,
            max_loss=float(noninferiority_margin),
            min_pairs=int(min_pairs),
        )
    return _metric_row(
        domain=domain,
        check=f"{domain}_{treatment}_vs_{control}_{metric}",
        stats=stats,
        status=status,
        source_path=source_path,
    )


def _from_existing_checks(domain: str, data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in data.get("paired_checks", []) or []:
        if not isinstance(row, dict):
            continue
        stats = dict(row)
        out.append(_metric_row(
            domain=domain,
            check=f"{domain}_{row.get('check', 'unknown')}",
            stats=stats,
            status=str(row.get("status") or claim_status(stats, min_pairs=5)),
            source_path=path,
        ))
    return out


def _summary_only_status(*, improvement: float, noninferiority_margin: float | None) -> str:
    if improvement > 0.0:
        return "summary_only_positive"
    if noninferiority_margin is not None and improvement >= -float(noninferiority_margin):
        return "summary_only_noharm"
    return "summary_only_not_supported"


def _summary_only_row(
    *,
    domain: str,
    metric: str,
    treatment: str,
    control: str,
    treatment_value: float,
    control_value: float,
    lower_is_better: bool,
    source_path: Path,
    noninferiority_margin: float | None = None,
) -> dict[str, Any]:
    delta = float(treatment_value) - float(control_value)
    improvement = -delta if lower_is_better else delta
    status = _summary_only_status(
        improvement=float(improvement),
        noninferiority_margin=noninferiority_margin,
    )
    return _metric_row(
        domain=domain,
        check=f"{domain}_{treatment}_vs_{control}_{metric}_summary_only",
        stats={
            "metric": metric,
            "treatment": treatment,
            "control": control,
            "direction": "decrease" if lower_is_better else "increase",
            "n_common": 1,
            "delta_mean": delta,
            "delta_ci95_low": delta,
            "delta_ci95_high": delta,
            "improvement_mean": improvement,
            "improvement_ci95_low": improvement,
            "improvement_ci95_high": improvement,
            "win_rate": 1.0 if improvement > 0.0 else 0.0,
        },
        status=status,
        source_path=source_path,
    )


def _ppo_trajectory_checks(domain: str, data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    history = data.get("model", {}).get("history", []) or []
    baseline = next((row for row in history if int(row.get("iteration", -99)) == -1), None)
    summary = data.get("summary", {}) or {}
    if not isinstance(baseline, dict) or not isinstance(summary, dict):
        return []
    if domain.startswith("trading"):
        specs = [
            ("LowerLFDrift", "LowerLFDrift_mean", True, None),
            ("total_return", "total_return_mean", False, 0.02),
            ("sharpe", "sharpe_mean", False, 0.75),
            ("max_drawdown", "max_drawdown_mean", True, 0.01),
        ]
    else:
        specs = [
            ("LowerLFDrift", "LowerLFDrift_mean", True, None),
            ("reward_mean", "reward_mean_mean", False, 0.05),
            ("wait_proxy", "wait_proxy_mean", True, 0.05),
            ("headway_cv", "headway_cv_mean", True, 0.01),
        ]
    out: list[dict[str, Any]] = []
    for metric, key, lower, margin in specs:
        if key not in summary or key not in baseline:
            continue
        out.append(_summary_only_row(
            domain=domain,
            metric=metric,
            treatment="ppo_primal_dual_final",
            control="ppo_primal_dual_initial",
            treatment_value=float(summary[key]),
            control_value=float(baseline[key]),
            lower_is_better=lower,
            source_path=path,
            noninferiority_margin=margin,
        ))
    return out


def _trading_checks(data: dict[str, Any], path: Path, min_pairs: int) -> list[dict[str, Any]]:
    rows = data.get("per_seed", []) or []
    if not rows:
        return []
    checks = []
    for metric, lower, margin in [
        ("LowerLFDrift", True, None),
        ("total_return", False, 0.01),
        ("sharpe", False, 0.50),
        ("FocusScore", False, 0.02),
    ]:
        checks.append(_paired(
            rows,
            domain="trading_constraint",
            pair_keys=("seed", "scenario"),
            treatment="freq_hrl",
            control="no_leakage",
            metric=metric,
            lower_is_better=lower,
            min_pairs=int(min_pairs),
            source_path=path,
            noninferiority_margin=margin,
        ))
    return checks


def _surrogate_checks(data: dict[str, Any], path: Path, min_pairs: int) -> list[dict[str, Any]]:
    rows = data.get("rows", []) or []
    if not rows:
        return []
    checks = []
    for metric, lower, margin in [
        ("LowerLFDrift", True, None),
        ("RawLowerLFDriftAbs", True, None),
        ("control_objective", False, 0.10),
        ("reward_mean", False, 0.10),
        ("wait_proxy", True, 0.10),
    ]:
        if metric not in rows[0]:
            continue
        checks.append(_paired(
            rows,
            domain="transit_real_surrogate",
            pair_keys=("source", "seed"),
            treatment="full_real_freqhrl",
            control="base_real_ema",
            metric=metric,
            lower_is_better=lower,
            min_pairs=int(min_pairs),
            source_path=path,
            noninferiority_margin=margin,
        ))
    return checks


def _domain_verdict(domain: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    group = [row for row in checks if row["domain"] == domain]
    if not group:
        return {"domain": domain, "checks": 0, "verdict": "missing"}
    drift = [
        row for row in group
        if row["metric"] in DRIFT_METRICS
    ]
    perf = [
        row for row in group
        if row["metric"] in CORE_PERFORMANCE_METRICS and row["metric"] not in DRIFT_METRICS
    ]
    drift_ok_supported = bool(drift) and any(row["status"] == "supported" for row in drift)
    drift_ok_summary = bool(drift) and any(row["status"] == "summary_only_positive" for row in drift)
    if domain.startswith("native_real_demand"):
        perf = [
            row for row in perf
            if row["metric"] in NATIVE_REAL_DEMAND_REQUIRED_PERF
        ]
        required_perf = set(NATIVE_REAL_DEMAND_REQUIRED_PERF)
    else:
        required_perf = set()
    perf_metrics = sorted({row["metric"] for row in perf})
    if required_perf:
        perf_metrics = sorted(required_perf)
    perf_ok = bool(perf) and all(
        any(
            row["metric"] == metric and row["status"] in ACCEPTED_PERFORMANCE_STATUSES
            for row in perf
        )
        for metric in perf_metrics
    )
    strict_perf_ok = bool(perf) and all(
        any(row["metric"] == metric and row["status"] == "supported" for row in perf)
        for metric in perf_metrics
    )
    if drift and drift_ok_supported and perf_ok:
        verdict = "no_tradeoff_strict_supported" if strict_perf_ok else "no_tradeoff_supported"
    elif drift and drift_ok_summary and perf_ok:
        verdict = "summary_only_noharm"
    elif not drift and perf_ok:
        verdict = "performance_noharm_only"
    elif any(row["status"] in {"supported", "summary_only_positive", "summary_only_noharm"} for row in group):
        verdict = "partial"
    else:
        verdict = "not_supported"
    return {
        "domain": domain,
        "checks": len(group),
        "drift_checks": len(drift),
        "performance_checks": len(perf),
        "required_performance_metrics": sorted(required_perf),
        "strict_performance_supported": bool(strict_perf_ok),
        "supported": sum(1 for row in group if row["status"] == "supported"),
        "noninferiority_supported": sum(1 for row in group if row["status"] == "noninferiority_supported"),
        "positive_mixed": sum(1 for row in group if row["status"] == "positive_mixed"),
        "summary_only_positive": sum(1 for row in group if row["status"] == "summary_only_positive"),
        "summary_only_noharm": sum(1 for row in group if row["status"] == "summary_only_noharm"),
        "not_supported": sum(1 for row in group if row["status"] == "not_supported"),
        "verdict": verdict,
    }


def build_leakage_matrix(result_paths: dict[str, Path], *, min_pairs: int) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    for domain, path in result_paths.items():
        data = _read_json(path)
        if data is None:
            missing.append(str(path))
            continue
        if domain == "trading_constraint":
            checks.extend(_trading_checks(data, path, min_pairs=int(min_pairs)))
        elif domain in {"trading_ppo_primal_dual", "transit_ppo_primal_dual"}:
            checks.extend(_ppo_trajectory_checks(domain, data, path))
        elif domain == "transit_real_surrogate":
            checks.extend(_surrogate_checks(data, path, min_pairs=int(min_pairs)))
        elif domain.startswith("native_real_demand"):
            checks.extend(_from_existing_checks(domain, data, path))
    domains = sorted(set(result_paths) | {row["domain"] for row in checks})
    verdicts = [_domain_verdict(domain, checks) for domain in domains]
    return {
        "paired_checks": checks,
        "domain_verdicts": verdicts,
        "missing": missing,
        "boundary": (
            "No-tradeoff is supported only when leakage/drift reduction and "
            "performance noninferiority are both supported in the same domain. "
            "Native real-demand artifacts expose wait/alighting/reward and "
            "LowerLFDrift checks; they are no-tradeoff evidence only when both "
            "drift reduction and performance noninferiority are supported."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_checks.csv", payload["paired_checks"])
    _write_csv(output_dir / "domain_verdicts.csv", payload["domain_verdicts"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Leakage No-Tradeoff Matrix",
        "",
        payload["boundary"],
        "",
        "| domain | verdict | checks | drift checks | performance checks | supported | noninferiority | positive mixed | summary positive | summary no-harm | not supported |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["domain_verdicts"]:
        lines.append(
            f"| {row['domain']} | {row['verdict']} | {row.get('checks', 0)} "
            f"| {row.get('drift_checks', 0)} | {row.get('performance_checks', 0)} "
            f"| {row.get('supported', 0)} | {row.get('noninferiority_supported', 0)} "
            f"| {row.get('positive_mixed', 0)} | {row.get('summary_only_positive', 0)} "
            f"| {row.get('summary_only_noharm', 0)} | {row.get('not_supported', 0)} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("transit_hrl/results"))
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/leakage_no_tradeoff_matrix"),
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    paths = {
        domain: root / path.relative_to("transit_hrl/results")
        for domain, path in DEFAULT_RESULT_PATHS.items()
    }
    payload = build_leakage_matrix(paths, min_pairs=int(args.min_pairs))
    write_outputs(args.output_dir, payload)
    supported = sum(
        1 for row in payload["domain_verdicts"]
        if row["verdict"] in {"no_tradeoff_supported", "no_tradeoff_strict_supported"}
    )
    print(f"leakage_no_tradeoff_matrix domains={len(payload['domain_verdicts'])} supported={supported}")


if __name__ == "__main__":
    main()
