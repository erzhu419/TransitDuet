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
    "transit_real_surrogate": Path("transit_hrl/results/transit_real_demand_control/summary.json"),
    "native_real_demand": Path("transit_hrl/results/transit_native_real_demand_control/summary.json"),
    "native_real_demand_waitaware_v2": Path("transit_hrl/results/transit_native_real_demand_waitaware_v2_24seed_merged/summary.json"),
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
        if row["metric"] in {"LowerLFDrift", "RawLowerLFDriftAbs", "LowerLFDriftAbs"}
    ]
    perf = [
        row for row in group
        if row["metric"] not in {"LowerLFDrift", "RawLowerLFDriftAbs", "LowerLFDriftAbs"}
    ]
    drift_ok = bool(drift) and any(row["status"] == "supported" for row in drift)
    perf_ok = bool(perf) and all(row["status"] in {"supported", "positive_mixed", "noninferiority_supported"} for row in perf)
    if drift and drift_ok and perf_ok:
        verdict = "no_tradeoff_supported"
    elif not drift and perf_ok:
        verdict = "performance_noharm_only"
    elif any(row["status"] == "supported" for row in group):
        verdict = "partial"
    else:
        verdict = "not_supported"
    return {
        "domain": domain,
        "checks": len(group),
        "drift_checks": len(drift),
        "performance_checks": len(perf),
        "supported": sum(1 for row in group if row["status"] == "supported"),
        "noninferiority_supported": sum(1 for row in group if row["status"] == "noninferiority_supported"),
        "positive_mixed": sum(1 for row in group if row["status"] == "positive_mixed"),
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
            "Native real-demand artifacts currently expose wait/alighting/reward "
            "checks but not LowerLFDrift, so they are performance/no-harm evidence "
            "unless native drift metrics are added."
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
        "| domain | verdict | checks | drift checks | performance checks | supported | noninferiority | positive mixed | not supported |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["domain_verdicts"]:
        lines.append(
            f"| {row['domain']} | {row['verdict']} | {row.get('checks', 0)} "
            f"| {row.get('drift_checks', 0)} | {row.get('performance_checks', 0)} "
            f"| {row.get('supported', 0)} | {row.get('noninferiority_supported', 0)} "
            f"| {row.get('positive_mixed', 0)} | {row.get('not_supported', 0)} |"
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
    supported = sum(1 for row in payload["domain_verdicts"] if row["verdict"] == "no_tradeoff_supported")
    print(f"leakage_no_tradeoff_matrix domains={len(payload['domain_verdicts'])} supported={supported}")


if __name__ == "__main__":
    main()
