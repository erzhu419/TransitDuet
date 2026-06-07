"""Build a cross-domain encoder evidence matrix for Freq-HRL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats


DEFAULT_RESULT_PATHS = {
    "trading_synthetic": Path("transit_hrl/results/trading_encoder_ablation_adaptive/summary.json"),
    "trading_synthetic_neural": Path("transit_hrl/results/trading_encoder_ablation_neural/summary.json"),
    "public_market_daily": Path("transit_hrl/results/trading_public_market_encoder_ablation/summary.json"),
    "public_market_intraday": Path("transit_hrl/results/trading_public_market_intraday_encoder_ablation/summary.json"),
    "order_book_l2": Path("transit_hrl/results/trading_order_book_matching_validation/summary.json"),
    "order_book_l3": Path("transit_hrl/results/trading_order_book_l3_replay_validation/summary.json"),
    "transit_synthetic_demand": Path("transit_hrl/results/transit_demand_estimator_validation/summary.json"),
    "transit_real_demand": Path("transit_hrl/results/transit_local_demand_estimator_validation/summary.json"),
}

TRADING_TREATMENTS = ("state_space", "haar_wavelet", "adaptive_wavelet", "neural_state_space")
TRADING_METRICS = (
    ("sharpe", False),
    ("total_return", False),
    ("max_drawdown", True),
    ("LowerLFDrift", True),
    ("FocusScore", False),
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _check_row(
    *,
    domain: str,
    check: str,
    metric: str,
    treatment: str,
    control: str,
    direction: str,
    n_common: int,
    delta_mean: float,
    delta_ci95_low: float,
    delta_ci95_high: float,
    improvement_mean: float,
    improvement_ci95_low: float,
    improvement_ci95_high: float,
    win_rate: float,
    status: str,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "domain": str(domain),
        "check": str(check),
        "metric": str(metric),
        "treatment": str(treatment),
        "control": str(control),
        "direction": str(direction),
        "n_common": int(n_common),
        "delta_mean": float(delta_mean),
        "delta_ci95_low": float(delta_ci95_low),
        "delta_ci95_high": float(delta_ci95_high),
        "improvement_mean": float(improvement_mean),
        "improvement_ci95_low": float(improvement_ci95_low),
        "improvement_ci95_high": float(improvement_ci95_high),
        "win_rate": float(win_rate),
        "status": str(status),
        "source_path": str(source_path),
    }


def _rows_from_existing_checks(domain: str, data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in data.get("paired_checks", []) or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or claim_status(row, min_pairs=5))
        out.append(_check_row(
            domain=domain,
            check=str(row.get("check", "unknown")),
            metric=str(row.get("metric", "unknown")),
            treatment=str(row.get("treatment", "")),
            control=str(row.get("control", "")),
            direction=str(row.get("direction", "")),
            n_common=int(row.get("n_common", 0)),
            delta_mean=float(row.get("delta_mean", 0.0)),
            delta_ci95_low=float(row.get("delta_ci95_low", 0.0)),
            delta_ci95_high=float(row.get("delta_ci95_high", 0.0)),
            improvement_mean=float(row.get("improvement_mean", row.get("delta_mean", 0.0))),
            improvement_ci95_low=float(row.get("improvement_ci95_low", row.get("delta_ci95_low", 0.0))),
            improvement_ci95_high=float(row.get("improvement_ci95_high", row.get("delta_ci95_high", 0.0))),
            win_rate=float(row.get("win_rate", 0.0)),
            status=status,
            source_path=path,
        ))
    return out


def _rows_from_paired_deltas(domain: str, data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in data.get("paired_deltas", []) or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or claim_status(row, min_pairs=5))
        out.append(_check_row(
            domain=domain,
            check=str(row.get("comparison", "unknown")),
            metric=str(row.get("metric", "unknown")),
            treatment=str(row.get("treatment", "")),
            control=str(row.get("control", "")),
            direction=str(row.get("direction", "")),
            n_common=int(row.get("n_common", 0)),
            delta_mean=float(row.get("delta_mean", 0.0)),
            delta_ci95_low=float(row.get("delta_ci95_low", 0.0)),
            delta_ci95_high=float(row.get("delta_ci95_high", 0.0)),
            improvement_mean=float(row.get("improvement_mean", row.get("delta_mean", 0.0))),
            improvement_ci95_low=float(row.get("improvement_ci95_low", row.get("delta_ci95_low", 0.0))),
            improvement_ci95_high=float(row.get("improvement_ci95_high", row.get("delta_ci95_high", 0.0))),
            win_rate=float(row.get("win_rate", 0.0)),
            status=status,
            source_path=path,
        ))
    return out


def _rows_from_trading_per_seed(domain: str, data: dict[str, Any], path: Path, min_pairs: int) -> list[dict[str, Any]]:
    rows = data.get("per_seed", []) or []
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    treatments = sorted({
        str(row.get("freq_method", ""))
        for row in rows
        if str(row.get("freq_method", "")) not in {"", "ema"}
    })
    for treatment in treatments:
        if treatment not in TRADING_TREATMENTS:
            continue
        for metric, lower_is_better in TRADING_METRICS:
            if not any(metric in row for row in rows):
                continue
            stats = paired_delta_stats(
                rows,
                variant_key="freq_method",
                pair_keys=("seed", "scenario"),
                metric=metric,
                treatment=treatment,
                control="ema",
                lower_is_better=lower_is_better,
            )
            out.append(_check_row(
                domain=domain,
                check=f"{treatment}_vs_ema_{metric}",
                metric=metric,
                treatment=treatment,
                control="ema",
                direction=str(stats.get("direction", "")),
                n_common=int(stats.get("n_common", 0)),
                delta_mean=float(stats.get("delta_mean", 0.0)),
                delta_ci95_low=float(stats.get("delta_ci95_low", 0.0)),
                delta_ci95_high=float(stats.get("delta_ci95_high", 0.0)),
                improvement_mean=float(stats.get("improvement_mean", 0.0)),
                improvement_ci95_low=float(stats.get("improvement_ci95_low", 0.0)),
                improvement_ci95_high=float(stats.get("improvement_ci95_high", 0.0)),
                win_rate=float(stats.get("win_rate", 0.0)),
                status=claim_status(stats, min_pairs=int(min_pairs)),
                source_path=path,
            ))
    return out


def _rows_from_public_summary(domain: str, data: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = data.get("summary", []) or []
    baseline = next((row for row in rows if row.get("freq_method") == "ema"), None)
    if not baseline:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        treatment = str(row.get("freq_method", ""))
        if treatment in {"", "ema"}:
            continue
        for metric, lower_is_better in [
            ("sharpe", False),
            ("total_return", False),
            ("max_drawdown", True),
            ("turnover", True),
        ]:
            if metric not in row or metric not in baseline:
                continue
            delta = float(row[metric]) - float(baseline[metric])
            improvement = -delta if lower_is_better else delta
            out.append(_check_row(
                domain=domain,
                check=f"{treatment}_vs_ema_{metric}",
                metric=metric,
                treatment=treatment,
                control="ema",
                direction="decrease" if lower_is_better else "increase",
                n_common=1,
                delta_mean=delta,
                delta_ci95_low=delta,
                delta_ci95_high=delta,
                improvement_mean=improvement,
                improvement_ci95_low=improvement,
                improvement_ci95_high=improvement,
                win_rate=1.0 if improvement > 0 else 0.0,
                status="summary_only",
                source_path=path,
            ))
    return out


def build_encoder_matrix(
    result_paths: dict[str, Path],
    *,
    min_pairs: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    for domain, path in result_paths.items():
        data = _read_json(path)
        if data is None:
            missing.append(str(path))
            continue
        if domain.startswith("trading_synthetic"):
            checks.extend(_rows_from_trading_per_seed(domain, data, path, min_pairs=int(min_pairs)))
        elif domain.startswith("public_market"):
            checks.extend(_rows_from_public_summary(domain, data, path))
        elif domain.startswith("order_book"):
            checks.extend(_rows_from_existing_checks(domain, data, path))
        elif domain.startswith("transit"):
            checks.extend(_rows_from_paired_deltas(domain, data, path))
    domain_summary: list[dict[str, Any]] = []
    for domain in sorted({row["domain"] for row in checks}):
        group = [row for row in checks if row["domain"] == domain]
        supported = sum(1 for row in group if row["status"] == "supported")
        positive = sum(1 for row in group if row["status"] == "positive_mixed")
        not_supported = sum(1 for row in group if row["status"] == "not_supported")
        summary_only = sum(1 for row in group if row["status"] == "summary_only")
        domain_summary.append({
            "domain": domain,
            "checks": len(group),
            "supported": supported,
            "positive_mixed": positive,
            "not_supported": not_supported,
            "summary_only": summary_only,
            "best_status": "supported" if supported else (
                "positive_mixed" if positive else ("summary_only" if summary_only else "not_supported")
            ),
        })
    return {
        "paired_checks": checks,
        "domain_summary": domain_summary,
        "missing": missing,
        "boundary": (
            "Cross-domain encoder matrix assembled from existing experiment "
            "artifacts. Public market rows without paired seeds are marked "
            "summary_only; scheduler reruns should replace them with paired "
            "multi-seed or multi-window checks."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "paired_checks.csv", payload["paired_checks"])
    _write_csv(output_dir / "domain_summary.csv", payload["domain_summary"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Encoder Cross-Domain Matrix",
        "",
        payload["boundary"],
        "",
        "## Domain Summary",
        "",
        "| domain | checks | supported | positive mixed | not supported | summary only | best status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["domain_summary"]:
        lines.append(
            f"| {row['domain']} | {row['checks']} | {row['supported']} "
            f"| {row['positive_mixed']} | {row['not_supported']} "
            f"| {row['summary_only']} | {row['best_status']} |"
        )
    lines.extend([
        "",
        "## Checks",
        "",
        "| domain | check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["paired_checks"]:
        lines.append(
            f"| {row['domain']} | {row['check']} | {row['status']} | {row['metric']} "
            f"| {row['n_common']} | {row['delta_mean']:+.4f} "
            f"| {row['delta_ci95_low']:+.4f} | {row['delta_ci95_high']:+.4f} "
            f"| {row['win_rate']:.2f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("transit_hrl/results"))
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/encoder_cross_domain_matrix"),
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    paths = {
        domain: root / path.relative_to("transit_hrl/results")
        for domain, path in DEFAULT_RESULT_PATHS.items()
    }
    payload = build_encoder_matrix(paths, min_pairs=int(args.min_pairs))
    write_outputs(args.output_dir, payload)
    supported = sum(1 for row in payload["paired_checks"] if row["status"] == "supported")
    print(
        "encoder_cross_domain_matrix "
        f"checks={len(payload['paired_checks'])} supported={supported}"
    )


if __name__ == "__main__":
    main()
