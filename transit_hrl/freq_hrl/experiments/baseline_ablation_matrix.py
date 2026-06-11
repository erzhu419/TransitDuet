"""Unified baseline and ablation evidence for Freq-HRL claims."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats


DEFAULT_RESULT_PATHS = {
    "trading_performance": Path("transit_hrl/results/trading_performance/summary.json"),
    "trading_pressure_matrix": Path("transit_hrl/results/trading_pressure_matrix/summary.json"),
}

KEY_BASELINES = (
    "vanilla_rl",
    "hrl_raw",
    "raw_history",
    "freq_single_policy",
    "allfreq_alllayers",
    "swapped",
    "no_promotion",
    "no_leakage",
    "lf_upper_only",
    "hf_lower_only",
)

CORE_METRICS = (
    ("sharpe", False),
    ("total_return", False),
    ("FocusScore", False),
    ("LowerLFDrift", True),
)

ACCEPTED_POSITIVE = {"supported", "positive_mixed"}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _rows_from_payload(data: dict[str, Any] | None, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        return []
    rows = data.get("per_seed", []) or []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["source_artifact"] = str(source)
        item.setdefault("scenario", "default")
        out.append(item)
    return out


def _paired_check(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    metric: str,
    lower_is_better: bool,
    min_pairs: int,
) -> dict[str, Any]:
    stats = paired_delta_stats(
        rows,
        variant_key="baseline",
        pair_keys=("source_artifact", "scenario", "seed"),
        metric=metric,
        treatment="freq_hrl",
        control=str(baseline),
        lower_is_better=lower_is_better,
    )
    return {
        "check": f"freq_hrl_vs_{baseline}_{metric}",
        **stats,
        "status": claim_status(stats, min_pairs=int(min_pairs)),
    }


def _scenario_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios = sorted({str(row.get("scenario", "default")) for row in rows})
    out: list[dict[str, Any]] = []
    for scenario in scenarios:
        group = [row for row in rows if str(row.get("scenario", "default")) == scenario]
        if not group:
            continue
        baselines = sorted({str(row.get("baseline", "")) for row in group})
        means: dict[str, float] = {}
        for baseline in baselines:
            vals = [
                float(row.get("sharpe", 0.0))
                for row in group
                if str(row.get("baseline", "")) == baseline
            ]
            if vals:
                means[baseline] = sum(vals) / len(vals)
        if not means:
            continue
        best = max(means, key=means.get)
        freq = means.get("freq_hrl")
        tuned = means.get("freq_hrl_recovery_tuned")
        freq_family_best = max(
            [value for value in (freq, tuned) if value is not None],
            default=None,
        )
        out.append({
            "scenario": scenario,
            "best_baseline": best,
            "best_sharpe": float(means[best]),
            "freq_hrl_sharpe": float(freq) if freq is not None else None,
            "freq_hrl_recovery_tuned_sharpe": float(tuned) if tuned is not None else None,
            "freq_family_best_sharpe": float(freq_family_best) if freq_family_best is not None else None,
            "freq_family_wins": bool(freq_family_best is not None and freq_family_best >= means[best] - 1e-12),
        })
    return out


def build_baseline_ablation_matrix(
    result_paths: dict[str, Path],
    *,
    min_pairs: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, path in result_paths.items():
        data = _read_json(path)
        if data is None:
            missing.append(str(path))
            continue
        rows.extend(_rows_from_payload(data, source=name))
    checks: list[dict[str, Any]] = []
    if rows:
        present_baselines = {str(row.get("baseline", "")) for row in rows}
        for baseline in KEY_BASELINES:
            if baseline not in present_baselines:
                continue
            for metric, lower_is_better in CORE_METRICS:
                if metric not in rows[0]:
                    continue
                checks.append(_paired_check(
                    rows,
                    baseline=baseline,
                    metric=metric,
                    lower_is_better=lower_is_better,
                    min_pairs=int(min_pairs),
                ))
    required = {"vanilla_rl", "hrl_raw", "allfreq_alllayers", "swapped", "no_promotion", "no_leakage"}
    sharpe_checks = {
        row["control"]: row
        for row in checks
        if row.get("metric") == "sharpe"
    }
    required_present = sorted(required & set(sharpe_checks))
    required_positive = [
        baseline for baseline in required_present
        if sharpe_checks[baseline].get("status") in ACCEPTED_POSITIVE
    ]
    scenario_winners = _scenario_winners(rows)
    scenario_win_rate = (
        sum(1 for row in scenario_winners if row["freq_family_wins"]) / len(scenario_winners)
        if scenario_winners else 0.0
    )
    if required_present and len(required_positive) == len(required_present) and scenario_win_rate >= 0.50:
        claim_status_value = "supported"
    elif required_positive:
        claim_status_value = "partial"
    elif rows:
        claim_status_value = "not_supported"
    else:
        claim_status_value = "missing"
    return {
        "rows": rows,
        "paired_checks": checks,
        "scenario_winners": scenario_winners,
        "summary": {
            "rows": len(rows),
            "missing": missing,
            "required_baselines_present": required_present,
            "required_baselines_positive": required_positive,
            "scenario_count": len(scenario_winners),
            "scenario_freq_family_win_rate": float(scenario_win_rate),
            "claim_status": claim_status_value,
        },
        "boundary": (
            "Baseline/ablation evidence is paired over identical seeds and "
            "stress scenarios. It checks whether Freq-HRL beats non-frequency, "
            "misrouted-frequency, no-promotion, and no-leakage alternatives; "
            "it does not replace native Transit learned-policy validation."
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
    _write_csv(output_dir / "scenario_winners.csv", payload["scenario_winners"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Baseline And Ablation Matrix",
        "",
        payload["boundary"],
        "",
        f"- claim status: `{payload['summary']['claim_status']}`",
        f"- scenario Freq-HRL-family win rate: `{payload['summary']['scenario_freq_family_win_rate']:.3f}`",
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["paired_checks"]:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['metric']} "
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
        default=Path("transit_hrl/results/baseline_ablation_matrix"),
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    paths = {
        key: root / path.relative_to("transit_hrl/results")
        for key, path in DEFAULT_RESULT_PATHS.items()
    }
    payload = build_baseline_ablation_matrix(paths, min_pairs=int(args.min_pairs))
    write_outputs(args.output_dir, payload)
    print(
        "baseline_ablation_matrix "
        f"status={payload['summary']['claim_status']} "
        f"rows={payload['summary']['rows']}"
    )


if __name__ == "__main__":
    main()
