"""Unified baseline and ablation evidence for Freq-HRL claims."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.trading.metrics import METRIC_CONTRACT_VERSION


DEFAULT_RESULT_PATHS = {
    "trading_performance": Path("transit_hrl/results/trading_performance/summary.json"),
    "trading_pressure_matrix": Path("transit_hrl/results/trading_pressure_matrix/summary.json"),
    "strong_learned_baseline_validation": Path(
        "transit_hrl/results/strong_learned_baseline_validation_latest/summary.json"
    ),
    "native_promotion_v47": Path(
        "transit_hrl/results/transit_native_promotion_v47_odshift_wait_first_512seed_summaryonly/summary.json"
    ),
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

STRONG_LEARNED_BASELINES = (
    "flat_ppo",
    "flat_sac",
    "flat_td3",
    "generic_hrl_ppo",
)

BASELINE_ROSTER = KEY_BASELINES + STRONG_LEARNED_BASELINES

CORE_METRICS = (
    ("sharpe", False),
    ("total_return", False),
    ("FocusScore", False),
    ("LowerLFDrift", True),
)

ACCEPTED_POSITIVE = {"supported", "positive_mixed"}
PROMOTION_SUPPORT_METRICS = {"ep_reward", "avg_wait_min"}
LEARNED_BASELINE_MAIN_METRICS = ("sharpe", "total_return", "FocusScore")
CONTRACT_GATED_METRICS = {"sharpe", "total_return"}


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


def _promotion_ablation_support(
    data: dict[str, Any] | None,
    *,
    source: str,
) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    supported: dict[str, dict[str, Any]] = {}
    for row in data.get("paired_checks", []) or []:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric", ""))
        treatment = str(row.get("treatment", ""))
        control = str(row.get("control", ""))
        if metric not in PROMOTION_SUPPORT_METRICS:
            continue
        if control not in {"interval_only", "no_promotion"}:
            continue
        if "wait_aware" not in treatment and treatment not in {"freq_hrl", "promotion_replan"}:
            continue
        if str(row.get("status", "")) in ACCEPTED_POSITIVE:
            supported[metric] = dict(row)
    status = "supported" if PROMOTION_SUPPORT_METRICS <= set(supported) else "missing"
    return {
        "baseline": "no_promotion",
        "source_artifact": str(source),
        "status": status,
        "supported_metrics": sorted(supported),
        "boundary": (
            "No-promotion ablation is credited from the native promotion stress "
            "artifact, where interval_only is the no-promotion control. Raw "
            "global trading Sharpe remains reported separately."
        ),
    }


def _paired_check(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    metric: str,
    lower_is_better: bool,
    min_pairs: int,
) -> dict[str, Any]:
    relevant = [
        row for row in rows
        if str(row.get("baseline", "")) in {"freq_hrl", str(baseline)}
        and metric in row
    ]
    observed_contracts = sorted({
        str(row.get("metric_contract_version", "missing")) for row in relevant
    })
    contract_valid = bool(
        metric not in CONTRACT_GATED_METRICS
        or (
            relevant
            and observed_contracts == [METRIC_CONTRACT_VERSION]
        )
    )
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
        "metric_contract_valid": contract_valid,
        "metric_contract_versions": observed_contracts,
        "status": (
            claim_status(stats, min_pairs=int(min_pairs))
            if contract_valid else "invalid_legacy_metric_contract"
        ),
    }


def _scenario_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scenarios = sorted({str(row.get("scenario", "default")) for row in rows})
    out: list[dict[str, Any]] = []
    for scenario in scenarios:
        group = [
            row for row in rows
            if str(row.get("scenario", "default")) == scenario
            and str(row.get("metric_contract_version", "")) == METRIC_CONTRACT_VERSION
        ]
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


def _learned_baseline_manifest(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks_by_control: dict[str, dict[str, dict[str, Any]]] = {}
    for row in checks:
        control = str(row.get("control", ""))
        if control not in STRONG_LEARNED_BASELINES:
            continue
        metric = str(row.get("metric", ""))
        checks_by_control.setdefault(control, {})[metric] = row

    purpose = {
        "flat_ppo": "capacity-matched factorized joint-action flat PPO baseline",
        "flat_sac": "strong off-policy entropy-regularized learned policy baseline",
        "flat_td3": "strong deterministic actor-critic learned policy baseline",
        "generic_hrl_ppo": "non-frequency learned HRL baseline with comparable hierarchy capacity",
    }
    rows: list[dict[str, Any]] = []
    for baseline in STRONG_LEARNED_BASELINES:
        metrics = checks_by_control.get(baseline, {})
        statuses = [
            str(metrics.get(metric, {}).get("status", "missing"))
            for metric in LEARNED_BASELINE_MAIN_METRICS
        ]
        supported = [
            metric for metric in LEARNED_BASELINE_MAIN_METRICS
            if str(metrics.get(metric, {}).get("status", "")) in ACCEPTED_POSITIVE
        ]
        if metrics and len(supported) == len(LEARNED_BASELINE_MAIN_METRICS):
            evidence_status = "supported"
        elif metrics and supported:
            evidence_status = "partial"
        elif metrics:
            evidence_status = "not_supported"
        else:
            evidence_status = "registered_missing"
        rows.append({
            "baseline": baseline,
            "purpose": purpose[baseline],
            "registration_status": "registered",
            "evidence_status": evidence_status,
            "required_metrics": ",".join(LEARNED_BASELINE_MAIN_METRICS),
            "supported_metrics": ",".join(supported),
            "metric_statuses": json.dumps(dict(zip(LEARNED_BASELINE_MAIN_METRICS, statuses)), sort_keys=True),
            "paper_role": "must_complete_or_limit",
            "claim_boundary": (
                "This row is not credited as a strong learned baseline unless "
                "paired evidence exists for all main metrics."
            ),
        })
    return rows


def _learned_baseline_status(manifest: list[dict[str, Any]]) -> str:
    if not manifest:
        return "missing"
    supported = [
        row for row in manifest
        if str(row.get("evidence_status", "")) == "supported"
    ]
    partial = [
        row for row in manifest
        if str(row.get("evidence_status", "")) == "partial"
    ]
    if len(supported) == len(manifest):
        return "supported"
    if supported or partial:
        return "partial"
    if all(str(row.get("evidence_status", "")) == "registered_missing" for row in manifest):
        return "registered_missing"
    return "not_supported"


def build_baseline_ablation_matrix(
    result_paths: dict[str, Path],
    *,
    min_pairs: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    ablation_support: list[dict[str, Any]] = []
    for name, path in result_paths.items():
        data = _read_json(path)
        if data is None:
            missing.append(str(path))
            continue
        rows.extend(_rows_from_payload(data, source=name))
        support = _promotion_ablation_support(data, source=name)
        if support is not None and support.get("status") != "missing":
            ablation_support.append(support)
    checks: list[dict[str, Any]] = []
    if rows:
        present_baselines = {str(row.get("baseline", "")) for row in rows}
        for baseline in BASELINE_ROSTER:
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
    required_positive_raw = [
        baseline for baseline in required_present
        if sharpe_checks[baseline].get("status") in ACCEPTED_POSITIVE
    ]
    support_overrides = {
        str(row["baseline"]): row
        for row in ablation_support
        if str(row.get("status", "")) in ACCEPTED_POSITIVE
    }
    required_present_effective = sorted(set(required_present) | (required & set(support_overrides)))
    required_positive = sorted(set(required_positive_raw) | (required & set(support_overrides)))
    required_inconclusive = [
        baseline for baseline in required_present
        if sharpe_checks[baseline].get("status") == "inconclusive"
        and baseline not in required_positive
    ]
    required_not_supported = [
        baseline for baseline in required_present
        if sharpe_checks[baseline].get("status") == "not_supported"
        and baseline not in required_positive
    ]
    required_missing = sorted(required - set(required_present_effective))
    scenario_winners = _scenario_winners(rows)
    learned_manifest = _learned_baseline_manifest(checks)
    learned_status = _learned_baseline_status(learned_manifest)
    scenario_win_rate = (
        sum(1 for row in scenario_winners if row["freq_family_wins"]) / len(scenario_winners)
        if scenario_winners else 0.0
    )
    if (
        required_present_effective
        and len(required_positive) == len(required_present_effective)
        and scenario_win_rate >= 0.50
    ):
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
            "required_baselines_present": required_present_effective,
            "required_baselines_present_raw": required_present,
            "required_baselines_positive": required_positive,
            "required_baselines_positive_raw": required_positive_raw,
            "required_baselines_inconclusive": required_inconclusive,
            "required_baselines_not_supported": required_not_supported,
            "required_baselines_missing": required_missing,
            "ablation_support_overrides": ablation_support,
            "learned_baseline_manifest": learned_manifest,
            "strong_learned_baseline_status": learned_status,
            "scenario_count": len(scenario_winners),
            "scenario_freq_family_win_rate": float(scenario_win_rate),
            "claim_status": claim_status_value,
        },
        "boundary": (
            "Baseline/ablation evidence is paired over identical seeds and "
            "stress scenarios. It checks whether Freq-HRL beats non-frequency, "
            "misrouted-frequency, no-promotion, and no-leakage alternatives; "
            "it does not replace native Transit learned-policy validation. "
            "Flat PPO/SAC/TD3 and generic learned HRL are registered separately "
            "and are not credited unless their paired rows are present."
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
    _write_csv(
        output_dir / "learned_baseline_manifest.csv",
        payload["summary"].get("learned_baseline_manifest", []),
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Baseline And Ablation Matrix",
        "",
        payload["boundary"],
        "",
        f"- claim status: `{payload['summary']['claim_status']}`",
        f"- scenario Freq-HRL-family win rate: `{payload['summary']['scenario_freq_family_win_rate']:.3f}`",
        f"- required baselines positive: `{payload['summary'].get('required_baselines_positive', [])}`",
        f"- support overrides: `{payload['summary'].get('ablation_support_overrides', [])}`",
        f"- strong learned baseline status: `{payload['summary'].get('strong_learned_baseline_status', '')}`",
        f"- required baselines inconclusive: `{payload['summary'].get('required_baselines_inconclusive', [])}`",
        f"- required baselines not supported: `{payload['summary'].get('required_baselines_not_supported', [])}`",
        f"- required baselines missing: `{payload['summary'].get('required_baselines_missing', [])}`",
        "",
        "## Strong Learned Baseline Registration",
        "",
        "| baseline | evidence status | required metrics | supported metrics | paper role |",
        "|---|---|---|---|---|",
    ]
    for row in payload["summary"].get("learned_baseline_manifest", []):
        lines.append(
            f"| {row['baseline']} | {row['evidence_status']} | "
            f"{row['required_metrics']} | {row['supported_metrics']} | {row['paper_role']} |"
        )
    lines.extend([
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
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
