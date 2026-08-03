"""Unified top-journal evidence matrix for Freq-HRL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.evidence_policy import (
    OBSERVED_EVIDENCE,
    PROJECTION_EVIDENCE,
    annotate_check,
    is_headline_eligible,
)
from freq_hrl.experiments.statistics import (
    claim_status,
    noninferiority_status,
    paired_delta_stats,
)


PRIMARY_PROMOTION_ARTIFACT = "native_promotion_v47_odshift"
PERSISTENT_PROMOTION_ARTIFACT = "native_promotion_v42"
ODSHIFT_PROMOTION_ARTIFACT = "native_promotion_v47_odshift"
PRIMARY_REAL_DEMAND_ARTIFACT = "native_real_demand_service_response_v7"

DEFAULT_ARTIFACTS = {
    "native_promotion_v47_odshift": Path("transit_hrl/results/transit_native_promotion_v47_odshift_wait_first_512seed_summaryonly/summary.json"),
    "native_promotion_v46_odshift": Path("transit_hrl/results/scheduler_native_promotion_v46_odshift_reward_wait_guard_512seed_merged/summary.json"),
    "native_promotion_v45_odshift": Path("transit_hrl/results/scheduler_native_promotion_v45_odshift_reward_floor_active_512seed_merged/summary.json"),
    "native_promotion_v44_odshift": Path("transit_hrl/results/scheduler_native_promotion_v44_odshift_reward_floor_smoke64_merged_retry3_retry4/summary.json"),
    "native_promotion_v42_odshift": Path("transit_hrl/results/transit_native_promotion_v42_odshift_512seed_merged/summary.json"),
    "native_promotion_v42": Path("transit_hrl/results/scheduler_native_promotion_risk_banded_delta_floor_v42_512seed_merged/summary.json"),
    "native_promotion_v32": Path("transit_hrl/results/transit_native_promotion_reward_guarded_highpressure_wait_v32_512seed_w16_evidence/summary.json"),
    "native_promotion_v31": Path("transit_hrl/results/transit_native_promotion_final_delta_floor_reward_wait_v31_512seed_w16r2_merged/summary.json"),
    "native_promotion_v27": Path("transit_hrl/results/transit_native_promotion_selective_reward_wait_v27_512seed_merged/summary.json"),
    "native_promotion_v26": Path("transit_hrl/results/transit_native_promotion_reward_floor_throughput_v26_512seed_merged/summary.json"),
    "native_promotion_v25": Path("transit_hrl/results/transit_native_promotion_reward_floor_throughput_v25_512seed_merged/summary.json"),
    "native_promotion_v24_fixed": Path("transit_hrl/results/transit_native_promotion_pressure_guarded_wait_v24_2048seed_fixed_w32_evidence/summary.json"),
    "native_promotion_v24": Path("transit_hrl/results/transit_native_promotion_pressure_guarded_wait_v24_2048seed_merged/summary.json"),
    "native_promotion_v21": Path("transit_hrl/results/transit_native_promotion_reward_guarded_projected_wait_v21_8192seed_w32x6_merged/summary.json"),
    "native_real_demand_service_response_v7": Path("transit_hrl/results/transit_native_real_demand_service_response_v7_48pair_merged/summary.json"),
    "native_real_demand_throughput_safe_wait_v6": Path("transit_hrl/results/transit_native_real_demand_throughput_safe_wait_v6_48pair_merged/summary.json"),
    "native_real_demand_alighting_throughput_v5": Path("transit_hrl/results/transit_native_real_demand_alighting_throughput_v5_24pair_merged/summary.json"),
    "native_real_demand_alighting_wait_v4": Path("transit_hrl/results/transit_native_real_demand_alighting_wait_v4_24pair_merged/summary.json"),
    "native_real_demand_alighting_safe_v2": Path("transit_hrl/results/transit_native_real_demand_alighting_safe_v2_24pair_merged/summary.json"),
    "native_real_demand_v5": Path("transit_hrl/results/scheduler_native_real_demand_selective_reward_wait_v5_24pair/summary.json"),
    "native_real_demand_v4": Path("transit_hrl/results/scheduler_native_real_demand_wait_pressure_v4_24pair/summary.json"),
    "native_real_demand_v3": Path("transit_hrl/results/scheduler_native_real_demand_reward_floor_throughput_v3_24pair/summary.json"),
    "native_real_demand_v2": Path("transit_hrl/results/transit_native_real_demand_waitaware_v2_24seed_merged_drift/summary.json"),
    "agency_demand_onboard_coverage": Path("transit_hrl/results/agency_demand_onboard_coverage_latest/summary.json"),
    "order_book_l2_matching": Path("transit_hrl/results/trading_order_book_matching_validation/summary.json"),
    "order_book_l3_replay": Path("transit_hrl/results/trading_order_book_l3_replay_validation/summary.json"),
    "order_book_manifest": Path("transit_hrl/results/order_book_lobster_venue_grade_multisymbol/summary.json"),
    "encoder_matrix": Path("transit_hrl/results/encoder_cross_domain_matrix/summary.json"),
    "encoder_matrix_latest": Path("transit_hrl/results/encoder_cross_domain_matrix_latest/summary.json"),
    "leakage_matrix_latest_patch": Path("transit_hrl/results/leakage_no_tradeoff_matrix_latest_patch/summary.json"),
    "leakage_matrix_v27_v5": Path("transit_hrl/results/scheduler_leakage_no_tradeoff_matrix_v27_v5/summary.json"),
    "leakage_matrix_v26_v4": Path("transit_hrl/results/scheduler_leakage_no_tradeoff_matrix_v26_v4/summary.json"),
    "leakage_matrix_v25_v3": Path("transit_hrl/results/scheduler_leakage_no_tradeoff_matrix_v25_v3/summary.json"),
    "leakage_matrix": Path("transit_hrl/results/leakage_no_tradeoff_matrix/summary.json"),
    "leakage_matrix_latest": Path("transit_hrl/results/leakage_no_tradeoff_matrix_latest/summary.json"),
    "theory_appendix_latest": Path("transit_hrl/results/freq_hrl_theory_appendix_latest/summary.json"),
    "theory_appendix": Path("transit_hrl/results/freq_hrl_theory_appendix/summary.json"),
    "theory_appendix_scheduler": Path("transit_hrl/results/scheduler_freq_hrl_theory_appendix/summary.json"),
    "baseline_ablation_matrix_latest": Path("transit_hrl/results/baseline_ablation_matrix_latest/summary.json"),
    "baseline_ablation_matrix": Path("transit_hrl/results/baseline_ablation_matrix/summary.json"),
    "strong_learned_baseline_validation": Path("transit_hrl/results/strong_learned_baseline_validation_latest/summary.json"),
    "trading_pressure_matrix": Path("transit_hrl/results/trading_pressure_matrix/summary.json"),
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _check_by_metric(
    data: dict[str, Any] | None,
    metric: str,
    *,
    treatment: str = "",
    check_contains: str = "",
) -> dict[str, Any]:
    if not data:
        return {}
    for row in data.get("paired_checks", []) or []:
        if row.get("metric") != metric:
            continue
        if treatment and row.get("treatment") != treatment:
            continue
        if check_contains and check_contains not in str(row.get("check", "")):
            continue
        return row
    return {}


def _supported(row: dict[str, Any]) -> bool:
    return is_headline_eligible(row) and str(row.get("status", "")) == "supported"


def _positive(row: dict[str, Any]) -> bool:
    return is_headline_eligible(row) and str(row.get("status", "")) in {
        "supported", "positive_mixed", "noninferiority_supported"
    }


def _mean_improved(row: dict[str, Any]) -> bool:
    try:
        return float(row.get("improvement_mean", 0.0)) > 0.0
    except (TypeError, ValueError):
        return False


def _status_from_flags(*, present: bool, supported: bool, partial: bool) -> str:
    if not present:
        return "missing"
    if supported:
        return "supported"
    if partial:
        return "partial"
    return "not_supported"


def _count_checks(data: dict[str, Any], statuses: set[str]) -> int:
    if not isinstance(data, dict):
        return 0
    return sum(
        1
        for row in data.get("paired_checks", []) or []
        if str(row.get("status", "")) in statuses
    )


def _has_non_synthetic_sources(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    sources = data.get("sources", []) or []
    return any("synthetic" not in str(source).lower() for source in sources)


def _observed_legacy_check(
    data: dict[str, Any] | None,
    metric: str,
    *,
    treatment: str,
    check_contains: str = "",
) -> dict[str, Any]:
    row = _check_by_metric(
        data,
        metric,
        treatment=treatment,
        check_contains=check_contains,
    )
    return (
        annotate_check(row, evidence_class=OBSERVED_EVIDENCE, headline_eligible=True)
        if row else {}
    )


def _promotion_artifact_evidence(
    artifacts: dict[str, dict[str, Any] | None],
    paths: dict[str, str],
    key: str,
) -> dict[str, Any]:
    data = artifacts.get(key)
    if not data:
        return {
            "key": "missing",
            "status": "missing",
            "reward": {},
            "reward_noninferiority": {},
            "wait": {},
            "wait_noninferiority": {},
            "control_score": {},
            "artifact": paths[key],
            "selection_policy": "frozen_artifact_key",
        }
    reward = _observed_legacy_check(
        data, "promotion_raw_ep_reward", treatment="native_wait_aware_replan"
    )
    wait = _observed_legacy_check(
        data, "promotion_raw_avg_wait_min", treatment="native_wait_aware_replan"
    )
    score = _observed_legacy_check(
        data, "promotion_raw_score", treatment="native_wait_aware_replan"
    )
    status = _status_from_flags(
        present=True,
        supported=_supported(reward) and _supported(wait),
        partial=_mean_improved(reward) or _mean_improved(wait),
    )
    return {
        "key": key,
        "status": status,
        "reward": reward,
        "reward_noninferiority": {},
        "wait": wait,
        "wait_noninferiority": {},
        "control_score": score,
        "best_score": score,
        "artifact": paths[key],
        "selection_policy": "frozen_artifact_key",
        "raw_only": True,
    }


def _promotion_evidence(
    artifacts: dict[str, dict[str, Any] | None],
    paths: dict[str, str],
) -> dict[str, Any]:
    return _promotion_artifact_evidence(
        artifacts, paths, PRIMARY_PROMOTION_ARTIFACT
    )


def _promotion_cross_stress_evidence(
    artifacts: dict[str, dict[str, Any] | None],
    paths: dict[str, str],
) -> dict[str, Any]:
    persistent = _promotion_artifact_evidence(
        artifacts, paths, PERSISTENT_PROMOTION_ARTIFACT
    )
    odshift = _promotion_artifact_evidence(
        artifacts, paths, ODSHIFT_PROMOTION_ARTIFACT
    )
    persistent_ok = persistent.get("status") == "supported"
    odshift_supported = odshift.get("status") == "supported"
    status = _status_from_flags(
        present=bool(persistent.get("key") != "missing" or odshift.get("key") != "missing"),
        supported=bool(persistent_ok and odshift_supported),
        partial=bool(persistent_ok or odshift_supported),
    )
    return {
        "status": status,
        "persistent_key": persistent.get("key", "missing"),
        "persistent_status": persistent.get("status", "missing"),
        "odshift_key": odshift.get("key", "missing"),
        "odshift_reward": odshift.get("reward", {}),
        "odshift_wait": odshift.get("wait", {}),
        "odshift_reward_noninferiority": {},
        "odshift_wait_noninferiority": {},
        "artifact": f"{persistent.get('artifact', '')} | {odshift.get('artifact', '')}",
        "selection_policy": "two_distinct_frozen_artifact_keys",
    }


def _observed_real_demand_rows(data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not data:
        return []
    aliases = {
        "avg_wait_min": "native_raw_avg_wait_min",
        "native_avg_board_wait_min": "native_raw_native_avg_board_wait_min",
        "native_boarded_pax": "native_raw_native_boarded_pax",
        "native_alighted_pax": "native_raw_native_alighted_pax",
        "native_completed_throughput_pax": "native_raw_native_completed_throughput_pax",
        "native_unalighted_pax": "native_raw_native_unalighted_pax",
        "LowerLFDrift": "native_raw_LowerLFDrift",
    }
    observed: list[dict[str, Any]] = []
    for source in data.get("rows", []) or []:
        row = dict(source)
        for canonical, raw_key in aliases.items():
            if raw_key in source:
                row[canonical] = float(source[raw_key])
        completed = min(
            float(row.get("native_boarded_pax", 0.0)),
            float(row.get("native_alighted_pax", 0.0)),
        )
        row["native_completed_throughput_pax"] = completed
        row["native_unalighted_pax"] = max(
            float(row.get("native_boarded_pax", 0.0))
            - float(row.get("native_alighted_pax", 0.0)),
            0.0,
        )
        row["control_score"] = (
            float(row.get("ep_reward", 0.0))
            - 10.0 * float(row.get("avg_wait_min", 0.0))
            - 2.0 * float(row.get("headway_cv", 0.0))
            - 0.5 * float(row.get("native_avg_board_wait_min", 0.0))
            + 25.0 * completed
        )
        observed.append(row)
    return observed


def _real_demand_observed_check(
    data: dict[str, Any] | None,
    metric: str,
    *,
    lower_is_better: bool,
    noninferiority_margin: float | None = None,
) -> dict[str, Any]:
    rows = _observed_real_demand_rows(data)
    if not rows:
        return {}
    stats = paired_delta_stats(
        rows,
        variant_key="variant",
        pair_keys=("source", "seed"),
        metric=metric,
        treatment="native_real_freqhrl",
        control="native_real_interval",
        lower_is_better=lower_is_better,
    )
    min_pairs = max(3, int((data or {}).get("min_pairs", 3) or 3))
    if noninferiority_margin is None:
        status = claim_status(stats, min_pairs=min_pairs)
        check = f"native_real_demand_observed_{metric}"
    else:
        status = noninferiority_status(
            stats,
            max_loss=float(noninferiority_margin),
            min_pairs=min_pairs,
        )
        check = f"native_real_demand_observed_{metric}_noninferiority"
    return annotate_check({
        "check": check,
        **stats,
        "status": status,
        "noninferiority_margin": noninferiority_margin,
        "recomputed_from": "per-seed raw simulator fields",
    }, evidence_class=OBSERVED_EVIDENCE, headline_eligible=True)


def _pressure_stress_evidence(
    baseline_ablation: dict[str, Any],
    pressure_matrix: dict[str, Any],
) -> dict[str, Any]:
    required = [
        "stationary_low_noise",
        "stationary_high_noise",
        "localized_burst",
        "persistent_shift",
        "ood_period",
    ]
    winners = {
        str(row.get("scenario")): row
        for row in baseline_ablation.get("scenario_winners", []) or []
        if isinstance(row, dict)
    }
    pressure_rows = pressure_matrix.get("per_seed", []) if isinstance(pressure_matrix, dict) else []
    observed = sorted({str(row.get("scenario", "")) for row in pressure_rows if isinstance(row, dict)})
    regime_status: dict[str, str] = {}
    for regime in required:
        if regime in winners:
            regime_status[regime] = "supported" if bool(winners[regime].get("freq_family_wins")) else "not_supported"
        elif regime in observed:
            regime_status[regime] = "observed_without_ablation_winner"
        else:
            regime_status[regime] = "missing"
    supported = [key for key, value in regime_status.items() if value == "supported"]
    present = [key for key, value in regime_status.items() if value != "missing"]
    status = _status_from_flags(
        present=bool(present),
        supported=len(supported) == len(required),
        partial=bool(supported),
    )
    return {
        "status": status,
        "required": required,
        "supported": supported,
        "present": present,
        "missing": [key for key, value in regime_status.items() if value == "missing"],
        "regime_status": regime_status,
        "observed": observed,
    }


def build_unified_matrix(results_root: Path) -> dict[str, Any]:
    artifacts = {
        key: _read_json(results_root / path.relative_to("transit_hrl/results"))
        for key, path in DEFAULT_ARTIFACTS.items()
    }
    paths = {
        key: str(results_root / path.relative_to("transit_hrl/results"))
        for key, path in DEFAULT_ARTIFACTS.items()
    }

    promotion = _promotion_evidence(artifacts, paths)
    promotion_cross_stress = _promotion_cross_stress_evidence(artifacts, paths)
    promotion_reward = promotion["reward"]
    promotion_reward_noninferiority = promotion["reward_noninferiority"]
    promotion_wait = promotion["wait"]
    promotion_wait_noninferiority = promotion["wait_noninferiority"]
    promotion_score = promotion.get("best_score", {})
    real = artifacts[PRIMARY_REAL_DEMAND_ARTIFACT]
    real_score = _real_demand_observed_check(real, "control_score", lower_is_better=False)
    real_reward = _real_demand_observed_check(real, "ep_reward", lower_is_better=False)
    real_wait = _real_demand_observed_check(
        real, "native_avg_board_wait_min", lower_is_better=True
    )
    real_alighted = _real_demand_observed_check(
        real, "native_alighted_pax", lower_is_better=False
    )
    real_throughput = _real_demand_observed_check(
        real, "native_completed_throughput_pax", lower_is_better=False
    )
    real_wait_proxy_noninferiority = _real_demand_observed_check(
        real, "avg_wait_min", lower_is_better=True, noninferiority_margin=0.10
    )
    real_wait_noninferiority = _real_demand_observed_check(
        real,
        "native_avg_board_wait_min",
        lower_is_better=True,
        noninferiority_margin=0.10,
    )
    signal_check = _check_by_metric(real, "service_adjustment_signal")
    real_service_signal = (
        annotate_check(
            signal_check,
            evidence_class=PROJECTION_EVIDENCE,
            headline_eligible=False,
        )
        if signal_check else {}
    )
    real_alighted_noninferiority = _real_demand_observed_check(
        real, "native_alighted_pax", lower_is_better=False, noninferiority_margin=1.0
    )
    real_throughput_noninferiority = _real_demand_observed_check(
        real,
        "native_completed_throughput_pax",
        lower_is_better=False,
        noninferiority_margin=1.0,
    )
    agency_coverage = artifacts["agency_demand_onboard_coverage"] or {}
    agency_boundary_status = {
        str(row.get("evidence_item", "")): str(row.get("status", ""))
        for row in agency_coverage.get("claim_boundaries", []) or []
        if isinstance(row, dict)
    }
    agency_supported_boundaries = sorted(
        key for key, value in agency_boundary_status.items() if value == "supported"
    )
    agency_external_missing = sorted(
        key for key, value in agency_boundary_status.items() if value == "external_missing"
    )
    external_truth_items = {
        "real_public_bus_stop_board_alight",
        "real_public_bus_stop_onboard_load",
        "real_public_subway_od_estimate",
    }
    agency_external_truth_supported = external_truth_items.issubset(set(agency_supported_boundaries))
    agency_scope = str(
        agency_coverage.get("summary", {}).get("evidence_scope", "")
        if isinstance(agency_coverage, dict)
        else ""
    )
    order_book_manifest = artifacts["order_book_manifest"] or {}
    order_book_l2 = artifacts["order_book_l2_matching"] or {}
    order_book_l3 = artifacts["order_book_l3_replay"] or {}
    encoder = artifacts["encoder_matrix"] or artifacts["encoder_matrix_latest"] or {}
    leakage_candidate_keys = [
        "leakage_matrix_latest",
        "leakage_matrix_latest_patch",
        "leakage_matrix_v27_v5",
        "leakage_matrix_v26_v4",
        "leakage_matrix_v25_v3",
        "leakage_matrix",
    ]
    leakage_key = ""
    leakage: dict[str, Any] = {}
    for key in leakage_candidate_keys:
        data = artifacts.get(key)
        if isinstance(data, dict) and data.get("adaptive_native_real_demand_selector"):
            leakage_key = key
            leakage = data
            break
    if not leakage:
        for key in leakage_candidate_keys:
            data = artifacts.get(key)
            if isinstance(data, dict):
                leakage_key = key
                leakage = data
                break
    baseline_ablation = artifacts["baseline_ablation_matrix_latest"] or artifacts["baseline_ablation_matrix"] or {}
    strong_learned = artifacts["strong_learned_baseline_validation"] or {}
    pressure_matrix = artifacts["trading_pressure_matrix"] or {}
    pressure_stress = _pressure_stress_evidence(baseline_ablation, pressure_matrix)
    theory = (
        artifacts["theory_appendix_latest"]
        or artifacts["theory_appendix"]
        or artifacts["theory_appendix_scheduler"]
        or {}
    )

    encoder_domains = encoder.get("domain_summary", []) if isinstance(encoder, dict) else []
    encoder_supported_domains = [
        row.get("domain") for row in encoder_domains
        if int(row.get("supported", 0)) > 0
    ]
    encoder_required_domains = {
        "public_market_daily",
        "public_market_intraday",
        "order_book_l3",
        "transit_real_demand",
    }
    encoder_primary_metrics = {
        "sharpe",
        "total_return",
        "reward",
        "ep_reward",
        "avg_wait_min",
        "native_avg_board_wait_min",
    }
    encoder_primary_supported_domains = {
        str(row.get("domain", ""))
        for row in encoder.get("paired_checks", []) or []
        if str(row.get("status", "")) == "supported"
        and int(row.get("n_common", 0) or 0) >= 5
        and str(row.get("metric", "")) in encoder_primary_metrics
    }
    leakage_verdicts = leakage.get("domain_verdicts", []) if isinstance(leakage, dict) else []
    leakage_supported_domains = [
        row.get("domain") for row in leakage_verdicts
        if row.get("verdict") in {"no_tradeoff_supported", "no_tradeoff_strict_supported"}
    ]
    leakage_strict_domains = [
        row.get("domain") for row in leakage_verdicts
        if row.get("verdict") == "no_tradeoff_strict_supported"
    ]
    leakage_partial_domains = [
        row.get("domain") for row in leakage_verdicts
        if row.get("verdict") in {"partial", "performance_noharm_only", "summary_only_noharm"}
    ]
    leakage_native_selector = (
        leakage.get("adaptive_native_real_demand_selector", {})
        if isinstance(leakage, dict)
        else {}
    )
    leakage_selected_domain = str(leakage_native_selector.get("selected_domain", ""))
    leakage_projection_contaminated = "service_response" in leakage_selected_domain
    leakage_supported_domains = [
        domain for domain in leakage_supported_domains
        if "service_response" not in str(domain)
    ]
    leakage_strict_domains = [
        domain for domain in leakage_strict_domains
        if "service_response" not in str(domain)
    ]
    leakage_native_supported = bool(
        leakage_native_selector.get("supported", False)
        and not leakage_projection_contaminated
    )
    leakage_native_strict = bool(
        leakage_native_selector.get("strict_supported", False)
        and not leakage_projection_contaminated
    )
    theory_examples = theory.get("examples", {}) if isinstance(theory, dict) else {}
    order_book_l2_supported = _count_checks(order_book_l2, {"supported"})
    order_book_l3_positive = _count_checks(order_book_l3, {"supported", "positive_mixed"})
    order_book_has_real_l2_l3 = (
        bool(order_book_manifest.get("coverage", {}).get("real_l2_files", 0))
        and bool(order_book_manifest.get("coverage", {}).get("real_l3_files", 0))
    ) or (_has_non_synthetic_sources(order_book_l2) and _has_non_synthetic_sources(order_book_l3))
    order_book_source_quality = str(
        order_book_manifest.get("coverage", {}).get("source_quality_status", "")
    )
    order_book_venue_session_pairs = int(
        order_book_manifest.get("coverage", {}).get("venue_grade_l2_l3_session_pairs", 0)
        or 0
    )
    venue_sessions = order_book_manifest.get("coverage", {}).get("venue_grade_sessions", []) or []
    order_book_symbols = {
        str(row.get("symbol", "")) for row in venue_sessions if isinstance(row, dict)
    }
    order_book_sessions = {
        str(row.get("session", "")) for row in venue_sessions if isinstance(row, dict)
    }
    order_book_steps = int(order_book_manifest.get("coverage", {}).get("steps", 0) or 0)
    order_book_levels = int(order_book_manifest.get("coverage", {}).get("levels", 0) or 0)
    baseline_summary = baseline_ablation.get("summary", {}) if isinstance(baseline_ablation, dict) else {}
    baseline_checks = baseline_ablation.get("paired_checks", []) if isinstance(baseline_ablation, dict) else []
    positive_baseline_checks = [
        row for row in baseline_checks
        if row.get("metric") == "sharpe" and row.get("status") in {"supported", "positive_mixed"}
    ]
    baseline_support_overrides = baseline_summary.get("ablation_support_overrides", [])
    pressure_rows = pressure_matrix.get("per_seed", []) if isinstance(pressure_matrix, dict) else []
    heuristic_baseline_status = str(baseline_summary.get("claim_status", ""))
    if not heuristic_baseline_status:
        heuristic_baseline_status = "partial" if pressure_rows else "missing"
    strong_summary = strong_learned.get("summary", {}) if isinstance(strong_learned, dict) else {}
    strong_baselines_supported = bool(
        strong_summary.get("ppo_strong_baseline_status") == "supported"
        and strong_summary.get("parameter_budget_status") == "matched"
        and strong_summary.get("trainer_budget_status") == "matched"
        and strong_summary.get("sac_td3_status") == "supported"
    )
    baseline_status = _status_from_flags(
        present=bool(baseline_ablation or strong_learned),
        supported=bool(heuristic_baseline_status == "supported" and strong_baselines_supported),
        partial=bool(baseline_ablation or strong_learned),
    )
    c2_supported = (
        _supported(real_score)
        and _supported(real_reward)
        and _supported(real_wait)
        and _supported(real_alighted)
        and _supported(real_throughput)
    )
    c2_partial = (
        _supported(real_score)
        and _supported(real_reward)
        and _positive(real_wait_proxy_noninferiority)
        and _positive(real_wait_noninferiority)
        and _positive(real_alighted_noninferiority)
        and _positive(real_throughput_noninferiority)
    )
    c2_status = _status_from_flags(
        present=bool(real),
        supported=c2_supported,
        partial=c2_partial,
    )
    if c2_status == "supported":
        if agency_external_truth_supported:
            c2_remaining_gap = (
                "Closed for the current public AFC/APC-profile raw native "
                "validation and public external board/alight/load/estimated-OD "
                "source coverage. Remaining boundary: the MBTA/MTA truth files "
                "are not yet one joint agency OD/onboard-load control loop, and "
                "GTFS-ride-native replication remains optional."
            )
        else:
            c2_remaining_gap = (
                "Closed for the current public AFC/APC-profile raw native "
                "validation. External real OD/onboard-load/alighting ground truth "
                "remains a data boundary unless supplied through GTFS-ride or an "
                "agency APC/OD export."
            )
    else:
        c2_remaining_gap = (
            "The frozen artifact does not support strict improvement from raw "
            "simulator outcomes. projected_* service-response estimates are "
            "sensitivity-only and cannot close this claim."
        )
    c7_remaining_gap = (
        "Closed for the current pre-registered persistent-stress and OD-shift "
        "promotion matrices; remaining work is broader external stress replication."
        if promotion_cross_stress["status"] == "supported"
        else "Scale a pre-registered OD-shift profile until reward and wait improvement CIs are both supported."
    )
    c5_status = _status_from_flags(
        present=bool(leakage_verdicts),
        supported=(
            leakage_native_supported
            and len(leakage_strict_domains) >= 1
            and len(leakage_supported_domains) >= 2
        ),
        partial=bool(leakage_supported_domains or leakage_native_selector),
    )
    c5_remaining_gap = (
        "Closed for the current native real-demand service-response and transit "
        "surrogate leakage matrix; remaining work is independent real-agency and "
        "market-data replication."
        if c5_status == "supported"
        else (
            "Native real-demand C5 uses the adaptive selector from the leakage "
            "matrix. If this remains partial, the selected profile still lacks "
            "joint drift reduction and reward/wait/alighting/throughput no-harm "
            "or strict CI support."
        )
    )
    c3_supported = bool(
        order_book_has_real_l2_l3
        and order_book_source_quality == "venue_grade_ready"
        and order_book_venue_session_pairs >= 20
        and len(order_book_symbols) >= 5
        and len(order_book_sessions) >= 5
        and order_book_steps >= 10000
        and order_book_levels >= 5
    )
    c3_remaining_gap = (
        "Closed for the predeclared large-replay scale gate."
        if c3_supported
        else "Current artifact is a small replay path. The large-replay gate requires at least 20 paired files, 5 symbols, 5 sessions, 10k events per run, and 5 depth levels."
    )

    claims = [
        {
            "id": "C1",
            "claim": "Native learned promotion improves reward and wait",
            "status": promotion["status"],
            "evidence": (
                f"frozen_artifact={promotion['key']} "
                f"selection={promotion.get('selection_policy', 'missing')} "
                f"raw_only={promotion.get('raw_only', False)} "
                f"reward={promotion_reward.get('status', 'missing')} "
                f"reward_noharm={promotion_reward_noninferiority.get('status', 'missing')} "
                f"wait={promotion_wait.get('status', 'missing')} "
                f"wait_noharm={promotion_wait_noninferiority.get('status', 'missing')} "
                f"score={promotion_score.get('status', 'missing')}"
            ),
            "remaining_gap": (
                "Run one frozen v2 native promotion protocol on untouched seeds; "
                "only raw reward and wait outcomes are eligible."
            ),
            "artifact": promotion["artifact"],
        },
        {
            "id": "C2",
            "claim": "Native real AFC/APC-profile demand improves observed score/reward and strict wait/alighting/throughput",
            "status": c2_status,
            "evidence": (
                f"score={real_score.get('status', 'missing')} "
                f"reward={real_reward.get('status', 'missing')} "
                f"wait={real_wait.get('status', 'missing')} "
                f"wait_proxy_noharm={real_wait_proxy_noninferiority.get('status', 'missing')} "
                f"wait_noharm={real_wait_noninferiority.get('status', 'missing')} "
                f"alighted={real_alighted.get('status', 'missing')} "
                f"alighted_noharm={real_alighted_noninferiority.get('status', 'missing')} "
                f"throughput={real_throughput.get('status', 'missing')} "
                f"throughput_noharm={real_throughput_noninferiority.get('status', 'missing')} "
                f"projection_signal={real_service_signal.get('status', 'missing')} "
                f"projection_headline_eligible={real_service_signal.get('headline_eligible', False)} "
                f"agency_scope={agency_scope or 'missing'} "
                f"agency_supported={agency_supported_boundaries} "
                f"agency_external_missing={agency_external_missing}"
            ),
            "remaining_gap": c2_remaining_gap,
            "artifact": (
                paths[PRIMARY_REAL_DEMAND_ARTIFACT]
                + (
                    f" | {paths['agency_demand_onboard_coverage']}"
                    if artifacts["agency_demand_onboard_coverage"] else ""
                )
            ),
        },
        {
            "id": "C3",
            "claim": "Large-scale venue-grade L2/L3 order-book replay is validated",
            "status": _status_from_flags(
                present=bool(order_book_manifest or order_book_l2 or order_book_l3),
                supported=c3_supported,
                partial=bool(order_book_l2 and order_book_l3),
            ),
            "evidence": (
                f"l2_supported_checks={order_book_l2_supported} "
                f"l3_positive_checks={order_book_l3_positive} "
                f"source_quality={order_book_source_quality or 'missing'} "
                f"venue_l2_l3_pairs={order_book_venue_session_pairs} "
                f"symbols={len(order_book_symbols)} sessions={len(order_book_sessions)} "
                f"steps={order_book_steps} levels={order_book_levels} "
                f"manifest_coverage={order_book_manifest.get('coverage', {})}"
            ),
            "remaining_gap": c3_remaining_gap,
            "artifact": (
                paths["order_book_manifest"]
                if order_book_manifest
                else f"{paths['order_book_l2_matching']} | {paths['order_book_l3_replay']}"
                if order_book_l2 or order_book_l3
                else paths["order_book_manifest"]
            ),
        },
        {
            "id": "C4",
            "claim": "Advanced encoder evidence spans Quant and Transit",
            "status": _status_from_flags(
                present=bool(encoder_domains),
                supported=encoder_required_domains.issubset(encoder_primary_supported_domains),
                partial=bool(encoder_supported_domains),
            ),
            "evidence": (
                f"any_supported_domains={encoder_supported_domains} "
                f"primary_supported_domains={sorted(encoder_primary_supported_domains)} "
                f"required={sorted(encoder_required_domains)}"
            ),
            "remaining_gap": "Advanced encoders need primary-outcome paired CIs on public daily/intraday market, real L3, and real-demand Transit; isolated diagnostic wins do not close C4.",
            "artifact": paths["encoder_matrix"] if artifacts["encoder_matrix"] else paths["encoder_matrix_latest"],
        },
        {
            "id": "C5",
            "claim": "Leakage no-tradeoff holds beyond surrogate",
            "status": c5_status,
            "evidence": (
                f"strict_no_tradeoff_domains={leakage_strict_domains} "
                f"no_tradeoff_domains={leakage_supported_domains} "
                f"partial_domains={leakage_partial_domains} "
                f"native_selector_status={leakage_native_selector.get('status', 'missing')} "
                f"native_selector_domain={leakage_native_selector.get('selected_domain', '')} "
                f"native_selector_strict={leakage_native_strict} "
                f"projection_contaminated={leakage_projection_contaminated}"
            ),
            "remaining_gap": c5_remaining_gap,
            "artifact": (
                paths[leakage_key]
                if leakage_key
                else paths["leakage_matrix_latest"]
            ),
        },
        {
            "id": "C6",
            "claim": "Formal theory appendix covers main protocol claims",
            "status": _status_from_flags(
                present=bool(theory),
                supported=(
                    str(theory.get("proof_verification_status", "")) == "verified"
                    and bool(theory.get("theorems"))
                    and "primal_dual_avg_violation_bound_example" in theory_examples
                ),
                partial=bool(theory.get("theorems") or theory_examples),
            ),
            "evidence": (
                f"verification={theory.get('proof_verification_status', 'missing')} "
                f"examples={sorted(theory_examples.keys())}"
            ),
            "remaining_gap": "Structured propositions are present, but C6 remains partial until assumptions and proofs receive an explicit verification audit.",
            "artifact": (
                paths["theory_appendix_latest"]
                if artifacts["theory_appendix_latest"]
                else paths["theory_appendix"]
                if artifacts["theory_appendix"]
                else paths["theory_appendix_scheduler"]
            ),
        },
        {
            "id": "C7",
            "claim": "Native promotion reward/wait improvement replicates across stress regimes",
            "status": promotion_cross_stress["status"],
            "evidence": (
                f"persistent={promotion_cross_stress.get('persistent_key', 'missing')} "
                f"persistent_status={promotion_cross_stress.get('persistent_status', 'missing')} "
                f"odshift={promotion_cross_stress.get('odshift_key', 'missing')} "
                f"odshift_reward={promotion_cross_stress.get('odshift_reward', {}).get('status', 'missing')} "
                f"odshift_reward_noharm={promotion_cross_stress.get('odshift_reward_noninferiority', {}).get('status', 'missing')} "
                f"odshift_wait={promotion_cross_stress.get('odshift_wait', {}).get('status', 'missing')} "
                f"odshift_wait_noharm={promotion_cross_stress.get('odshift_wait_noninferiority', {}).get('status', 'missing')}"
            ),
            "remaining_gap": c7_remaining_gap,
            "artifact": promotion_cross_stress["artifact"],
        },
        {
            "id": "C8",
            "claim": "Strong baseline and ablation table supports frequency-responsibility claim",
            "status": baseline_status,
            "evidence": (
                f"claim_status={baseline_status} "
                f"heuristic_status={heuristic_baseline_status} "
                f"strong_ppo={strong_summary.get('ppo_strong_baseline_status', 'missing')} "
                f"parameter_budget={strong_summary.get('parameter_budget_status', 'missing')} "
                f"trainer_budget={strong_summary.get('trainer_budget_status', 'missing')} "
                f"offpolicy={strong_summary.get('sac_td3_status', 'missing')} "
                f"positive_sharpe_baselines={[row.get('control') for row in positive_baseline_checks]} "
                f"support_overrides={baseline_support_overrides} "
                f"inconclusive={baseline_summary.get('required_baselines_inconclusive', [])} "
                f"not_supported={baseline_summary.get('required_baselines_not_supported', [])} "
                f"missing={baseline_summary.get('required_baselines_missing', [])} "
                f"scenario_win_rate={baseline_summary.get('scenario_freq_family_win_rate', 'NA')} "
                f"pressure_rows={len(pressure_rows)}"
            ),
            "remaining_gap": (
                "Closed only when heuristic ablations and matched PPO/SAC/TD3 learned baselines all pass."
                if baseline_status == "supported"
                else "Implement and run matched v2 flat PPO, generic HRL, SAC, and TD3 baselines; heuristic ablations alone cannot support C8."
            ),
            "artifact": (
                (paths["baseline_ablation_matrix_latest"]
                 if artifacts["baseline_ablation_matrix_latest"]
                 else paths["baseline_ablation_matrix"]
                 if artifacts["baseline_ablation_matrix"]
                 else paths["trading_pressure_matrix"])
                + f" | {paths['strong_learned_baseline_validation']}"
            ),
        },
        {
            "id": "C9",
            "claim": "Synthetic pressure validation covers registered stationary, burst, persistent, and OOD regimes",
            "status": pressure_stress["status"],
            "evidence": (
                f"supported={pressure_stress['supported']} "
                f"present={pressure_stress['present']} "
                f"missing={pressure_stress['missing']} "
                f"regime_status={pressure_stress['regime_status']}"
            ),
            "remaining_gap": "Any missing or not-supported regime must stay outside the global stress-generalization claim.",
            "artifact": (
                paths["baseline_ablation_matrix_latest"]
                if artifacts["baseline_ablation_matrix_latest"]
                else paths["baseline_ablation_matrix"]
                if artifacts["baseline_ablation_matrix"]
                else paths["trading_pressure_matrix"]
            ),
        },
    ]
    supported = sum(1 for row in claims if row["status"] == "supported")
    partial = sum(1 for row in claims if row["status"] == "partial")
    return {
        "claims": claims,
        "summary": {
            "claims": len(claims),
            "supported": supported,
            "partial": partial,
            "not_supported_or_missing": len(claims) - supported - partial,
        },
        "artifacts": paths,
        "boundary": "Unified matrix records current evidence quality; it is not itself a performance validation run.",
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
    _write_csv(output_dir / "claims.csv", payload["claims"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Freq-HRL Unified Top-Journal Evidence Matrix",
        "",
        payload["boundary"],
        "",
        f"- supported: `{payload['summary']['supported']}`",
        f"- partial: `{payload['summary']['partial']}`",
        f"- not supported or missing: `{payload['summary']['not_supported_or_missing']}`",
        "",
        "| id | claim | status | evidence | remaining gap |",
        "|---|---|---|---|---|",
    ]
    for row in payload["claims"]:
        lines.append(
            f"| {row['id']} | {row['claim']} | {row['status']} "
            f"| {row['evidence']} | {row['remaining_gap']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("transit_hrl/results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/top_journal_unified_matrix"),
    )
    args = parser.parse_args()
    payload = build_unified_matrix(Path(args.results_root))
    write_outputs(Path(args.output_dir), payload)
    print(
        "top_journal_unified_matrix "
        f"supported={payload['summary']['supported']} partial={payload['summary']['partial']}"
    )


if __name__ == "__main__":
    main()
