"""Generate paper-level Freq-HRL claim/evidence diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.statistics import (
    claim_status,
    format_ci,
    noninferiority_status,
    paired_delta_stats,
)
from freq_hrl.experiments.transit.demand_estimator_validation import COUNT_CALIBRATION_ID


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def collect_gap_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*gap_closure*/summary.json")):
        data = read_json(path)
        payloads = data.get("payloads", {})
        if not isinstance(payloads, dict):
            continue
        for variant, payload in payloads.items():
            per_seed = payload.get("per_seed", []) if isinstance(payload, dict) else []
            for row in per_seed:
                item = dict(row)
                item["source"] = path.parent.name
                item["variant"] = str(variant)
                rows.append(item)
    return rows


def collect_demand_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*demand_estimator*/summary.json")):
        data = read_json(path)
        metadata = data.get("metadata", {})
        if metadata.get("estimator_calibration") != COUNT_CALIBRATION_ID:
            continue
        for row in data.get("rows", []):
            item = dict(row)
            item["source"] = path.parent.name
            rows.append(item)
    return rows


def collect_payload_rows(results_root: Path, dirname: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = read_json(results_root / dirname / "summary.json")
    payloads = data.get("payloads", {})
    if not isinstance(payloads, dict):
        return rows
    for variant, payload in payloads.items():
        per_seed = payload.get("per_seed", []) if isinstance(payload, dict) else []
        for row in per_seed:
            item = dict(row)
            item["source"] = dirname
            item["variant"] = str(variant)
            rows.append(item)
    return rows


def collect_summary_rows(results_root: Path, dirname: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = read_json(results_root / dirname / "summary.json")
    for row in data.get("rows", []):
        item = dict(row)
        item["source"] = dirname
        rows.append(item)
    return rows


def collect_per_seed_rows(
    results_root: Path,
    variants: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, dirname in variants.items():
        for row in read_csv_rows(results_root / dirname / "per_seed.csv"):
            item = dict(row)
            item["variant"] = variant
            item["source"] = dirname
            rows.append(item)
    return rows


def build_statistical_checks(results_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(
        check: str,
        claim: str,
        stats: dict[str, Any],
        *,
        min_pairs: int = 3,
        require_ci: bool = False,
        status: str | None = None,
    ) -> None:
        checks.append({
            "check": check,
            "claim": claim,
            **stats,
            "status": status if status is not None else claim_status(
                stats,
                min_pairs=min_pairs,
                require_ci=require_ci,
            ),
        })

    gap_rows = collect_gap_rows(results_root)
    if gap_rows:
        transit_full_drift_metric = (
            "RawLowerLFDriftAbs"
            if any("RawLowerLFDriftAbs" in row for row in gap_rows)
            else "LowerLFDrift"
        )
        add(
            "transit_full_reward_vs_base",
            "integrated Transit Freq-HRL improves task reward",
            paired_delta_stats(
                gap_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="reward_mean",
                treatment="full_freqhrl",
                control="base_ema_direct",
            ),
            min_pairs=3,
        )
        add(
            "transit_full_wait_vs_base",
            "integrated Transit Freq-HRL lowers passenger wait proxy",
            paired_delta_stats(
                gap_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="wait_proxy",
                treatment="full_freqhrl",
                control="base_ema_direct",
                lower_is_better=True,
            ),
            min_pairs=3,
        )
        add(
            "transit_full_lower_lf_vs_base",
            "integrated Transit Freq-HRL reduces lower low-frequency drift",
            paired_delta_stats(
                gap_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric=transit_full_drift_metric,
                treatment="full_freqhrl",
                control="base_ema_direct",
                lower_is_better=True,
            ),
            min_pairs=3,
        )
        add(
            "transit_wait_credit_vs_no_wait",
            "frequency-attributed wait credit improves wait proxy",
            paired_delta_stats(
                gap_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="wait_proxy",
                treatment="full_freqhrl",
                control="full_no_wait",
                lower_is_better=True,
            ),
            min_pairs=3,
        )

    learned_promotion_rows = collect_payload_rows(results_root, "transit_promotion_learned_replan")
    if learned_promotion_rows:
        add(
            "transit_learned_promotion_reward_vs_interval",
            "learned promotion gate improves Transit reward after persistent shocks",
            paired_delta_stats(
                learned_promotion_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="reward_mean",
                treatment="learned_gate",
                control="interval_only",
            ),
            min_pairs=3,
        )
        add(
            "transit_learned_promotion_wait_vs_interval",
            "learned promotion gate lowers Transit wait after persistent shocks",
            paired_delta_stats(
                learned_promotion_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="wait_proxy",
                treatment="learned_gate",
                control="interval_only",
                lower_is_better=True,
            ),
            min_pairs=3,
        )
        add(
            "transit_learned_promotion_replans_vs_interval",
            "learned promotion gate increases shock-triggered replans",
            paired_delta_stats(
                learned_promotion_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="promotion_replan_count",
                treatment="learned_gate",
                control="interval_only",
            ),
            min_pairs=3,
        )
        learned_drift_metric = (
            "RawLowerLFDriftAbs"
            if any("RawLowerLFDriftAbs" in row for row in learned_promotion_rows)
            else "LowerLFDrift"
        )
        add(
            "transit_learned_promotion_raw_lf_vs_interval",
            "learned promotion gate does not increase lower LF drift",
            paired_delta_stats(
                learned_promotion_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric=learned_drift_metric,
                treatment="learned_gate",
                control="interval_only",
                lower_is_better=True,
            ),
            min_pairs=3,
        )

    native_promotion_rows = collect_summary_rows(results_root, "transit_native_promotion_replan")
    if native_promotion_rows:
        add(
            "transit_native_promotion_reward_vs_interval",
            "native promotion replanning improves Transit episode reward",
            paired_delta_stats(
                native_promotion_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="ep_reward",
                treatment="native_promotion_replan",
                control="interval_only",
            ),
            min_pairs=5,
        )
        add(
            "transit_native_promotion_wait_vs_interval",
            "native promotion replanning lowers Transit wait",
            paired_delta_stats(
                native_promotion_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="avg_wait_min",
                treatment="native_promotion_replan",
                control="interval_only",
                lower_is_better=True,
            ),
            min_pairs=5,
        )
        add(
            "transit_native_promotion_replans_vs_interval",
            "native promotion replanning increases upper timetable replans",
            paired_delta_stats(
                native_promotion_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="upper_plan_decisions",
                treatment="native_promotion_replan",
                control="interval_only",
            ),
            min_pairs=5,
        )
        if any(row.get("variant") == "native_learned_gate" for row in native_promotion_rows):
            add(
                "transit_native_learned_gate_reward_vs_interval",
                "native learned promotion gate improves Transit episode reward",
                paired_delta_stats(
                    native_promotion_rows,
                    variant_key="variant",
                    pair_keys=("source", "seed"),
                    metric="ep_reward",
                    treatment="native_learned_gate",
                    control="interval_only",
                ),
                min_pairs=5,
            )
            add(
                "transit_native_learned_gate_wait_vs_interval",
                "native learned promotion gate lowers Transit wait",
                paired_delta_stats(
                    native_promotion_rows,
                    variant_key="variant",
                    pair_keys=("source", "seed"),
                    metric="avg_wait_min",
                    treatment="native_learned_gate",
                    control="interval_only",
                    lower_is_better=True,
                ),
                min_pairs=5,
            )
            add(
                "transit_native_learned_gate_score_vs_interval",
                "native learned promotion gate improves Transit wait/headway control score",
                paired_delta_stats(
                    native_promotion_rows,
                    variant_key="variant",
                    pair_keys=("source", "seed"),
                    metric="score",
                    treatment="native_learned_gate",
                    control="interval_only",
                ),
                min_pairs=5,
            )
            add(
                "transit_native_learned_gate_replans_vs_interval",
                "native learned promotion gate increases upper timetable replans",
                paired_delta_stats(
                    native_promotion_rows,
                    variant_key="variant",
                    pair_keys=("source", "seed"),
                    metric="upper_plan_decisions",
                    treatment="native_learned_gate",
                    control="interval_only",
                ),
                min_pairs=5,
            )
            add(
                "transit_native_learned_gate_gate_replans_vs_interval",
                "native learned promotion gate fires gate-triggered replans",
                paired_delta_stats(
                    native_promotion_rows,
                    variant_key="variant",
                    pair_keys=("source", "seed"),
                    metric="shared_ppo_gate_replans",
                    treatment="native_learned_gate",
                    control="interval_only",
                ),
                min_pairs=5,
            )

    native_wait_rows = collect_summary_rows(results_root, "transit_native_wait_credit")
    if native_wait_rows:
        add(
            "transit_native_wait_credit_final_wait_vs_no_wait",
            "native frequency-attributed wait credit lowers final Transit wait",
            paired_delta_stats(
                native_wait_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="final_avg_wait_min",
                treatment="native_wait_credit",
                control="no_wait_credit",
                lower_is_better=True,
            ),
            min_pairs=5,
        )
        add(
            "transit_native_wait_credit_mean_wait_vs_no_wait",
            "native frequency-attributed wait credit lowers mean Transit wait",
            paired_delta_stats(
                native_wait_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="avg_wait_min_mean",
                treatment="native_wait_credit",
                control="no_wait_credit",
                lower_is_better=True,
            ),
            min_pairs=5,
        )
        add(
            "transit_native_wait_credit_reward_vs_no_wait",
            "native frequency-attributed wait credit improves episode reward",
            paired_delta_stats(
                native_wait_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="final_ep_reward",
                treatment="native_wait_credit",
                control="no_wait_credit",
            ),
            min_pairs=5,
        )
        add(
            "transit_native_wait_credit_score_vs_no_wait",
            "native frequency-attributed wait credit improves wait/headway score",
            paired_delta_stats(
                native_wait_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="final_score",
                treatment="native_wait_credit",
                control="no_wait_credit",
            ),
            min_pairs=5,
        )
        add(
            "transit_native_wait_credit_active_vs_no_wait",
            "native frequency-attributed wait credit is active in PPO rewards",
            paired_delta_stats(
                native_wait_rows,
                variant_key="variant",
                pair_keys=("source", "seed"),
                metric="freq_wait_upper_credit_std",
                treatment="native_wait_credit",
                control="no_wait_credit",
            ),
            min_pairs=5,
        )

    demand_rows = collect_demand_rows(results_root)
    demand_methods = {str(row.get("method")) for row in demand_rows}
    if {"dynamic_harmonic_nb", "fourier"} <= demand_methods:
        for metric in ("mse", "mae", "poisson_nll_no_const"):
            add(
                f"demand_nb_vs_fourier_{metric}",
                "dynamic harmonic count estimator is competitive with Fourier",
                paired_delta_stats(
                    demand_rows,
                    variant_key="method",
                    pair_keys=("source", "seed"),
                    metric=metric,
                    treatment="dynamic_harmonic_nb",
                    control="fourier",
                    lower_is_better=True,
                ),
                min_pairs=5,
                require_ci=(metric == "mse"),
            )

    trading_rows = collect_per_seed_rows(
        results_root,
        {
            "trading_plan": "trading_ppo_plan_actor_critic",
            "trading_constrained": "trading_ppo_primal_dual_leakage",
        },
    )
    if trading_rows:
        add(
            "trading_constraint_lower_lf",
            "primal-dual constraint lowers trading lower-LF drift",
            paired_delta_stats(
                trading_rows,
                variant_key="variant",
                pair_keys=("seed",),
                metric="LowerLFDrift",
                treatment="trading_constrained",
                control="trading_plan",
                lower_is_better=True,
            ),
            min_pairs=5,
        )
        trading_return = paired_delta_stats(
            trading_rows,
            variant_key="variant",
            pair_keys=("seed",),
            metric="total_return",
            treatment="trading_constrained",
            control="trading_plan",
        )
        add(
            "trading_constraint_return_tradeoff",
            "trading leakage constraint has no return tradeoff",
            trading_return,
            min_pairs=5,
            status=noninferiority_status(trading_return, max_loss=0.01, min_pairs=5),
        )
        trading_raw_metric = (
            "RawLowerLFDriftAbs"
            if any("RawLowerLFDriftAbs" in row for row in trading_rows)
            else "RawLowerLFDrift"
        )
        if any(trading_raw_metric in row for row in trading_rows):
            add(
                "trading_constraint_raw_lower_lf",
                "primal-dual constraint lowers raw trading lower-LF drift",
                paired_delta_stats(
                    trading_rows,
                    variant_key="variant",
                    pair_keys=("seed",),
                    metric=trading_raw_metric,
                    treatment="trading_constrained",
                    control="trading_plan",
                    lower_is_better=True,
                ),
                min_pairs=5,
            )

    transit_rows = collect_per_seed_rows(
        results_root,
        {
            "transit_plan": "transit_ppo_plan_surrogate",
            "transit_constrained": "transit_ppo_primal_dual_leakage",
        },
    )
    if transit_rows:
        add(
            "transit_constraint_lower_lf",
            "primal-dual constraint lowers Transit lower-LF drift",
            paired_delta_stats(
                transit_rows,
                variant_key="variant",
                pair_keys=("seed",),
                metric="LowerLFDrift",
                treatment="transit_constrained",
                control="transit_plan",
                lower_is_better=True,
            ),
            min_pairs=5,
        )
        transit_reward = paired_delta_stats(
            transit_rows,
            variant_key="variant",
            pair_keys=("seed",),
            metric="reward_mean",
            treatment="transit_constrained",
            control="transit_plan",
        )
        add(
            "transit_constraint_reward_tradeoff",
            "Transit leakage constraint has no reward tradeoff",
            transit_reward,
            min_pairs=5,
            status=noninferiority_status(transit_reward, max_loss=0.005, min_pairs=5),
        )
        transit_raw_metric = (
            "RawLowerLFDriftAbs"
            if any("RawLowerLFDriftAbs" in row for row in transit_rows)
            else "RawLowerLFDrift"
        )
        if any(transit_raw_metric in row for row in transit_rows):
            add(
                "transit_constraint_raw_lower_lf",
                "primal-dual constraint lowers raw Transit lower-LF drift",
                paired_delta_stats(
                    transit_rows,
                    variant_key="variant",
                    pair_keys=("seed",),
                    metric=transit_raw_metric,
                    treatment="transit_constrained",
                    control="transit_plan",
                    lower_is_better=True,
                ),
                min_pairs=5,
            )

    return checks


def _check_status(checks: list[dict[str, Any]], name: str) -> str:
    row = next((item for item in checks if item.get("check") == name), None)
    return str(row.get("status", "missing")) if row else "missing"


def _check_metric(checks: list[dict[str, Any]], name: str, digits: int = 4) -> str:
    row = next((item for item in checks if item.get("check") == name), None)
    return format_ci(row, digits=digits) if row else "NA"


def _all_supported(checks: list[dict[str, Any]], names: list[str]) -> bool:
    return all(_check_status(checks, name) in {"supported", "positive_mixed"} for name in names)


def build_claim_matrix(results_root: Path, transit_root: Path) -> list[dict[str, str]]:
    plan = read_json(results_root / "trading_ppo_plan_actor_critic" / "summary.json")
    constrained = read_json(results_root / "trading_ppo_primal_dual_leakage" / "summary.json")
    replan = read_json(results_root / "trading_promotion_replan" / "summary.json")
    intraday = read_json(results_root / "trading_public_market_intraday_encoder_ablation" / "summary.json")
    order_book = read_json(results_root / "trading_order_book_encoder_ablation" / "summary.json")
    encoder = read_json(results_root / "trading_encoder_ablation_adaptive" / "summary.json")
    neural_encoder = read_json(results_root / "trading_encoder_ablation_neural" / "summary.json")
    native_audit = read_json(results_root / "transit_native_shared_ppo_audit" / "summary.json")
    native_loop = read_json(results_root / "transit_native_shared_ppo_loop" / "summary.json")
    native_offpolicy = read_json(results_root / "transit_native_offpolicy_smoke" / "summary.json")
    transit = read_csv_rows(transit_root / "transit_performance_validation" / "summary.csv")
    checks = build_statistical_checks(results_root)

    plan_summary = plan.get("summary", {})
    constrained_summary = constrained.get("summary", {})
    replan_delta = replan.get("paired_delta", {})
    intraday_rows = intraday.get("summary", [])
    order_book_rows = order_book.get("summary", [])
    encoder_rows = encoder.get("summary", [])
    neural_encoder_rows = neural_encoder.get("summary", [])
    transit_freq = next((row for row in transit if row.get("config") == "T_freqhrl_terminal"), {})
    best_intraday = max(intraday_rows, key=lambda row: float(row.get("sharpe", -1e9)), default={})
    best_order_book = max(order_book_rows, key=lambda row: float(row.get("sharpe", -1e9)), default={})
    adaptive = next((row for row in encoder_rows if row.get("freq_method") == "adaptive_wavelet"), {})
    neural = next((row for row in neural_encoder_rows if row.get("freq_method") == "neural_state_space"), {})
    ema = next((row for row in encoder_rows if row.get("freq_method") == "ema"), {})
    native_contract = native_audit.get("contract", {}) if isinstance(native_audit, dict) else {}
    native_status = str(native_audit.get("status", "missing")) if isinstance(native_audit, dict) else "missing"
    native_loop_status = str(native_loop.get("status", "missing")) if isinstance(native_loop, dict) else "missing"
    native_loop_summary = native_loop.get("summary", {}) if isinstance(native_loop, dict) else {}
    native_loop_contract = native_loop.get("contract", {}) if isinstance(native_loop, dict) else {}
    native_offpolicy_status = str(native_offpolicy.get("status", "missing")) if isinstance(native_offpolicy, dict) else "missing"
    native_offpolicy_updates = native_offpolicy.get("offpolicy_replay_updates", "NA") if isinstance(native_offpolicy, dict) else "NA"
    learned_promotion_supported = (
        _check_status(checks, "transit_learned_promotion_reward_vs_interval") == "supported"
        and _check_status(checks, "transit_learned_promotion_wait_vs_interval") == "supported"
        and _check_status(checks, "transit_learned_promotion_replans_vs_interval") == "supported"
    )
    native_promotion_reward_status = _check_status(checks, "transit_native_promotion_reward_vs_interval")
    native_promotion_replan_status = _check_status(checks, "transit_native_promotion_replans_vs_interval")
    native_learned_reward_status = _check_status(checks, "transit_native_learned_gate_reward_vs_interval")
    native_learned_score_status = _check_status(checks, "transit_native_learned_gate_score_vs_interval")
    native_learned_replan_status = _check_status(checks, "transit_native_learned_gate_gate_replans_vs_interval")
    native_wait_credit_status = _check_status(checks, "transit_native_wait_credit_final_wait_vs_no_wait")
    if learned_promotion_supported and native_promotion_reward_status == "supported":
        promotion_status = "supported learned+native reward"
    elif (
        learned_promotion_supported
        and native_learned_score_status == "supported"
        and native_learned_replan_status == "supported"
    ):
        promotion_status = "supported learned; native learned-gate score"
    elif (
        learned_promotion_supported
        and native_learned_reward_status in {"supported", "positive_mixed"}
        and native_learned_replan_status == "supported"
    ):
        promotion_status = "supported learned; native learned-gate path"
    elif learned_promotion_supported and native_promotion_replan_status == "supported":
        promotion_status = "supported learned; native replan"
    elif learned_promotion_supported:
        promotion_status = "supported learned"
    else:
        promotion_status = "supported deterministic"

    return [
        {
            "claim": "C1: frequency-separated HRL can share one training core",
            "evidence": "Shared dual PPO loop drives Trading and Transit surrogate adapters, and a native Transit episode loop now uses the same PPO core for upper/lower actions.",
            "metric": (
                f"trading plan return={_fmt(plan_summary.get('total_return_mean'))}; "
                f"transit composite={transit_freq.get('composite_mean', 'NA')}; "
                f"native bridge={native_status} "
                f"U={native_contract.get('upper_state_dim', native_loop_contract.get('upper_state_dim', 'NA'))}x{native_contract.get('upper_action_dim', native_loop_contract.get('upper_action_dim', 'NA'))} "
                f"L={native_contract.get('lower_state_dim', native_loop_contract.get('lower_state_dim', 'NA'))}x{native_contract.get('lower_action_dim', native_loop_contract.get('lower_action_dim', 'NA'))}; "
                f"native loop={native_loop_status}, wait={_fmt(native_loop_summary.get('avg_wait_min_mean'))}; "
                f"offpolicy native={native_offpolicy_status}, replay_updates={native_offpolicy_updates}"
            ),
            "status": (
                "supported native loop"
                if native_loop_status == "supported_native_episode_loop"
                else ("supported interface" if native_status == "supported_interface" else "partial")
            ),
            "remaining_gap": "Native shared-PPO episode loop exists; multi-seed native performance validation remains.",
        },
        {
            "claim": "C2: high-level plan variables can be learned as curves",
            "evidence": "Upper PPO action can parameterize Bernstein coefficients.",
            "metric": f"plan-PPO return={_fmt(plan_summary.get('total_return_mean'))}, LowerLFDrift={_fmt(plan_summary.get('LowerLFDrift_mean'))}",
            "status": "supported synthetic",
            "remaining_gap": "Public-data and copied-Transit learned plan-coefficient training remain open.",
        },
        {
            "claim": "C3: promotion should trigger replanning after persistent shocks",
            "evidence": "Deterministic replan improves trading recovery, a learned PPO promotion gate improves Transit surrogate reward/wait, and native Transit promotion-replan increases timetable replans.",
            "metric": (
                f"return delta={_fmt(replan_delta.get('total_return_delta_mean'))}, "
                f"recovery regret delta={_fmt(replan_delta.get('recovery_regret_120_delta_mean'))}; "
                f"learned transit reward={_check_metric(checks, 'transit_learned_promotion_reward_vs_interval')}, "
                f"wait={_check_metric(checks, 'transit_learned_promotion_wait_vs_interval')}, "
                f"replans={_check_metric(checks, 'transit_learned_promotion_replans_vs_interval')}; "
                f"native reward={_check_metric(checks, 'transit_native_promotion_reward_vs_interval')}, "
                f"native replans={_check_metric(checks, 'transit_native_promotion_replans_vs_interval')}; "
                f"native learned reward={_check_metric(checks, 'transit_native_learned_gate_reward_vs_interval')}, "
                f"native learned score={_check_metric(checks, 'transit_native_learned_gate_score_vs_interval')}, "
                f"native learned gate replans={_check_metric(checks, 'transit_native_learned_gate_gate_replans_vs_interval')}"
            ),
            "status": promotion_status,
            "remaining_gap": "Native learned gate runs end-to-end with CI-supported control score/gate replans, but episode reward remains positive-mixed; larger off-policy/native training remains.",
        },
        {
            "claim": "C4: leakage can be constrained at loss level",
            "evidence": "PPO trajectory constraints with primal-dual multiplier reduce lower-LF drift.",
            "metric": (
                f"trading drift delta={_check_metric(checks, 'trading_constraint_lower_lf')}; "
                f"return delta={_check_metric(checks, 'trading_constraint_return_tradeoff')}"
            ),
            "status": (
                "supported"
                if _all_supported(checks, [
                    "trading_constraint_lower_lf",
                    "trading_constraint_return_tradeoff",
                ])
                else "not_supported"
            ),
            "remaining_gap": "Projected and raw lower-drift constraints are supported in surrogate diagnostics; native and real-data confirmation remain.",
        },
        {
            "claim": "C5: advanced causal encoders can be swapped by domain",
            "evidence": "Adaptive lifting wavelet and neural/PINN state-space encoders run in Trading and Transit trackers and ablations.",
            "metric": (
                f"adaptive Sharpe={_fmt(adaptive.get('sharpe_mean'))}; "
                f"neural Sharpe={_fmt(neural.get('sharpe_mean'))}; "
                f"EMA Sharpe={_fmt(ema.get('sharpe_mean'))}"
            ),
            "status": "supported path" if neural else "mixed",
            "remaining_gap": "Neural/PINN encoder path exists; larger cross-domain performance validation is still needed.",
        },
        {
            "claim": "C6: public-data validation covers more than daily bars",
            "evidence": "Yahoo 5-minute SPY/QQQ/IWM encoder ablation and order-book microstructure CSV adapter artifacts.",
            "metric": (
                f"best intraday encoder={best_intraday.get('freq_method', 'NA')}, "
                f"Sharpe={_fmt(best_intraday.get('sharpe'))}; "
                f"best order-book encoder={best_order_book.get('freq_method', 'NA')}, "
                f"Sharpe={_fmt(best_order_book.get('sharpe'))}"
            ),
            "status": "supported path",
            "remaining_gap": "Order-book adapter exists with deterministic CI fixture; larger real L2/L3 feeds remain for the strongest data claim.",
        },
        {
            "claim": "C7: integrated native Transit Freq-HRL closes the copied-runner gap",
            "evidence": "Gap-closure matrix combines count demand state, plan curves, promotion replanning, native lower context, wait credit, and drift constraint; native loop runs the same shared PPO core inside the copied Transit simulator.",
            "metric": (
                f"reward delta={_check_metric(checks, 'transit_full_reward_vs_base')}; "
                f"wait delta={_check_metric(checks, 'transit_full_wait_vs_base')}; "
                f"drift delta={_check_metric(checks, 'transit_full_lower_lf_vs_base')}; "
                f"native-loop samples={_fmt((native_loop.get('rows') or [{}])[0].get('shared_ppo_lower_samples') if isinstance(native_loop, dict) else None, digits=0)}"
            ),
            "status": _check_status(checks, "transit_full_reward_vs_base"),
            "remaining_gap": "Supported on surrogate performance plus a native shared-PPO episode loop; still needs multi-seed native performance and real-demand validation.",
        },
        {
            "claim": "C8: passenger waiting-time frequency credit improves control quality",
            "evidence": "Full Freq-HRL path is paired against the same path without wait attribution in both the surrogate gate and the native shared-PPO Transit loop.",
            "metric": (
                f"surrogate wait delta={_check_metric(checks, 'transit_wait_credit_vs_no_wait')}; "
                f"native final wait delta={_check_metric(checks, 'transit_native_wait_credit_final_wait_vs_no_wait')}; "
                f"native score delta={_check_metric(checks, 'transit_native_wait_credit_score_vs_no_wait')}; "
                f"native reward delta={_check_metric(checks, 'transit_native_wait_credit_reward_vs_no_wait')}"
            ),
            "status": (
                "supported native"
                if native_wait_credit_status == "supported"
                else (
                    "supported surrogate; native positive-mixed"
                    if (
                        _check_status(checks, "transit_wait_credit_vs_no_wait") == "supported"
                        and native_wait_credit_status == "positive_mixed"
                    )
                    else _check_status(checks, "transit_wait_credit_vs_no_wait")
                )
            ),
            "remaining_gap": (
                "Native wait-credit path is positive-mixed in the shared-PPO loop; still needs more seeds/episodes and real AFC/APC demand."
                if native_wait_credit_status == "positive_mixed"
                else (
                    "Native wait-credit path is supported in the shared-PPO loop; still needs real AFC/APC demand."
                    if native_wait_credit_status == "supported"
                    else "Native wait-credit validation harness exists, but native timetable performance is not yet supported."
                )
            ),
        },
        {
            "claim": "C9: leakage constraints achieve no-tradeoff responsibility separation",
            "evidence": "Trading and Transit primal-dual constraints are checked on paired seeds for drift and task-return deltas.",
            "metric": (
                f"trading drift={_check_status(checks, 'trading_constraint_lower_lf')}, "
                f"trading return={_check_status(checks, 'trading_constraint_return_tradeoff')}, "
                f"transit drift={_check_status(checks, 'transit_constraint_lower_lf')}, "
                f"transit reward={_check_status(checks, 'transit_constraint_reward_tradeoff')}; "
                f"raw drift trading={_check_status(checks, 'trading_constraint_raw_lower_lf')}, "
                f"raw drift transit={_check_status(checks, 'transit_constraint_raw_lower_lf')}"
            ),
            "status": (
                "supported"
                if _all_supported(checks, [
                    "trading_constraint_lower_lf",
                    "trading_constraint_return_tradeoff",
                    "transit_constraint_lower_lf",
                    "transit_constraint_reward_tradeoff",
                ])
                else "not_supported"
            ),
            "remaining_gap": "Supported on surrogate Trading/Transit with raw-drift diagnostics; still needs native Transit and real-data confirmation.",
        },
        {
            "claim": "C10: dynamic harmonic count-state demand estimator is competitive",
            "evidence": "Poisson/NB harmonic estimator is paired against Fourier by seed/source, including synthetic counts, copied TransitDuet local OD traces, and public GTFS schedule-event traces.",
            "metric": f"MSE delta={_check_metric(checks, 'demand_nb_vs_fourier_mse')}",
            "status": _check_status(checks, "demand_nb_vs_fourier_mse"),
            "remaining_gap": "The count-state path now covers public GTFS schedule proxies; true AFC/APC passenger-demand feeds remain for the strongest claim.",
        },
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_statistical_checks(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "check",
        "claim",
        "status",
        "metric",
        "treatment",
        "control",
        "direction",
        "n_common",
        "delta_mean",
        "delta_ci95_low",
        "delta_ci95_high",
        "improvement_mean",
        "improvement_ci95_low",
        "improvement_ci95_high",
        "win_rate",
        "sign_p_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    claims: list[dict[str, str]],
    checks: list[dict[str, Any]],
) -> None:
    lines = [
        "# Freq-HRL Paper Diagnostics",
        "",
        "## Formal Objects",
        "",
        "- Exogenous stream: causal bins `x_t` emitted by a domain adapter.",
        "- Encoder: `z_t = (x_low, x_mid, x_high, energy, persistence)` with no access to future bins.",
        "- Upper policy: low-frequency plan action `a_U`, optionally Bernstein coefficients over a horizon.",
        "- Lower policy: high-frequency execution/control action `a_L` conditioned on the active upper plan.",
        "- Promotion gate: persistent high-frequency residual detector that can promote regime evidence into the upper plan.",
        "- Leakage: action-effect mismatch `UpperHFPower + LowerLFDrift`, computed causally from upper and lower effects.",
        "",
        "## Diagnostic Bounds",
        "",
        "For shaped rewards `r'_t = r_t - lambda * L_t`, cumulative shaped-return deviation from task return is bounded by `lambda * sum_t L_t`. With `L_t >= 0`, optimizing shaped return is a conservative lower bound on task return when leakage is treated as a constraint cost. The primal-dual PPO path makes this explicit by adding `eta * (cost_t - c)` to the clipped policy objective and updating `eta` from observed cost excess.",
        "",
        "Promotion false positives and false negatives are controlled by the persistence window, residual threshold, regime buffer, and strength threshold. Lower thresholds reduce detection delay but raise stationary/high-noise false positives; the pressure matrix and promotion-replan validation should be reported together.",
        "",
        "## Claim Matrix",
        "",
        "| claim | status | metric | remaining gap |",
        "|---|---|---|---|",
    ]
    for row in claims:
        lines.append(
            f"| {row['claim']} | {row['status']} | {row['metric']} | {row['remaining_gap']} |"
        )
    lines.extend([
        "",
        "## Statistical Claim Gates",
        "",
        "Deltas are `treatment - control`; `direction=decrease` means negative raw delta is the desired effect. Bootstrap intervals are paired by seed where possible.",
        "No-tradeoff gates use a small noninferiority margin: 0.01 total-return for trading and 0.005 reward-mean for Transit.",
        "",
        "| check | status | metric | n | delta CI95 | win rate | sign p |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in checks:
        lines.append(
            f"| {row['check']} "
            f"| {row['status']} "
            f"| {row['metric']} "
            f"| {row['n_common']} "
            f"| {format_ci(row)} "
            f"| {_fmt(row.get('win_rate'), 2)} "
            f"| {_fmt(row.get('sign_p_value'), 4)} |"
        )
    lines.extend([
        "",
        "## Paper Boundary",
        "",
        "The current evidence supports a frequency-routed HRL protocol prototype with trading, surrogate Transit, native Transit shared-PPO validation, and public GTFS schedule-proxy data paths. It does not yet justify a fully validated domain-general algorithm claim because native learned-promotion reward, larger real intraday/order-book feeds, true AFC/APC passenger-demand feeds, and broader seed-level statistical tests remain open.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("transit_hrl/results"))
    parser.add_argument(
        "--transit-root",
        type=Path,
        default=Path("transit_hrl/freq_transitduet/results_freqhrl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/freq_hrl_paper_diagnostics"))
    args = parser.parse_args()
    claims = build_claim_matrix(args.results_root, args.transit_root)
    checks = build_statistical_checks(args.results_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "claim_matrix.csv", claims)
    write_statistical_checks(args.output_dir / "statistical_checks.csv", checks)
    with (args.output_dir / "claim_matrix.json").open("w", encoding="utf-8") as f:
        json.dump({"claims": claims, "statistical_checks": checks}, f, indent=2)
    write_report(args.output_dir / "report.md", claims, checks)
    print(f"wrote {args.output_dir}")
    print(f"paper_diagnostics claims={len(claims)} checks={len(checks)}")


if __name__ == "__main__":
    main()
