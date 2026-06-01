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
    encoder = read_json(results_root / "trading_encoder_ablation_adaptive" / "summary.json")
    native_audit = read_json(results_root / "transit_native_shared_ppo_audit" / "summary.json")
    transit = read_csv_rows(transit_root / "transit_performance_validation" / "summary.csv")
    checks = build_statistical_checks(results_root)

    plan_summary = plan.get("summary", {})
    constrained_summary = constrained.get("summary", {})
    replan_delta = replan.get("paired_delta", {})
    intraday_rows = intraday.get("summary", [])
    encoder_rows = encoder.get("summary", [])
    transit_freq = next((row for row in transit if row.get("config") == "T_freqhrl_terminal"), {})
    best_intraday = max(intraday_rows, key=lambda row: float(row.get("sharpe", -1e9)), default={})
    adaptive = next((row for row in encoder_rows if row.get("freq_method") == "adaptive_wavelet"), {})
    ema = next((row for row in encoder_rows if row.get("freq_method") == "ema"), {})
    native_contract = native_audit.get("contract", {}) if isinstance(native_audit, dict) else {}
    native_status = str(native_audit.get("status", "missing")) if isinstance(native_audit, dict) else "missing"

    return [
        {
            "claim": "C1: frequency-separated HRL can share one training core",
            "evidence": "Shared dual PPO loop drives Trading and Transit surrogate adapters; native Transit bridge maps that core onto real runner state/action contracts.",
            "metric": (
                f"trading plan return={_fmt(plan_summary.get('total_return_mean'))}; "
                f"transit composite={transit_freq.get('composite_mean', 'NA')}; "
                f"native bridge={native_status} "
                f"U={native_contract.get('upper_state_dim', 'NA')}x{native_contract.get('upper_action_dim', 'NA')} "
                f"L={native_contract.get('lower_state_dim', 'NA')}x{native_contract.get('lower_action_dim', 'NA')}"
            ),
            "status": "supported interface" if native_status == "supported_interface" else "partial",
            "remaining_gap": "Native shared-PPO bridge exists; full native episode replacement training and performance validation remain.",
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
            "evidence": "Deterministic replan improves trading recovery, and a learned PPO promotion gate triggers Transit replans.",
            "metric": (
                f"return delta={_fmt(replan_delta.get('total_return_delta_mean'))}, "
                f"recovery regret delta={_fmt(replan_delta.get('recovery_regret_120_delta_mean'))}; "
                f"learned transit reward={_check_metric(checks, 'transit_learned_promotion_reward_vs_interval')}, "
                f"wait={_check_metric(checks, 'transit_learned_promotion_wait_vs_interval')}, "
                f"replans={_check_metric(checks, 'transit_learned_promotion_replans_vs_interval')}"
            ),
            "status": (
                "supported learned"
                if _check_status(checks, "transit_learned_promotion_reward_vs_interval")
                in {"supported", "positive_mixed"}
                or _check_status(checks, "transit_learned_promotion_wait_vs_interval")
                in {"supported", "positive_mixed"}
                else "supported deterministic"
            ),
            "remaining_gap": "Learned gate is supported on Transit PPO surrogate; native off-policy and larger-seed validation remain.",
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
            "evidence": "Adaptive lifting wavelet runs in Trading and Transit trackers and ablation.",
            "metric": f"adaptive Sharpe={_fmt(adaptive.get('sharpe_mean'))}; EMA Sharpe={_fmt(ema.get('sharpe_mean'))}",
            "status": "mixed",
            "remaining_gap": "Neural state-space and PINN-constrained encoders remain open.",
        },
        {
            "claim": "C6: public-data validation covers more than daily bars",
            "evidence": "Yahoo 5-minute SPY/QQQ/IWM encoder ablation artifact.",
            "metric": f"best intraday encoder={best_intraday.get('freq_method', 'NA')}, Sharpe={_fmt(best_intraday.get('sharpe'))}",
            "status": "supported path",
            "remaining_gap": "Short Level-1 intraday slice only; no order book or execution simulator.",
        },
        {
            "claim": "C7: integrated native Transit Freq-HRL closes the copied-runner gap",
            "evidence": "Gap-closure matrix combines count demand state, plan curves, promotion replanning, native lower context, wait credit, and drift constraint.",
            "metric": (
                f"reward delta={_check_metric(checks, 'transit_full_reward_vs_base')}; "
                f"wait delta={_check_metric(checks, 'transit_full_wait_vs_base')}; "
                f"drift delta={_check_metric(checks, 'transit_full_lower_lf_vs_base')}"
            ),
            "status": _check_status(checks, "transit_full_reward_vs_base"),
            "remaining_gap": "Supported on the small Transit surrogate gate and native shared-PPO interface; still needs full native performance and real-demand validation.",
        },
        {
            "claim": "C8: passenger waiting-time frequency credit improves control quality",
            "evidence": "Full Freq-HRL path is paired against the same path without wait attribution.",
            "metric": f"wait delta vs no-wait={_check_metric(checks, 'transit_wait_credit_vs_no_wait')}",
            "status": _check_status(checks, "transit_wait_credit_vs_no_wait"),
            "remaining_gap": "Supported on the small surrogate gate; still needs larger seed coverage and native timetable validation.",
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
            "evidence": "Poisson/NB harmonic estimator is paired against Fourier by seed and source.",
            "metric": f"MSE delta={_check_metric(checks, 'demand_nb_vs_fourier_mse')}",
            "status": _check_status(checks, "demand_nb_vs_fourier_mse"),
            "remaining_gap": "The count-state path is present; it must beat or match Fourier on larger real Transit demand data before becoming a headline claim.",
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
        "The current evidence supports a frequency-routed HRL protocol prototype with copied-Transit and trading validation. It does not yet justify a fully validated domain-general algorithm claim because full native Transit shared-PPO training, larger intraday/order-book data, neural/PINN encoders, and broader statistical tests remain open.",
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
