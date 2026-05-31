"""Generate paper-level Freq-HRL claim/evidence diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


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


def build_claim_matrix(results_root: Path, transit_root: Path) -> list[dict[str, str]]:
    plan = read_json(results_root / "trading_ppo_plan_actor_critic" / "summary.json")
    constrained = read_json(results_root / "trading_ppo_primal_dual_leakage" / "summary.json")
    replan = read_json(results_root / "trading_promotion_replan" / "summary.json")
    intraday = read_json(results_root / "trading_public_market_intraday_encoder_ablation" / "summary.json")
    encoder = read_json(results_root / "trading_encoder_ablation_adaptive" / "summary.json")
    transit = read_csv_rows(transit_root / "transit_performance_validation" / "summary.csv")

    plan_summary = plan.get("summary", {})
    constrained_summary = constrained.get("summary", {})
    replan_delta = replan.get("paired_delta", {})
    intraday_rows = intraday.get("summary", [])
    encoder_rows = encoder.get("summary", [])
    transit_freq = next((row for row in transit if row.get("config") == "T_freqhrl_terminal"), {})
    best_intraday = max(intraday_rows, key=lambda row: float(row.get("sharpe", -1e9)), default={})
    adaptive = next((row for row in encoder_rows if row.get("freq_method") == "adaptive_wavelet"), {})
    ema = next((row for row in encoder_rows if row.get("freq_method") == "ema"), {})

    return [
        {
            "claim": "C1: frequency-separated HRL can share one training core",
            "evidence": "Shared dual PPO loop drives Trading and Transit surrogate adapters.",
            "metric": f"trading plan return={_fmt(plan_summary.get('total_return_mean'))}; transit composite={transit_freq.get('composite_mean', 'NA')}",
            "status": "partial",
            "remaining_gap": "Copied Transit native simulator still uses copied RESAC runner.",
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
            "evidence": "Promotion-forced plan replan beats interval-only replan on recovery scenario.",
            "metric": f"return delta={_fmt(replan_delta.get('total_return_delta_mean'))}, recovery regret delta={_fmt(replan_delta.get('recovery_regret_120_delta_mean'))}",
            "status": "supported deterministic",
            "remaining_gap": "Not yet embedded in learned PPO/off-policy runner.",
        },
        {
            "claim": "C4: leakage can be constrained at loss level",
            "evidence": "PPO trajectory constraints with primal-dual multiplier reduce lower-LF drift.",
            "metric": f"constrained LowerLFDrift={_fmt(constrained_summary.get('LowerLFDrift_mean'))}, return={_fmt(constrained_summary.get('total_return_mean'))}",
            "status": "supported with tradeoff",
            "remaining_gap": "Constraint trades off return/Sharpe and did not improve Transit surrogate drift.",
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
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, claims: list[dict[str, str]]) -> None:
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
        "## Paper Boundary",
        "",
        "The current evidence supports a frequency-routed HRL protocol prototype with copied-Transit and trading validation. It does not yet justify a fully validated domain-general algorithm claim because copied Transit native training, larger intraday/order-book data, neural/PINN encoders, and broader statistical tests remain open.",
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "claim_matrix.csv", claims)
    with (args.output_dir / "claim_matrix.json").open("w", encoding="utf-8") as f:
        json.dump({"claims": claims}, f, indent=2)
    write_report(args.output_dir / "report.md", claims)
    print(f"wrote {args.output_dir}")
    print(f"paper_diagnostics claims={len(claims)}")


if __name__ == "__main__":
    main()
