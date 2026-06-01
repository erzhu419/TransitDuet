"""Generate a compact theory appendix for Freq-HRL diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def shaped_return_deviation_bound(leakage_weight: float, leakage_costs: list[float]) -> float:
    """Bound |sum r - sum r'| for r' = r - lambda * leakage."""
    return float(max(leakage_weight, 0.0) * sum(max(float(cost), 0.0) for cost in leakage_costs))


def promotion_false_positive_bound(
    *,
    window_bins: int,
    persistence_ratio: float,
    event_probability: float,
) -> float:
    """Hoeffding upper bound for stationary false promotion events.

    The promotion gate fires when the trailing residual-event share exceeds
    `persistence_ratio`. If stationary noise exceeds the threshold with
    probability p < rho, P(mean >= rho) <= exp(-2 n (rho - p)^2).
    """
    n = max(int(window_bins), 1)
    rho = float(min(max(persistence_ratio, 0.0), 1.0))
    p = float(min(max(event_probability, 0.0), 1.0))
    if p >= rho:
        return 1.0
    return float(math.exp(-2.0 * n * (rho - p) ** 2))


def promotion_detection_delay_bound(
    *,
    update_interval_s: float,
    window_bins: int,
    persistence_ratio: float,
) -> float:
    """Worst-case delay when every new residual event is above threshold."""
    n = max(int(window_bins), 1)
    rho = float(min(max(persistence_ratio, 0.0), 1.0))
    required = max(1, int(math.ceil(rho * n)))
    return float(max(update_interval_s, 0.0) * max(required, n))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _check(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("check") == name), {})


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def build_theory_payload(results_root: Path) -> dict[str, Any]:
    checks = read_csv_rows(results_root / "freq_hrl_paper_diagnostics" / "statistical_checks.csv")
    examples = {
        "leakage_bound_example": shaped_return_deviation_bound(
            leakage_weight=0.30,
            leakage_costs=[0.12, 0.08, 0.05, 0.04],
        ),
        "promotion_false_positive_bound_example": promotion_false_positive_bound(
            window_bins=10,
            persistence_ratio=0.35,
            event_probability=0.10,
        ),
        "promotion_detection_delay_bound_s": promotion_detection_delay_bound(
            update_interval_s=60.0,
            window_bins=10,
            persistence_ratio=0.35,
        ),
    }
    cited_checks = {
        "transit_learned_promotion_wait": _check(checks, "transit_learned_promotion_wait_vs_interval"),
        "native_learned_gate_reward": _check(checks, "transit_native_learned_gate_reward_vs_interval"),
        "trading_leakage_constraint": _check(checks, "trading_constraint_lower_lf"),
        "transit_leakage_constraint": _check(checks, "transit_constraint_lower_lf"),
    }
    return {
        "formal_objects": [
            "causal exogenous stream x_t",
            "endogenous environment state z_t",
            "causal spectral encoder E_phi(x_<=t)",
            "upper low-frequency plan policy pi_U",
            "lower high-frequency controller pi_L",
            "promotion gate g_promote",
            "action-effect leakage cost L_t",
        ],
        "assumptions": [
            "A1: the encoder reads only current and past exogenous bins.",
            "A2: the upper action remains active across multiple lower decisions unless a scheduled or promoted replan occurs.",
            "A3: leakage costs are nonnegative and computed causally from action effects.",
            "A4: under stationary noise, residual-threshold events are conditionally bounded by a Bernoulli rate p.",
        ],
        "examples": examples,
        "cited_checks": cited_checks,
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Freq-HRL Theory Appendix",
        "",
        "## Formal Setup",
        "",
        "Freq-HRL assumes an endogenous state `z_t`, an exogenous time-series stream `x_t`, and a causal encoder `E_phi(x_<=t)` that emits low-frequency trend, middle-frequency regime buffer, high-frequency residual, uncertainty, energy, and persistence summaries.",
        "",
        "The upper policy `pi_U` consumes low-frequency trend/forecast plus bounded high-frequency summaries and emits a plan action. The lower policy `pi_L` consumes the active upper plan, local endogenous state, and high/middle-frequency residual context and emits high-frequency control actions.",
        "",
        "## Assumptions",
        "",
    ]
    for item in payload["assumptions"]:
        lines.append(f"- {item}")
    examples = payload["examples"]
    lines.extend([
        "",
        "## Theorem 1: Leakage-Shaped Return Bound",
        "",
        "For shaped rewards `r'_t = r_t - lambda L_t`, where `L_t >= 0`, the absolute deviation between task return and shaped return over an episode is bounded by `lambda * sum_t L_t`. Therefore, enforcing a leakage budget controls the maximum reward-shaping distortion while penalizing responsibility violations.",
        "",
        f"Example bound with `lambda=0.30`: `{_fmt(examples['leakage_bound_example'])}`.",
        "",
        "## Theorem 2: Stationary Promotion False-Positive Bound",
        "",
        "If residual threshold events occur with stationary probability `p < rho`, and promotion requires a trailing-window event share of at least `rho`, Hoeffding's inequality gives `P(false promote) <= exp(-2 n (rho-p)^2)` for window length `n`.",
        "",
        f"Example `n=10`, `rho=0.35`, `p=0.10`: `{_fmt(examples['promotion_false_positive_bound_example'], digits=6)}`.",
        "",
        "## Theorem 3: Persistent-Shock Detection Delay",
        "",
        "If every residual event after a regime shift exceeds threshold, the causal trailing-window gate detects the shift after at most one full persistence window. This is conservative and avoids future leakage.",
        "",
        f"Example delay bound: `{_fmt(examples['promotion_detection_delay_bound_s'], digits=1)}s`.",
        "",
        "## Empirical Anchors",
        "",
        "| check | status | delta CI95 |",
        "|---|---|---:|",
    ])
    for name, row in payload["cited_checks"].items():
        if not row:
            continue
        lines.append(
            f"| {name} "
            f"| {row.get('status', 'missing')} "
            f"| {row.get('delta_ci95_low', 'NA')} to {row.get('delta_ci95_high', 'NA')} |"
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "These results formalize the Freq-HRL protocol claims. They do not replace large-scale performance validation: native Transit, real AFC/APC/GTFS demand, and deeper order-book feeds still need broader seed and data coverage.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("transit_hrl/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/freq_hrl_theory_appendix"))
    args = parser.parse_args()
    payload = build_theory_payload(args.results_root)
    write_outputs(args.output_dir, payload)
    print(f"wrote {args.output_dir}")
    print(
        "theory_appendix "
        f"fp_bound={payload['examples']['promotion_false_positive_bound_example']:.6f} "
        f"delay_s={payload['examples']['promotion_detection_delay_bound_s']:.1f}"
    )


if __name__ == "__main__":
    main()
