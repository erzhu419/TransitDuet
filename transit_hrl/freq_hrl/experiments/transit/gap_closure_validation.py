"""Transit Freq-HRL gap-closure validation matrix.

The matrix is intentionally small enough to run locally, but each variant maps
to a specific claim from the FreqTransitDuet manual: count-likelihood demand
state, learned plan curves, promotion-triggered replanning, lower native HF
context, wait-credit reward, and lower-drift constraints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.transit.ppo_surrogate import train_transit_surrogate_ppo


VARIANTS: dict[str, dict[str, Any]] = {
    "base_ema_direct": {
        "tracker_method": "ema",
        "plan_basis_dim": 0,
        "include_native_lower_context": False,
        "upper_decision_interval": 1,
    },
    "plan_only": {
        "tracker_method": "ema",
        "plan_basis_dim": 2,
        "include_native_lower_context": False,
        "upper_decision_interval": 8,
        "plan_coefficient_scale_s": 1.0,
    },
    "full_no_wait": {
        "tracker_method": "dynamic_harmonic_nb",
        "plan_basis_dim": 2,
        "include_native_lower_context": True,
        "upper_decision_interval": 8,
        "promotion_forced_replan": True,
        "promotion_replan_strength_min": 0.10,
        "promotion_residual_threshold": 0.55,
        "promotion_persistence_ratio": 0.20,
        "plan_coefficient_scale_s": 1.0,
        "lower_lf_effect_filter_window": 12,
        "lower_lf_effect_filter_gain": 1.0,
        "lower_lf_raw_recenter_gain": 1.0,
    },
    "full_freqhrl": {
        "tracker_method": "dynamic_harmonic_nb",
        "plan_basis_dim": 2,
        "include_native_lower_context": True,
        "upper_decision_interval": 8,
        "promotion_forced_replan": True,
        "promotion_replan_strength_min": 0.10,
        "promotion_residual_threshold": 0.55,
        "promotion_persistence_ratio": 0.20,
        "plan_coefficient_scale_s": 1.0,
        "lower_lf_effect_filter_window": 12,
        "lower_lf_effect_filter_gain": 1.0,
        "lower_lf_raw_recenter_gain": 1.0,
        "wait_upper_weight": 0.005,
        "wait_lower_weight": 0.010,
        "wait_lower_board_credit_weight": 0.10,
        "wait_credit_control_gain": 2.0,
        "lower_lf_constraint_coef": 0.02,
        "lower_lf_constraint_target": 0.55,
        "lower_lf_dual_lr": 0.02,
        "lower_lf_objective_weight": 0.02,
    },
}


def row_objective(row: dict[str, Any]) -> float:
    return (
        float(row["reward_mean_mean"])
        - 0.08 * float(row["LowerLFDrift_mean"])
        - 0.04 * float(row["wait_proxy_mean"])
        - 0.01 * float(row["hold_mean_mean"])
    )


def run_gap_closure_matrix(
    output_dir: Path,
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    iterations: int,
    corridors: int,
    scenario: str,
    optimizer_seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    for idx, (name, overrides) in enumerate(VARIANTS.items()):
        payload, per_seed, _ = train_transit_surrogate_ppo(
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            steps=steps,
            corridors=corridors,
            scenario=scenario,
            iterations=iterations,
            seed=optimizer_seed + idx,
            **overrides,
        )
        payloads[name] = {"model": payload, "per_seed": per_seed}
        row = {"variant": name, **payload["summary"]}
        row["claim_objective"] = row_objective(row)
        rows.append(row)

    base = next(row for row in rows if row["variant"] == "base_ema_direct")
    for row in rows:
        row["delta_claim_objective_vs_base"] = (
            row["claim_objective"] - base["claim_objective"]
        )
        row["delta_wait_vs_base"] = row["wait_proxy_mean"] - base["wait_proxy_mean"]
        row["delta_lower_lf_drift_vs_base"] = (
            row["LowerLFDrift_mean"] - base["LowerLFDrift_mean"]
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "payloads": payloads}, f, indent=2)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_report(output_dir / "report.md", rows)
    return {"rows": rows, "payloads": payloads}


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Transit Freq-HRL Gap-Closure Matrix",
        "",
        "| variant | objective | delta obj | reward | wait | LowerLFDrift | upper decisions | promotion replans |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} "
            f"| {row['claim_objective']:.4f} "
            f"| {row['delta_claim_objective_vs_base']:+.4f} "
            f"| {row['reward_mean_mean']:.4f} "
            f"| {row['wait_proxy_mean']:.4f} "
            f"| {row['LowerLFDrift_mean']:.4f} "
            f"| {row['upper_decision_count_mean']:.1f} "
            f"| {row['promotion_replan_count_mean']:.1f} |"
        )
    lines.extend([
        "",
        "The `full_freqhrl` row is the integrated claim path: dynamic harmonic NB demand state, learned Bernstein plan actions, low-frequency upper reuse, promotion-triggered learned replanning, native lower HF context, wait-attributed reward, and lower-drift constraint.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 131, 151])
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--corridors", type=int, default=2)
    parser.add_argument("--scenario", default="persistent_shift")
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_gap_closure"),
    )
    args = parser.parse_args()
    result = run_gap_closure_matrix(
        output_dir=args.output_dir,
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        steps=int(args.steps),
        iterations=int(args.iterations),
        corridors=int(args.corridors),
        scenario=str(args.scenario),
        optimizer_seed=int(args.optimizer_seed),
    )
    best = max(result["rows"], key=lambda row: row["claim_objective"])
    print(f"wrote {args.output_dir}")
    print(
        "transit_gap_closure "
        f"best={best['variant']} "
        f"objective={best['claim_objective']:.4f} "
        f"delta={best['delta_claim_objective_vs_base']:+.4f}"
    )


if __name__ == "__main__":
    main()
