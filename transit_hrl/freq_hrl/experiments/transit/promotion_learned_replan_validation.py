"""Validate learned promotion-triggered replanning in Transit PPO."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.transit.ppo_surrogate import train_transit_surrogate_ppo


VARIANTS: dict[str, dict[str, Any]] = {
    "interval_only": {
        "promotion_forced_replan": False,
        "promotion_learned_replan": False,
    },
    "deterministic_forced": {
        "promotion_forced_replan": True,
        "promotion_learned_replan": False,
    },
    "learned_gate": {
        "promotion_forced_replan": False,
        "promotion_learned_replan": True,
        "promotion_learned_gate_threshold": 0.55,
    },
}


COMMON_CONFIG: dict[str, Any] = {
    "tracker_method": "dynamic_harmonic_nb",
    "plan_basis_dim": 2,
    "plan_coefficient_scale_s": 1.0,
    "include_native_lower_context": True,
    "upper_decision_interval": 8,
    "promotion_replan_strength_min": 0.0,
    "promotion_replan_recovery_gain": 0.060,
    "promotion_residual_threshold": 0.55,
    "promotion_persistence_ratio": 0.20,
    "wait_credit_control_gain": 2.0,
    "lower_lf_effect_filter_window": 12,
    "lower_lf_effect_filter_gain": 1.0,
    "lower_lf_raw_recenter_gain": 1.0,
}


def flatten_payload_rows(payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, payload in payloads.items():
        for row in payload.get("per_seed", []):
            item = dict(row)
            item["variant"] = variant
            rows.append(item)
    return rows


def paired_checks(payloads: dict[str, Any]) -> list[dict[str, Any]]:
    rows = flatten_payload_rows(payloads)
    checks = []
    for metric, lower_is_better in [
        ("reward_mean", False),
        ("wait_proxy", True),
        ("RawLowerLFDriftAbs", True),
        ("promotion_replan_count", False),
    ]:
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("seed",),
            metric=metric,
            treatment="learned_gate",
            control="interval_only",
            lower_is_better=lower_is_better,
        )
        checks.append({
            "check": f"learned_gate_vs_interval_{metric}",
            **stats,
            "status": claim_status(stats, min_pairs=3),
        })
    return checks


def run_validation(
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
    for idx, (variant, overrides) in enumerate(VARIANTS.items()):
        payload, per_seed, _ = train_transit_surrogate_ppo(
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            steps=steps,
            corridors=corridors,
            scenario=scenario,
            iterations=iterations,
            seed=optimizer_seed + idx,
            **COMMON_CONFIG,
            **overrides,
        )
        payloads[variant] = {"model": payload, "per_seed": per_seed}
        rows.append({"variant": variant, **payload["summary"]})

    base = next(row for row in rows if row["variant"] == "interval_only")
    for row in rows:
        row["delta_reward_vs_interval"] = row["reward_mean_mean"] - base["reward_mean_mean"]
        row["delta_wait_vs_interval"] = row["wait_proxy_mean"] - base["wait_proxy_mean"]
        row["delta_replans_vs_interval"] = row["promotion_replan_count_mean"] - base["promotion_replan_count_mean"]
        row["delta_raw_lower_lf_abs_vs_interval"] = (
            row["RawLowerLFDriftAbs_mean"] - base["RawLowerLFDriftAbs_mean"]
        )
    checks = paired_checks(payloads)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows, "payloads": payloads, "paired_checks": checks}, f, indent=2)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "paired_checks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checks)
    write_report(output_dir / "report.md", rows, checks)
    return {"rows": rows, "payloads": payloads, "paired_checks": checks}


def write_report(path: Path, rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    lines = [
        "# Transit Learned Promotion Replan Validation",
        "",
        "| variant | reward | wait | raw drift abs | replans | gate | delta reward | delta wait |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} "
            f"| {row['reward_mean_mean']:.4f} "
            f"| {row['wait_proxy_mean']:.4f} "
            f"| {row['RawLowerLFDriftAbs_mean']:.6f} "
            f"| {row['promotion_replan_count_mean']:.2f} "
            f"| {row['promotion_gate_value_mean']:.3f} "
            f"| {row['delta_reward_vs_interval']:+.4f} "
            f"| {row['delta_wait_vs_interval']:+.4f} |"
        )
    lines.extend([
        "",
        "Paired checks compare `learned_gate` against `interval_only` by eval seed.",
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in checks:
        lines.append(
            f"| {row['check']} "
            f"| {row['status']} "
            f"| {row['metric']} "
            f"| {row['n_common']} "
            f"| {row['delta_mean']:+.4f} "
            f"| {row['delta_ci95_low']:+.4f} "
            f"| {row['delta_ci95_high']:+.4f} "
            f"| {row['win_rate']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 23])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 131, 151])
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--corridors", type=int, default=2)
    parser.add_argument("--scenario", default="persistent_shift")
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_promotion_learned_replan"),
    )
    args = parser.parse_args()
    result = run_validation(
        output_dir=args.output_dir,
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        steps=int(args.steps),
        iterations=int(args.iterations),
        corridors=int(args.corridors),
        scenario=str(args.scenario),
        optimizer_seed=int(args.optimizer_seed),
    )
    learned = next(row for row in result["rows"] if row["variant"] == "learned_gate")
    print(f"wrote {args.output_dir}")
    print(
        "transit_learned_promotion "
        f"reward_delta={learned['delta_reward_vs_interval']:+.4f} "
        f"wait_delta={learned['delta_wait_vs_interval']:+.4f} "
        f"replans={learned['promotion_replan_count_mean']:.1f}"
    )


if __name__ == "__main__":
    main()
