"""Sensitivity and robustness matrix for Freq-HRL PPO trading."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from freq_hrl.experiments.statistics import noninferiority_status, paired_delta_stats
from freq_hrl.rl import summarize_numeric_rows

from .performance_validation import SCENARIOS
from .ppo_actor_critic import train_ppo_actor_critic
from .strong_learned_baseline_validation import selected_scenario_policy_pairs


DEFAULT_SCENARIOS = (
    "persistent_shift",
    "promotion_recovery",
    "stationary_high_noise",
    "localized_burst",
    "ood_period",
)

SENSITIVITY_PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    "plan_curve": {
        "plan_basis_dim": 3,
        "plan_coefficient_scale": 0.50,
    },
    "leakage_reward": {
        "leakage_scale": 0.00002,
    },
    "lower_lf_constraint": {
        "lower_lf_constraint_coef": 0.05,
        "lower_lf_constraint_target": 0.02,
        "lower_lf_dual_lr": 0.001,
    },
    "raw_recenter": {
        "lower_lf_raw_recenter_gain": 0.03,
        "lower_lf_raw_recenter_scale": 0.08,
    },
}

ROBUSTNESS_METRICS = (
    ("sharpe", False, 0.05),
    ("total_return", False, 0.005),
    ("FocusScore", False, 0.05),
    ("LowerLFDrift", True, 0.02),
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def _profile_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scenario in sorted({str(row.get("scenario", "")) for row in rows}):
        for profile in sorted({str(row.get("profile", "")) for row in rows}):
            group = [
                row for row in rows
                if str(row.get("scenario", "")) == scenario
                and str(row.get("profile", "")) == profile
            ]
            if not group:
                continue
            out.append({
                "scenario": scenario,
                "profile": profile,
                **summarize_numeric_rows(
                    group,
                    keys=["sharpe", "total_return", "FocusScore", "LowerLFDrift", "turnover"],
                ),
            })
    return out


def build_robustness_checks(
    rows: list[dict[str, Any]],
    *,
    profiles: list[str],
    min_pairs: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for profile in profiles:
        if profile == "default":
            continue
        for metric, lower_is_better, margin in ROBUSTNESS_METRICS:
            stats = paired_delta_stats(
                rows,
                variant_key="profile",
                pair_keys=("scenario", "seed"),
                metric=metric,
                treatment=profile,
                control="default",
                lower_is_better=lower_is_better,
            )
            checks.append({
                "check": f"{profile}_vs_default_{metric}_noninferiority",
                **stats,
                "noninferiority_margin": float(margin),
                "status": noninferiority_status(
                    stats,
                    max_loss=float(margin),
                    min_pairs=int(min_pairs),
                ),
            })
    return checks


def build_experiment_manifest(
    *,
    scenarios: list[str],
    profiles: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    pairs = selected_scenario_policy_pairs(
        scenarios,
        profiles,
        shard_index=int(shard_index),
        num_shards=int(num_shards),
    )
    rows: list[dict[str, Any]] = []
    for scenario, profile in pairs:
        rows.append({
            "scenario": scenario,
            "profile": profile,
            "train_seeds": " ".join(str(seed) for seed in train_seeds),
            "eval_seeds": " ".join(str(seed) for seed in eval_seeds),
            "steps": int(steps),
            "assets": int(assets),
            "iterations": int(iterations),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.trading.sensitivity_robustness_matrix "
                f"--scenarios {scenario} --profiles {profile} "
                f"--steps {int(steps)} --assets {int(assets)} --iterations {int(iterations)} "
                "--train-seeds "
                + " ".join(str(seed) for seed in train_seeds)
                + " --eval-seeds "
                + " ".join(str(seed) for seed in eval_seeds)
            ),
        })
    return rows


def run_sensitivity_robustness_matrix(
    *,
    scenarios: list[str],
    profiles: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
    optimizer_seed: int,
    min_pairs: int,
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    pairs = selected_scenario_policy_pairs(
        scenarios,
        profiles,
        shard_index=int(shard_index),
        num_shards=int(num_shards),
    )
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for pair_idx, (scenario, profile) in enumerate(pairs):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if profile not in SENSITIVITY_PROFILES:
            raise ValueError(f"unknown sensitivity profile: {profile}")
        params = dict(SENSITIVITY_PROFILES[profile])
        payload, heldout_rows, _ = train_ppo_actor_critic(
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            steps=int(steps),
            assets=int(assets),
            scenario=scenario,
            iterations=int(iterations),
            seed=int(optimizer_seed) + 1009 * pair_idx + 7919 * int(shard_index),
            policy_mode="freq_hrl",
            **params,
        )
        for row in heldout_rows:
            item = dict(row)
            item["scenario"] = scenario
            item["profile"] = profile
            item["source_artifact"] = "sensitivity_robustness_matrix"
            item["shard_index"] = int(shard_index)
            item["num_shards"] = int(num_shards)
            rows.append(item)
        run_rows.append({
            "scenario": scenario,
            "profile": profile,
            "train_seed_count": len(train_seeds),
            "eval_seed_count": len(eval_seeds),
            "steps": int(steps),
            "iterations": int(iterations),
            "sharpe_mean": float(payload["summary"].get("sharpe_mean", 0.0)),
            "total_return_mean": float(payload["summary"].get("total_return_mean", 0.0)),
            "FocusScore_mean": float(payload["summary"].get("FocusScore_mean", 0.0)),
            "LowerLFDrift_mean": float(payload["summary"].get("LowerLFDrift_mean", 0.0)),
            "profile_params": json.dumps(params, sort_keys=True),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        })
    checks = build_robustness_checks(rows, profiles=profiles, min_pairs=int(min_pairs))
    passing = [
        row for row in checks
        if row.get("status") in {"supported", "positive_mixed"}
    ]
    profile_count = len([profile for profile in profiles if profile != "default"])
    expected_checks = profile_count * len(ROBUSTNESS_METRICS)
    return {
        "per_seed": rows,
        "run_summary": run_rows,
        "profile_summary": _profile_summary(rows),
        "paired_checks": checks,
        "experiment_manifest": build_experiment_manifest(
            scenarios=scenarios,
            profiles=profiles,
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            steps=int(steps),
            assets=int(assets),
            iterations=int(iterations),
            shard_index=int(shard_index),
            num_shards=int(num_shards),
        ),
        "summary": {
            "rows": len(rows),
            "scenario_count": len(set(scenarios)),
            "selected_pair_count": len(pairs),
            "profile_count": len(profiles),
            "robustness_check_count": len(checks),
            "robustness_pass_count": len(passing),
            "robustness_status": (
                "supported" if expected_checks > 0 and len(passing) == expected_checks
                else ("partial" if passing else "registered_executable")
            ),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        },
        "boundary": (
            "Sensitivity evidence is a stress-registered noninferiority check "
            "over Freq-HRL hyperparameter profiles, not a universal robustness claim."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_seed.csv", payload["per_seed"])
    _write_csv(output_dir / "run_summary.csv", payload["run_summary"])
    _write_csv(output_dir / "profile_summary.csv", payload["profile_summary"])
    _write_csv(output_dir / "paired_checks.csv", payload["paired_checks"])
    _write_csv(output_dir / "experiment_manifest.csv", payload["experiment_manifest"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Sensitivity Robustness Matrix",
        "",
        payload["boundary"],
        "",
        f"- status: `{payload['summary']['robustness_status']}`",
        f"- checks: `{payload['summary']['robustness_check_count']}`",
        f"- pass count: `{payload['summary']['robustness_pass_count']}`",
        "",
        "| check | status | metric | n | improvement | CI95 low | CI95 high | margin |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["paired_checks"]:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['metric']} "
            f"| {row['n_common']} | {row['improvement_mean']:+.4f} "
            f"| {row['improvement_ci95_low']:+.4f} | {row['improvement_ci95_high']:+.4f} "
            f"| {row['noninferiority_margin']:.4f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--profiles", nargs="+", choices=sorted(SENSITIVITY_PROFILES), default=list(SENSITIVITY_PROFILES))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[31415, 27182, 16180])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/sensitivity_robustness_matrix_latest"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload = run_sensitivity_robustness_matrix(
        scenarios=list(args.scenarios),
        profiles=list(args.profiles),
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        steps=int(args.steps),
        assets=int(args.assets),
        iterations=int(args.iterations),
        optimizer_seed=int(args.optimizer_seed),
        min_pairs=int(args.min_pairs),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    write_outputs(args.output_dir, payload)
    print(
        "sensitivity_robustness_matrix "
        f"status={payload['summary']['robustness_status']} "
        f"checks={payload['summary']['robustness_check_count']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
