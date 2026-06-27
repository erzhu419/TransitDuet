"""Strong learned-baseline validation for Freq-HRL trading.

This runner is intentionally separate from heuristic trading validation.  It
compares Freq-HRL PPO against learned flat PPO and learned generic HRL PPO
through the same shared actor-critic trainer and matched parameter budgets.
SAC/TD3 are registered in the CS experiment matrix because they require a
separate off-policy implementation rather than a mislabeled PPO surrogate.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.rl import DualActorCriticPPO, summarize_numeric_rows

from .performance_validation import SCENARIOS
from .ppo_actor_critic import POLICY_MODES, train_ppo_actor_critic


DEFAULT_SCENARIOS = (
    "persistent_shift",
    "stationary_high_noise",
    "localized_burst",
    "ood_period",
)
DEFAULT_POLICY_MODES = (
    "freq_hrl",
    "flat_ppo",
    "generic_hrl_ppo",
)
MAIN_METRICS = (
    ("sharpe", False),
    ("total_return", False),
    ("FocusScore", False),
    ("LowerLFDrift", True),
)


def count_parameters(model: DualActorCriticPPO) -> int:
    return int(sum(param.numel() for param in (
        list(model.upper_actor.parameters())
        + list(model.lower_actor.parameters())
        + list(model.upper_value.parameters())
        + list(model.lower_value.parameters())
    )))


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


def _policy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for scenario in sorted({str(row.get("scenario", "")) for row in rows}):
        for mode in sorted({str(row.get("baseline", "")) for row in rows}):
            group = [
                row for row in rows
                if str(row.get("scenario", "")) == scenario
                and str(row.get("baseline", "")) == mode
            ]
            if not group:
                continue
            summary = summarize_numeric_rows(
                group,
                keys=[
                    "sharpe",
                    "total_return",
                    "FocusScore",
                    "LowerLFDrift",
                    "turnover",
                    "promotion_count",
                ],
            )
            out.append({
                "scenario": scenario,
                "baseline": mode,
                **summary,
            })
    return out


def build_paired_checks(
    rows: list[dict[str, Any]],
    *,
    controls: tuple[str, ...] = ("flat_ppo", "generic_hrl_ppo"),
    min_pairs: int = 3,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for control in controls:
        for metric, lower_is_better in MAIN_METRICS:
            stats = paired_delta_stats(
                rows,
                variant_key="baseline",
                pair_keys=("scenario", "seed"),
                metric=metric,
                treatment="freq_hrl",
                control=control,
                lower_is_better=lower_is_better,
            )
            checks.append({
                "check": f"freq_hrl_vs_{control}_{metric}",
                **stats,
                "status": claim_status(stats, min_pairs=int(min_pairs)),
                "baseline_class": "strong_learned_ppo",
            })
    return checks


def build_experiment_manifest(
    *,
    scenarios: list[str],
    policy_modes: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for mode in policy_modes:
            rows.append({
                "scenario": scenario,
                "policy_mode": mode,
                "train_seeds": " ".join(str(seed) for seed in train_seeds),
                "eval_seeds": " ".join(str(seed) for seed in eval_seeds),
                "steps": int(steps),
                "assets": int(assets),
                "iterations": int(iterations),
                "trainer": "shared_dual_level_ppo",
                "command": (
                    "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                    "freq_hrl.experiments.trading.ppo_actor_critic "
                    f"--scenario {scenario} --policy-mode {mode} "
                    f"--steps {int(steps)} --assets {int(assets)} --iterations {int(iterations)} "
                    "--train-seeds "
                    + " ".join(str(seed) for seed in train_seeds)
                    + " --eval-seeds "
                    + " ".join(str(seed) for seed in eval_seeds)
                ),
            })
    return rows


def run_strong_learned_baseline_validation(
    *,
    scenarios: list[str],
    policy_modes: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
    optimizer_seed: int,
    min_pairs: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    parameter_budget: list[dict[str, Any]] = []
    sample_efficiency: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        for mode_idx, mode in enumerate(policy_modes):
            if mode not in POLICY_MODES:
                raise ValueError(f"unknown policy_mode: {mode}")
            start = time.perf_counter()
            payload, heldout_rows, model = train_ppo_actor_critic(
                train_seeds=train_seeds,
                eval_seeds=eval_seeds,
                steps=int(steps),
                assets=int(assets),
                scenario=scenario,
                iterations=int(iterations),
                seed=int(optimizer_seed) + 1009 * mode_idx + 7919 * len(rows),
                policy_mode=mode,
            )
            elapsed = float(time.perf_counter() - start)
            params = count_parameters(model)
            for row in heldout_rows:
                item = dict(row)
                item["scenario"] = scenario
                item["baseline"] = mode
                item["policy_mode"] = mode
                item["trainer"] = payload["trainer"]
                item["source_artifact"] = "strong_learned_baseline_validation"
                rows.append(item)
            run_rows.append({
                "scenario": scenario,
                "policy_mode": mode,
                "elapsed_sec": elapsed,
                "train_seed_count": len(train_seeds),
                "eval_seed_count": len(eval_seeds),
                "steps": int(steps),
                "iterations": int(iterations),
                "parameter_count": params,
                "best_score": float(payload.get("best_score", 0.0)),
                "sharpe_mean": float(payload["summary"].get("sharpe_mean", 0.0)),
                "total_return_mean": float(payload["summary"].get("total_return_mean", 0.0)),
                "FocusScore_mean": float(payload["summary"].get("FocusScore_mean", 0.0)),
                "LowerLFDrift_mean": float(payload["summary"].get("LowerLFDrift_mean", 0.0)),
            })
            parameter_budget.append({
                "scenario": scenario,
                "policy_mode": mode,
                "parameter_count": params,
                "upper_state_dim": int(model.config.upper_state_dim),
                "lower_state_dim": int(model.config.lower_state_dim),
                "upper_action_dim": int(model.config.upper_action_dim),
                "lower_action_dim": int(model.config.lower_action_dim),
                "hidden_dim": int(model.config.hidden_dim),
                "matched_budget_group": "trading_shared_dual_ppo_linear",
            })
            train_steps = int(len(train_seeds) * steps * max(1, int(iterations)))
            sample_efficiency.append({
                "scenario": scenario,
                "policy_mode": mode,
                "environment_steps_train": train_steps,
                "environment_steps_eval": int(len(eval_seeds) * steps),
                "iterations": int(iterations),
                "best_score": float(payload.get("best_score", 0.0)),
                "heldout_objective_proxy": float(np.mean([
                    float(row.get("total_return", 0.0))
                    + 0.01 * float(row.get("sharpe", 0.0))
                    for row in heldout_rows
                ])) if heldout_rows else 0.0,
                "elapsed_sec": elapsed,
            })
    checks = build_paired_checks(rows, min_pairs=int(min_pairs))
    supported_or_mixed = {
        row["control"] for row in checks
        if row["metric"] in {"sharpe", "total_return", "FocusScore"}
        and row["status"] in {"supported", "positive_mixed"}
    }
    ppo_baseline_status = (
        "supported" if {"flat_ppo", "generic_hrl_ppo"} <= supported_or_mixed
        else ("partial" if supported_or_mixed else "not_supported")
    )
    return {
        "per_seed": rows,
        "run_summary": run_rows,
        "policy_summary": _policy_summary(rows),
        "paired_checks": checks,
        "parameter_budget": parameter_budget,
        "sample_efficiency": sample_efficiency,
        "experiment_manifest": build_experiment_manifest(
            scenarios=scenarios,
            policy_modes=policy_modes,
            train_seeds=train_seeds,
            eval_seeds=eval_seeds,
            steps=int(steps),
            assets=int(assets),
            iterations=int(iterations),
        ),
        "summary": {
            "rows": len(rows),
            "scenario_count": len(set(scenarios)),
            "policy_modes": list(policy_modes),
            "train_seed_count": len(train_seeds),
            "eval_seed_count": len(eval_seeds),
            "steps": int(steps),
            "assets": int(assets),
            "iterations": int(iterations),
            "ppo_strong_baseline_status": ppo_baseline_status,
            "sac_td3_status": "registered_external_missing",
            "parameter_budget_status": (
                "matched"
                if len({row["parameter_count"] for row in parameter_budget}) == 1
                else "mismatch"
            ),
        },
        "boundary": (
            "This artifact closes the PPO-family learned baseline path under a "
            "matched shared-core parameter budget. It does not claim SAC/TD3 "
            "coverage; those are registered as remaining off-policy baselines."
        ),
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_seed.csv", payload["per_seed"])
    _write_csv(output_dir / "paired_checks.csv", payload["paired_checks"])
    _write_csv(output_dir / "policy_summary.csv", payload["policy_summary"])
    _write_csv(output_dir / "run_summary.csv", payload["run_summary"])
    _write_csv(output_dir / "parameter_budget.csv", payload["parameter_budget"])
    _write_csv(output_dir / "sample_efficiency.csv", payload["sample_efficiency"])
    _write_csv(output_dir / "experiment_manifest.csv", payload["experiment_manifest"])
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Strong Learned Baseline Validation",
        "",
        payload["boundary"],
        "",
        f"- PPO baseline status: `{payload['summary']['ppo_strong_baseline_status']}`",
        f"- SAC/TD3 status: `{payload['summary']['sac_td3_status']}`",
        f"- parameter budget: `{payload['summary']['parameter_budget_status']}`",
        f"- scenarios: `{payload['summary']['scenario_count']}`",
        f"- eval seeds: `{payload['summary']['eval_seed_count']}`",
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
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--policy-modes", nargs="+", choices=POLICY_MODES, default=list(DEFAULT_POLICY_MODES))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[31415, 27182, 16180])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/strong_learned_baseline_validation_latest"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload = run_strong_learned_baseline_validation(
        scenarios=list(args.scenarios),
        policy_modes=list(args.policy_modes),
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        steps=int(args.steps),
        assets=int(args.assets),
        iterations=int(args.iterations),
        optimizer_seed=int(args.optimizer_seed),
        min_pairs=int(args.min_pairs),
    )
    write_outputs(args.output_dir, payload)
    print(
        "strong_learned_baseline_validation "
        f"status={payload['summary']['ppo_strong_baseline_status']} "
        f"rows={payload['summary']['rows']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
