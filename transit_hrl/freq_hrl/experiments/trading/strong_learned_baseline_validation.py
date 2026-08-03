"""Strong learned-baseline validation for Freq-HRL trading.

This runner is intentionally separate from heuristic trading validation.
During the v2 migration it also reports whether trainer and parameter budgets
are genuinely comparable. A mismatch blocks a strong-baseline claim instead
of being hidden by the result summary. SAC/TD3 remain separately registered
because they require real off-policy implementations.
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
from freq_hrl.rl import summarize_numeric_rows

from .performance_validation import SCENARIOS
from .metrics import METRIC_CONTRACT_VERSION
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
CONTRACT_GATED_METRICS = {"sharpe", "total_return"}


def count_parameters(model: Any) -> int:
    modules = [model.upper_actor, model.lower_actor, model.upper_value, model.lower_value]
    if hasattr(model, "lower_cost_value"):
        modules.append(model.lower_cost_value)
    return int(sum(param.numel() for module in modules for param in module.parameters()))


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def selected_scenario_policy_pairs(
    scenarios: list[str],
    policy_modes: list[str],
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[tuple[str, str]]:
    pairs = [(scenario, mode) for scenario in scenarios for mode in policy_modes]
    shards = max(1, int(num_shards))
    index = int(shard_index)
    if index < 0 or index >= shards:
        raise ValueError(f"shard_index must be in [0, {shards - 1}], got {index}")
    return [pair for idx, pair in enumerate(pairs) if idx % shards == index]


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
            relevant = [
                row for row in rows
                if str(row.get("baseline", "")) in {"freq_hrl", control}
                and metric in row
            ]
            contracts = sorted({
                str(row.get("metric_contract_version", "missing"))
                for row in relevant
            })
            contract_valid = bool(
                metric not in CONTRACT_GATED_METRICS
                or (relevant and contracts == [METRIC_CONTRACT_VERSION])
            )
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
                "metric_contract_valid": contract_valid,
                "metric_contract_versions": contracts,
                "status": (
                    claim_status(stats, min_pairs=int(min_pairs))
                    if contract_valid else "invalid_legacy_metric_contract"
                ),
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
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = selected_scenario_policy_pairs(
        scenarios,
        policy_modes,
        shard_index=int(shard_index),
        num_shards=int(num_shards),
    )
    for scenario, mode in pairs:
        rows.append({
            "scenario": scenario,
            "policy_mode": mode,
            "train_seeds": " ".join(str(seed) for seed in train_seeds),
            "eval_seeds": " ".join(str(seed) for seed in eval_seeds),
            "steps": int(steps),
            "assets": int(assets),
            "iterations": int(iterations),
            "trainer": (
                "frequency_separated_smdp_ppo_v2"
                if mode == "freq_hrl" else "legacy_shared_dual_level_ppo"
            ),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
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
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    parameter_budget: list[dict[str, Any]] = []
    sample_efficiency: list[dict[str, Any]] = []
    pairs = selected_scenario_policy_pairs(
        scenarios,
        policy_modes,
        shard_index=int(shard_index),
        num_shards=int(num_shards),
    )
    for pair_idx, (scenario, mode) in enumerate(pairs):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
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
            seed=int(optimizer_seed) + 1009 * pair_idx + 7919 * int(shard_index),
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
            item["shard_index"] = int(shard_index)
            item["num_shards"] = int(num_shards)
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
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
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
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
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
                + 0.01 * float(row.get(
                    "episode_information_ratio",
                    row.get("sharpe", 0.0),
                ))
                for row in heldout_rows
            ])) if heldout_rows else 0.0,
            "elapsed_sec": elapsed,
            "selection_metric": "episode_information_ratio",
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        })
    checks = build_paired_checks(rows, min_pairs=int(min_pairs))
    supported_or_mixed = {
        row["control"] for row in checks
        if row["metric"] in {"sharpe", "total_return", "FocusScore"}
        and row["status"] in {"supported", "positive_mixed"}
    }
    metric_status = (
        "supported" if {"flat_ppo", "generic_hrl_ppo"} <= supported_or_mixed
        else ("partial" if supported_or_mixed else "not_supported")
    )
    parameter_budget_status = (
        "matched"
        if len({row["parameter_count"] for row in parameter_budget}) == 1
        else "mismatch"
    )
    trainer_status = (
        "matched"
        if len({row["trainer"] for row in rows}) <= 1
        else "mismatch"
    )
    ppo_baseline_status = (
        metric_status
        if parameter_budget_status == "matched" and trainer_status == "matched"
        else "not_comparable_during_v2_migration"
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
            shard_index=int(shard_index),
            num_shards=int(num_shards),
        ),
        "summary": {
            "rows": len(rows),
            "scenario_count": len(set(scenarios)),
            "selected_scenario_count": len({scenario for scenario, _ in pairs}),
            "selected_pair_count": len(pairs),
            "policy_modes": list(policy_modes),
            "train_seed_count": len(train_seeds),
            "eval_seed_count": len(eval_seeds),
            "steps": int(steps),
            "assets": int(assets),
            "iterations": int(iterations),
            "ppo_strong_baseline_status": ppo_baseline_status,
            "ppo_metric_status": metric_status,
            "sac_td3_status": "registered_external_missing",
            "parameter_budget_status": parameter_budget_status,
            "trainer_budget_status": trainer_status,
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        },
        "boundary": (
            "This is a transitional v2 comparison. It is eligible for a strong "
            "baseline claim only when trainer_budget_status and "
            "parameter_budget_status are both matched. SAC/TD3 remain missing."
        ),
    }


def merge_strong_learned_baseline_shards(
    input_dirs: list[Path],
    *,
    min_pairs: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    parameter_budget: list[dict[str, Any]] = []
    sample_efficiency: list[dict[str, Any]] = []
    experiment_manifest: list[dict[str, Any]] = []
    for directory in input_dirs:
        base = Path(directory)
        rows.extend(_read_csv(base / "per_seed.csv"))
        run_rows.extend(_read_csv(base / "run_summary.csv"))
        parameter_budget.extend(_read_csv(base / "parameter_budget.csv"))
        sample_efficiency.extend(_read_csv(base / "sample_efficiency.csv"))
        experiment_manifest.extend(_read_csv(base / "experiment_manifest.csv"))
    checks = build_paired_checks(rows, min_pairs=int(min_pairs))
    supported_or_mixed = {
        row["control"] for row in checks
        if row["metric"] in {"sharpe", "total_return", "FocusScore"}
        and row["status"] in {"supported", "positive_mixed"}
    }
    metric_status = (
        "supported" if {"flat_ppo", "generic_hrl_ppo"} <= supported_or_mixed
        else ("partial" if supported_or_mixed else "not_supported")
    )
    param_counts = {
        str(row.get("parameter_count", ""))
        for row in parameter_budget
        if str(row.get("parameter_count", "")).strip()
    }
    scenarios = sorted({str(row.get("scenario", "")) for row in rows if str(row.get("scenario", ""))})
    policy_modes = sorted({str(row.get("policy_mode", row.get("baseline", ""))) for row in rows if str(row.get("policy_mode", row.get("baseline", "")))})
    trainers = {
        str(row.get("trainer", "")) for row in rows if str(row.get("trainer", "")).strip()
    }
    parameter_budget_status = "matched" if len(param_counts) == 1 else "mismatch"
    trainer_status = "matched" if len(trainers) <= 1 else "mismatch"
    ppo_baseline_status = (
        metric_status
        if parameter_budget_status == "matched" and trainer_status == "matched"
        else "not_comparable_during_v2_migration"
    )
    return {
        "per_seed": rows,
        "run_summary": run_rows,
        "policy_summary": _policy_summary(rows),
        "paired_checks": checks,
        "parameter_budget": parameter_budget,
        "sample_efficiency": sample_efficiency,
        "experiment_manifest": experiment_manifest,
        "summary": {
            "rows": len(rows),
            "scenario_count": len(scenarios),
            "selected_scenario_count": len(scenarios),
            "selected_pair_count": len({(row.get("scenario"), row.get("policy_mode")) for row in rows}),
            "policy_modes": policy_modes,
            "shard_count": len(input_dirs),
            "ppo_strong_baseline_status": ppo_baseline_status,
            "ppo_metric_status": metric_status,
            "sac_td3_status": "registered_external_missing",
            "parameter_budget_status": parameter_budget_status,
            "trainer_budget_status": trainer_status,
            "merge_status": "merged",
        },
        "boundary": (
            "Merged shard artifact for PPO-family learned baselines. SAC/TD3 "
            "remain registered external baselines until off-policy implementations run."
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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-inputs", nargs="*", type=Path, default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/strong_learned_baseline_validation_latest"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.merge_inputs:
        payload = merge_strong_learned_baseline_shards(
            list(args.merge_inputs),
            min_pairs=int(args.min_pairs),
        )
    else:
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
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
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
