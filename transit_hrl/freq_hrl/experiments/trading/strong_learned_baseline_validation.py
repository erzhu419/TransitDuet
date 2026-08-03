"""Strong learned-baseline validation for Freq-HRL trading.

This runner is intentionally separate from heuristic trading validation.
During the v2 migration it also reports whether trainer and parameter budgets
are genuinely comparable. A mismatch blocks a strong-baseline claim instead
of being hidden by the result summary. SAC and TD3 use complete local
off-policy implementations and share the environment-step and metric budgets;
their necessarily different network/optimizer families are reported rather
than mislabeled as exact parameter matches.
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

from freq_hrl.experiments.statistics import (
    apply_holm_correction,
    finite_float,
    paired_delta_stats,
)
from freq_hrl.rl import summarize_numeric_rows

from .performance_validation import SCENARIOS
from .metrics import METRIC_CONTRACT_VERSION
from .offpolicy_baseline_validation import (
    OFFPOLICY_MODES,
    train_flat_offpolicy_baseline,
)
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
    "flat_sac",
    "flat_td3",
)
ALL_POLICY_MODES = POLICY_MODES + OFFPOLICY_MODES
MAIN_METRICS = (
    ("total_return", False),
    ("episode_information_ratio", False),
    ("FocusScore", False),
    ("LowerLFDrift", True),
)
CONTRACT_GATED_METRICS = {"episode_information_ratio", "total_return"}
CONFIRMATORY_FAMILY = "strong_learned_all_baselines_endpoints"


def count_parameters(model: Any) -> int:
    if hasattr(model, "parameters"):
        parameters = list(model.parameters())
    else:
        modules = [
            model.upper_actor,
            model.lower_actor,
            model.upper_value,
            model.lower_value,
        ]
        if hasattr(model, "lower_cost_value"):
            modules.append(model.lower_cost_value)
        parameters = [
            parameter for module in modules for parameter in module.parameters()
        ]
    unique = {id(parameter): parameter for parameter in parameters}
    return int(sum(
        parameter.numel()
        for parameter in unique.values()
        if parameter.requires_grad
    ))


def _parameter_budget_row(
    model: Any,
    *,
    scenario: str,
    mode: str,
    parameter_count: int,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    config = model.config
    common = {
        "scenario": scenario,
        "policy_mode": mode,
        "parameter_count": int(parameter_count),
        "hidden_dim": int(config.hidden_dim),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
    }
    if mode in POLICY_MODES:
        return {
            **common,
            "algorithm_family": "on_policy_smdp_ppo",
            "state_dim": "",
            "action_dim": "",
            "upper_state_dim": int(config.upper_state_dim),
            "lower_state_dim": int(config.lower_state_dim),
            "upper_action_dim": int(config.upper_action_dim),
            "lower_action_dim": int(config.lower_action_dim),
            "matched_budget_group": "trading_capacity_matched_smdp_ppo_v2",
            "capacity_contract": "exact trainable-parameter and optimizer match within PPO family",
        }
    return {
        **common,
        "algorithm_family": f"off_policy_{config.algorithm}",
        "state_dim": int(config.state_dim),
        "action_dim": int(config.action_dim),
        "upper_state_dim": "",
        "lower_state_dim": "",
        "upper_action_dim": "",
        "lower_action_dim": "",
        "matched_budget_group": f"standard_flat_{config.algorithm}_twin_q_v1",
        "capacity_contract": "same environment-step budget; architecture follows algorithm family",
    }


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
                    "episode_information_ratio",
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
    controls: tuple[str, ...] = (
        "flat_ppo",
        "generic_hrl_ppo",
        "flat_sac",
        "flat_td3",
    ),
    min_pairs: int = 10,
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
                "multiplicity_family": CONFIRMATORY_FAMILY,
                "baseline_class": (
                    "strong_learned_ppo"
                    if control in POLICY_MODES else "strong_learned_offpolicy"
                ),
            })
    corrected = apply_holm_correction(
        checks,
        family_key="multiplicity_family",
        p_key="sign_p_value",
        alpha=0.05,
    )
    for row in corrected:
        n_independent = int(row.get("n_independent", 0) or 0)
        improvement = finite_float(row.get("improvement_mean"))
        ci_low = finite_float(row.get("improvement_ci95_low"))
        if not bool(row.get("metric_contract_valid", False)):
            status = "invalid_legacy_metric_contract"
        elif n_independent < int(min_pairs):
            status = "underpowered"
        elif ci_low is not None and ci_low > 0.0 and bool(row.get("holm_reject", False)):
            status = "supported"
        elif improvement is not None and improvement > 0.0:
            status = "positive_mixed"
        elif improvement is not None and improvement <= 0.0:
            status = "not_supported"
        else:
            status = "inconclusive"
        row["status"] = status
        row["confirmatory_gate"] = (
            "min independent seeds + positive cluster-bootstrap CI + "
            "Holm-adjusted two-sided sign test"
        )
    return corrected


def _metric_evidence_status(
    checks: list[dict[str, Any]],
    controls: tuple[str, ...],
    *,
    metrics: tuple[str, ...],
) -> str:
    if not controls:
        return "not_run"
    evidence: dict[tuple[str, str], str] = {}
    for row in checks:
        control = str(row.get("control", ""))
        metric = str(row.get("metric", ""))
        if control in controls and metric in metrics:
            evidence[(control, metric)] = str(row.get("status", ""))
    expected = [(control, metric) for control in controls for metric in metrics]
    statuses = [evidence.get(key, "missing") for key in expected]
    if statuses and all(status == "supported" for status in statuses):
        return "supported"
    if statuses and all(
        status in {"supported", "positive_mixed"} for status in statuses
    ):
        return "positive_mixed"
    if any(status in {"supported", "positive_mixed"} for status in statuses):
        return "partial"
    return "not_supported"


def build_experiment_manifest(
    *,
    scenarios: list[str],
    policy_modes: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
    optimizer_seed: int = 2026,
    offpolicy_hidden_dim: int = 64,
    offpolicy_replay_capacity: int = 100_000,
    offpolicy_warmup_steps: int = 256,
    offpolicy_batch_size: int = 64,
    offpolicy_updates_per_step: int = 1,
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
    scenario_rank = {scenario: idx for idx, scenario in enumerate(scenarios)}
    for scenario, mode in pairs:
        run_seed = int(optimizer_seed) + 1009 * int(scenario_rank[scenario])
        if mode in POLICY_MODES:
            trainer = "frequency_separated_smdp_ppo_v2"
            module = "freq_hrl.experiments.trading.ppo_actor_critic"
            mode_arg = f"--policy-mode {mode}"
            extra_args = ""
        else:
            algorithm = "sac" if mode == "flat_sac" else "td3"
            trainer = f"flat_{algorithm}_twin_q_v1"
            module = "freq_hrl.experiments.trading.offpolicy_baseline_validation"
            mode_arg = f"--policy-mode {mode}"
            extra_args = (
                f" --hidden-dim {int(offpolicy_hidden_dim)}"
                f" --replay-capacity {int(offpolicy_replay_capacity)}"
                f" --warmup-steps {int(offpolicy_warmup_steps)}"
                f" --batch-size {int(offpolicy_batch_size)}"
                f" --updates-per-step {int(offpolicy_updates_per_step)}"
            )
        rows.append({
            "scenario": scenario,
            "policy_mode": mode,
            "train_seeds": " ".join(str(seed) for seed in train_seeds),
            "eval_seeds": " ".join(str(seed) for seed in eval_seeds),
            "steps": int(steps),
            "assets": int(assets),
            "iterations": int(iterations),
            "trainer": trainer,
            "optimizer_seed": run_seed,
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                f"{module} "
                f"--scenario {scenario} {mode_arg} "
                f"--steps {int(steps)} --assets {int(assets)} --iterations {int(iterations)} "
                f"--optimizer-seed {run_seed} "
                "--train-seeds "
                + " ".join(str(seed) for seed in train_seeds)
                + " --eval-seeds "
                + " ".join(str(seed) for seed in eval_seeds)
                + extra_args
            ),
        })
    return rows


def _budget_statuses(
    rows: list[dict[str, Any]],
    parameter_budget: list[dict[str, Any]],
    sample_efficiency: list[dict[str, Any]],
) -> dict[str, str]:
    modes = {
        str(row.get("policy_mode", row.get("baseline", "")))
        for row in rows
    }
    ppo_modes = modes & set(POLICY_MODES)
    offpolicy_modes = modes & set(OFFPOLICY_MODES)
    ppo_counts_by_scenario: dict[str, set[int]] = {}
    for row in parameter_budget:
        if str(row.get("policy_mode", "")) not in POLICY_MODES:
            continue
        value = str(row.get("parameter_count", "")).strip()
        if not value:
            continue
        ppo_counts_by_scenario.setdefault(str(row.get("scenario", "")), set()).add(
            int(float(value))
        )
    if not ppo_modes:
        ppo_parameter_status = "not_run"
    elif ppo_counts_by_scenario and all(
        len(values) == 1 for values in ppo_counts_by_scenario.values()
    ):
        ppo_parameter_status = "matched"
    else:
        ppo_parameter_status = "mismatch"

    ppo_trainers = {
        str(row.get("trainer", ""))
        for row in rows
        if str(row.get("policy_mode", row.get("baseline", ""))) in POLICY_MODES
        and str(row.get("trainer", "")).strip()
    }
    ppo_trainer_status = (
        "not_run" if not ppo_modes
        else ("matched" if len(ppo_trainers) == 1 else "mismatch")
    )

    train_steps_by_scenario: dict[str, set[int]] = {}
    for row in sample_efficiency:
        value = str(row.get("environment_steps_train", "")).strip()
        if not value:
            continue
        train_steps_by_scenario.setdefault(str(row.get("scenario", "")), set()).add(
            int(float(value))
        )
    environment_step_status = (
        "not_run" if not train_steps_by_scenario
        else (
            "matched"
            if all(len(values) == 1 for values in train_steps_by_scenario.values())
            else "mismatch"
        )
    )
    offpolicy_status = (
        "complete" if set(OFFPOLICY_MODES) <= offpolicy_modes
        else ("partial" if offpolicy_modes else "not_run")
    )
    mixed_algorithms = bool(ppo_modes and offpolicy_modes)
    parameter_status = (
        "controlled_by_algorithm_family"
        if mixed_algorithms and ppo_parameter_status == "matched"
        else ppo_parameter_status
    )
    trainer_status = (
        "controlled_by_algorithm_family"
        if mixed_algorithms and ppo_trainer_status == "matched"
        else ppo_trainer_status
    )
    return {
        "ppo_parameter_budget_status": ppo_parameter_status,
        "ppo_trainer_budget_status": ppo_trainer_status,
        "environment_step_budget_status": environment_step_status,
        "sac_td3_status": offpolicy_status,
        "parameter_budget_status": parameter_status,
        "trainer_budget_status": trainer_status,
    }


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
    offpolicy_hidden_dim: int = 64,
    offpolicy_replay_capacity: int = 100_000,
    offpolicy_warmup_steps: int = 256,
    offpolicy_batch_size: int = 64,
    offpolicy_updates_per_step: int = 1,
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
    scenario_rank = {scenario: idx for idx, scenario in enumerate(scenarios)}
    for scenario, mode in pairs:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if mode not in ALL_POLICY_MODES:
            raise ValueError(f"unknown policy_mode: {mode}")
        start = time.perf_counter()
        run_seed = int(optimizer_seed) + 1009 * int(scenario_rank[scenario])
        if mode in POLICY_MODES:
            payload, heldout_rows, model = train_ppo_actor_critic(
                train_seeds=train_seeds,
                eval_seeds=eval_seeds,
                steps=int(steps),
                assets=int(assets),
                scenario=scenario,
                iterations=int(iterations),
                seed=run_seed,
                policy_mode=mode,
                use_handcrafted_frequency_prior=False,
            )
        else:
            payload, heldout_rows, model = train_flat_offpolicy_baseline(
                policy_mode=mode,
                train_seeds=train_seeds,
                eval_seeds=eval_seeds,
                steps=int(steps),
                assets=int(assets),
                scenario=scenario,
                iterations=int(iterations),
                seed=run_seed,
                hidden_dim=int(offpolicy_hidden_dim),
                replay_capacity=int(offpolicy_replay_capacity),
                warmup_steps=int(offpolicy_warmup_steps),
                batch_size=int(offpolicy_batch_size),
                updates_per_step=int(offpolicy_updates_per_step),
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
            "trainer": payload["trainer"],
            "optimizer_seed": run_seed,
            "gradient_updates_train": int(payload.get("gradient_updates_train", 0)),
            "actor_optimizer_steps_train": int(payload.get("actor_optimizer_steps_train", 0)),
            "critic_optimizer_steps_train": int(payload.get("critic_optimizer_steps_train", 0)),
            "temperature_optimizer_steps_train": int(
                payload.get("temperature_optimizer_steps_train", 0)
            ),
            "best_score": float(payload.get("best_score", 0.0)),
            "sharpe_mean": float(payload["summary"].get("sharpe_mean", 0.0)),
            "episode_information_ratio_mean": float(
                payload["summary"].get("episode_information_ratio_mean", 0.0)
            ),
            "total_return_mean": float(payload["summary"].get("total_return_mean", 0.0)),
            "FocusScore_mean": float(payload["summary"].get("FocusScore_mean", 0.0)),
            "LowerLFDrift_mean": float(payload["summary"].get("LowerLFDrift_mean", 0.0)),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        })
        parameter_budget.append(_parameter_budget_row(
            model,
            scenario=scenario,
            mode=mode,
            parameter_count=params,
            shard_index=int(shard_index),
            num_shards=int(num_shards),
        ))
        train_steps = int(payload.get(
            "environment_steps_train",
            len(train_seeds) * steps * max(1, int(iterations)),
        ))
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
            "gradient_updates_train": int(payload.get("gradient_updates_train", 0)),
            "actor_optimizer_steps_train": int(payload.get("actor_optimizer_steps_train", 0)),
            "critic_optimizer_steps_train": int(payload.get("critic_optimizer_steps_train", 0)),
            "temperature_optimizer_steps_train": int(
                payload.get("temperature_optimizer_steps_train", 0)
            ),
            "algorithm_family": (
                "on_policy_smdp_ppo" if mode in POLICY_MODES
                else f"off_policy_{model.config.algorithm}"
            ),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        })
    checks = build_paired_checks(rows, min_pairs=int(min_pairs))
    ppo_controls = tuple(mode for mode in POLICY_MODES if mode != "freq_hrl")
    offpolicy_controls = tuple(OFFPOLICY_MODES)
    metric_status = _metric_evidence_status(
        checks, ppo_controls, metrics=("total_return",)
    )
    offpolicy_metric_status = _metric_evidence_status(
        checks, offpolicy_controls, metrics=("total_return",)
    )
    all_metric_status = _metric_evidence_status(
        checks,
        ppo_controls + offpolicy_controls,
        metrics=("total_return",),
    )
    risk_adjusted_status = _metric_evidence_status(
        checks,
        ppo_controls + offpolicy_controls,
        metrics=("episode_information_ratio",),
    )
    responsibility_status = _metric_evidence_status(
        checks,
        ppo_controls + offpolicy_controls,
        metrics=("FocusScore", "LowerLFDrift"),
    )
    budgets = _budget_statuses(rows, parameter_budget, sample_efficiency)
    ppo_modes_run = {
        str(row.get("policy_mode", "")) for row in rows
        if str(row.get("policy_mode", "")) in POLICY_MODES
    }
    ppo_baseline_status = (
        metric_status
        if set(POLICY_MODES) <= ppo_modes_run
        and budgets["ppo_parameter_budget_status"] == "matched"
        and budgets["ppo_trainer_budget_status"] == "matched"
        else "partial_run_or_budget_mismatch"
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
            optimizer_seed=int(optimizer_seed),
            offpolicy_hidden_dim=int(offpolicy_hidden_dim),
            offpolicy_replay_capacity=int(offpolicy_replay_capacity),
            offpolicy_warmup_steps=int(offpolicy_warmup_steps),
            offpolicy_batch_size=int(offpolicy_batch_size),
            offpolicy_updates_per_step=int(offpolicy_updates_per_step),
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
            "offpolicy_metric_status": offpolicy_metric_status,
            "strong_learned_baseline_evidence_status": all_metric_status,
            "risk_adjusted_evidence_status": risk_adjusted_status,
            "responsibility_evidence_status": responsibility_status,
            **budgets,
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        },
        "boundary": (
            "PPO-family baselines use identical SMDP model dimensions, "
            "optimizer settings, initialization seeds, and environment-step "
            "budgets. Factorized flat PPO removes temporal abstraction and frequency "
            "routing; generic HRL retains temporal abstraction but uses raw "
            "features. Flat SAC/TD3 use raw observations and a single joint "
            "target/execution action. Cross-algorithm fairness is enforced by "
            "paired held-out seeds, equal environment-step budgets, the same "
            "environment/costs, and trading_metrics_v2; parameter equality is "
            "claimed only inside the PPO family."
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
    ppo_controls = tuple(mode for mode in POLICY_MODES if mode != "freq_hrl")
    offpolicy_controls = tuple(OFFPOLICY_MODES)
    metric_status = _metric_evidence_status(
        checks, ppo_controls, metrics=("total_return",)
    )
    offpolicy_metric_status = _metric_evidence_status(
        checks, offpolicy_controls, metrics=("total_return",)
    )
    all_metric_status = _metric_evidence_status(
        checks,
        ppo_controls + offpolicy_controls,
        metrics=("total_return",),
    )
    risk_adjusted_status = _metric_evidence_status(
        checks,
        ppo_controls + offpolicy_controls,
        metrics=("episode_information_ratio",),
    )
    responsibility_status = _metric_evidence_status(
        checks,
        ppo_controls + offpolicy_controls,
        metrics=("FocusScore", "LowerLFDrift"),
    )
    scenarios = sorted({str(row.get("scenario", "")) for row in rows if str(row.get("scenario", ""))})
    policy_modes = sorted({str(row.get("policy_mode", row.get("baseline", ""))) for row in rows if str(row.get("policy_mode", row.get("baseline", "")))})
    budgets = _budget_statuses(rows, parameter_budget, sample_efficiency)
    ppo_modes_run = {
        mode for mode in policy_modes if mode in POLICY_MODES
    }
    ppo_baseline_status = (
        metric_status
        if set(POLICY_MODES) <= ppo_modes_run
        and budgets["ppo_parameter_budget_status"] == "matched"
        and budgets["ppo_trainer_budget_status"] == "matched"
        else "partial_run_or_budget_mismatch"
    )
    eval_seeds = {
        str(row.get("seed", "")) for row in rows
        if str(row.get("seed", "")).strip()
    }
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
            "eval_seed_count": len(eval_seeds),
            "ppo_strong_baseline_status": ppo_baseline_status,
            "ppo_metric_status": metric_status,
            "offpolicy_metric_status": offpolicy_metric_status,
            "strong_learned_baseline_evidence_status": all_metric_status,
            "risk_adjusted_evidence_status": risk_adjusted_status,
            "responsibility_evidence_status": responsibility_status,
            **budgets,
            "merge_status": "merged",
        },
        "boundary": (
            "Merged learned-baseline shards. PPO comparisons are exact-capacity "
            "matched. SAC/TD3 use their native twin-Q architectures under the "
            "same paired evaluation and environment-step budget; cross-family "
            "parameter equality is not claimed."
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
        f"- all learned-baseline evidence: "
        f"`{payload['summary']['strong_learned_baseline_evidence_status']}`",
        f"- risk-adjusted evidence: "
        f"`{payload['summary']['risk_adjusted_evidence_status']}`",
        f"- responsibility evidence: "
        f"`{payload['summary']['responsibility_evidence_status']}`",
        f"- parameter budget: `{payload['summary']['parameter_budget_status']}`",
        f"- environment-step budget: "
        f"`{payload['summary']['environment_step_budget_status']}`",
        f"- scenarios: `{payload['summary']['scenario_count']}`",
        f"- eval seeds: `{payload['summary']['eval_seed_count']}`",
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate | Holm p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["paired_checks"]:
        delta = float(row["delta_mean"])
        ci_low = float(row["delta_ci95_low"])
        ci_high = float(row["delta_ci95_high"])
        win_rate = float(row["win_rate"])
        holm_p = float(row["holm_adjusted_p_value"])
        lines.append(
            f"| {row['check']} | {row['status']} | {row['metric']} "
            f"| {row['n_common']} | {delta:+.4f} "
            f"| {ci_low:+.4f} | {ci_high:+.4f} "
            f"| {win_rate:.2f} | {holm_p:.4f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--policy-modes", nargs="+", choices=ALL_POLICY_MODES, default=list(DEFAULT_POLICY_MODES))
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[31415, 27182, 16180, 14142, 17320, 22360, 24494, 26457, 28284, 31622],
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--offpolicy-hidden-dim", type=int, default=64)
    parser.add_argument("--offpolicy-replay-capacity", type=int, default=100_000)
    parser.add_argument("--offpolicy-warmup-steps", type=int, default=256)
    parser.add_argument("--offpolicy-batch-size", type=int, default=64)
    parser.add_argument("--offpolicy-updates-per-step", type=int, default=1)
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
            offpolicy_hidden_dim=int(args.offpolicy_hidden_dim),
            offpolicy_replay_capacity=int(args.offpolicy_replay_capacity),
            offpolicy_warmup_steps=int(args.offpolicy_warmup_steps),
            offpolicy_batch_size=int(args.offpolicy_batch_size),
            offpolicy_updates_per_step=int(args.offpolicy_updates_per_step),
            shard_index=int(args.shard_index),
            num_shards=int(args.num_shards),
        )
    write_outputs(args.output_dir, payload)
    print(
        "strong_learned_baseline_validation "
        f"status={payload['summary']['strong_learned_baseline_evidence_status']} "
        f"rows={payload['summary']['rows']} "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
