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
import hashlib
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
from freq_hrl.experiments.reproducibility import (
    is_hex_digest,
    validate_evaluation_seed_roles,
    validate_unique_seeds,
    verify_current_freq_hrl_source_identity,
)
from freq_hrl.rl import summarize_numeric_rows

from .performance_validation import SCENARIOS
from .metrics import (
    DEFAULT_TRAINING_REWARD_SCALE,
    METRIC_CONTRACT_VERSION,
    SELECTION_OBJECTIVE_VERSION,
    validation_utility,
)
from .offpolicy_baseline_validation import (
    OFFPOLICY_MODES,
    train_flat_offpolicy_baseline,
)
from .ppo_actor_critic import (
    FLAT_PPO_MODES,
    LEARNED_BASELINE_IMPLEMENTATION_VERSION,
    POLICY_MODES,
    train_ppo_actor_critic,
)


DEFAULT_SCENARIOS = (
    "stationary_low_noise",
    "persistent_shift",
    "stationary_high_noise",
    "localized_burst",
    "ood_period",
)
DEFAULT_POLICY_MODES = (
    "freq_hrl",
    "flat_ppo",
    "flat_gru_ppo",
    "generic_hrl_ppo",
    "generic_hrl_gru_ppo",
    "flat_sac",
    "flat_td3",
)
CONFIRMATORY_CONTROLS = tuple(
    mode for mode in DEFAULT_POLICY_MODES if mode != "freq_hrl"
)
DEFAULT_OPTIMIZER_SEEDS = (
    2026,
    2039,
    2053,
    2063,
    2081,
    2089,
    2099,
    2111,
    2129,
    2141,
)
DEFAULT_ROLLOUT_SEED_ROOTS = (42, 123, 456, 789, 2026)
DEFAULT_VALIDATION_SEEDS = (57721, 57727, 57731, 57737, 57751)
DEFAULT_EVAL_SEEDS = (
    31415,
    27182,
    16180,
    14142,
    17320,
    22360,
    24494,
    26457,
    28284,
    31622,
)
ALL_POLICY_MODES = POLICY_MODES + OFFPOLICY_MODES
MAIN_METRICS = (
    ("total_return", False),
    ("episode_information_ratio", False),
    ("FocusScore", False),
    ("LowerLFDrift", True),
)
CONTRACT_GATED_METRICS = {"episode_information_ratio", "total_return"}
CONFIRMATORY_ANALYSIS_VERSION = "strong_learned_confirmatory_analysis_v3"
PRACTICAL_EFFECT_THRESHOLDS = {
    "total_return": 0.005,
    "episode_information_ratio": 0.25,
    "FocusScore": 0.02,
    "LowerLFDrift": 0.05,
}
PRACTICAL_EFFECT_THRESHOLD_SOURCE = (
    "scale_calibrated_on_nested_validation_before_heldout_v1"
)
TRAINING_PATH_PROTOCOL = "fresh_deterministic_path_per_root_and_iteration_v2"
SELECTION_PROTOCOL = "disjoint_validation_paths"
LEARNING_GATE_MIN_FRACTION = 0.80


def validate_confirmatory_hyperparameter_metadata(
    *,
    confirmatory: bool,
    hyperparameter_source: str,
    frozen_config_sha256: str,
    selected_candidate_id: str,
    frozen_candidate_parameters_sha256: str,
) -> str:
    if not confirmatory:
        return "exploratory_unfrozen"
    digest = str(frozen_config_sha256).lower()
    if hyperparameter_source != "frozen_nested_validation":
        raise ValueError("confirmatory runs require frozen_nested_validation")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("confirmatory runs require a valid frozen config SHA-256")
    if not str(selected_candidate_id).strip():
        raise ValueError("confirmatory runs require a selected HPO candidate")
    parameter_digest = str(frozen_candidate_parameters_sha256).lower()
    if len(parameter_digest) != 64 or any(
        char not in "0123456789abcdef" for char in parameter_digest
    ):
        raise ValueError("confirmatory runs require a frozen candidate parameter hash")
    return "frozen_validation_only"


def canonical_hyperparameter_sha256(parameters: dict[str, Any]) -> str:
    canonical = json.dumps(
        parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def policy_hyperparameters(
    mode: str,
    *,
    ppo_hidden_dim: int,
    ppo_learning_rate: float,
    ppo_epochs: int,
    ppo_minibatch_size: int,
    ppo_init_log_std: float,
    training_reward_scale: float,
    offpolicy_hidden_dim: int,
    offpolicy_learning_rate: float,
    offpolicy_replay_capacity: int,
    offpolicy_warmup_steps: int,
    offpolicy_batch_size: int,
    offpolicy_updates_per_step: int,
) -> dict[str, Any]:
    if mode in POLICY_MODES:
        return {
            "hidden_dim": int(ppo_hidden_dim),
            "learning_rate": float(ppo_learning_rate),
            "epochs": int(ppo_epochs),
            "minibatch_size": int(ppo_minibatch_size),
            "init_log_std": float(ppo_init_log_std),
            "reward_scale": float(training_reward_scale),
        }
    return {
        "hidden_dim": int(offpolicy_hidden_dim),
        "learning_rate": float(offpolicy_learning_rate),
        "replay_capacity": int(offpolicy_replay_capacity),
        "warmup_steps": int(offpolicy_warmup_steps),
        "batch_size": int(offpolicy_batch_size),
        "updates_per_step": int(offpolicy_updates_per_step),
        "reward_scale": float(training_reward_scale),
    }


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
    training_replicate_seed: int,
    optimizer_seed: int,
    parameter_count: int,
    shard_index: int,
    num_shards: int,
    training_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = model.config
    training_payload = dict(training_payload or {})
    common = {
        "scenario": scenario,
        "policy_mode": mode,
        "training_replicate_seed": int(training_replicate_seed),
        "optimizer_seed": int(optimizer_seed),
        "parameter_count": int(parameter_count),
        "hidden_dim": int(config.hidden_dim),
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
    }
    if mode in FLAT_PPO_MODES:
        return {
            **common,
            "requested_hidden_dim": int(
                training_payload.get("requested_hidden_dim", config.hidden_dim)
            ),
            "effective_hidden_dim": int(config.hidden_dim),
            "algorithm_family": (
                "on_policy_joint_flat_gru_ppo"
                if mode == "flat_gru_ppo" else "on_policy_joint_flat_ppo"
            ),
            "state_dim": int(config.state_dim),
            "action_dim": int(config.action_dim),
            "upper_state_dim": "",
            "lower_state_dim": "",
            "upper_action_dim": "",
            "lower_action_dim": "",
            "capacity_ratio": float(training_payload.get("capacity_ratio", 1.0)),
            "capacity_match_status": str(
                training_payload.get("capacity_match_status", "unknown")
            ),
            "matched_budget_group": "trading_capacity_matched_ppo_v5",
            "capacity_contract": (
                "active parameter count within 5% and equal HPO search budget; "
                "canonical single-value joint-action PPO with the registered "
                "raw-history encoder"
            ),
        }
    if mode in POLICY_MODES:
        return {
            **common,
            "requested_hidden_dim": int(
                training_payload.get("requested_hidden_dim", config.hidden_dim)
            ),
            "effective_hidden_dim": int(config.hidden_dim),
            "algorithm_family": "on_policy_smdp_ppo",
            "state_dim": "",
            "action_dim": "",
            "upper_state_dim": int(config.upper_state_dim),
            "lower_state_dim": int(config.lower_state_dim),
            "upper_action_dim": int(config.upper_action_dim),
            "lower_action_dim": int(config.lower_action_dim),
            "capacity_ratio": float(training_payload.get("capacity_ratio", 1.0)),
            "capacity_match_status": str(
                training_payload.get("capacity_match_status", "unknown")
            ),
            "matched_budget_group": "trading_capacity_matched_ppo_v5",
            "capacity_contract": (
                "active parameter count within 5% of Freq-HRL and equal HPO "
                "search budget; raw generic HRL uses the complete causal window"
            ),
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


def selected_experiment_cells(
    scenarios: list[str],
    policy_modes: list[str],
    optimizer_seeds: list[int],
    *,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[tuple[str, str, int]]:
    if len(set(int(seed) for seed in optimizer_seeds)) != len(optimizer_seeds):
        raise ValueError("optimizer_seeds must contain unique training replicates")
    cells = [
        (scenario, mode, int(seed))
        for scenario in scenarios
        for mode in policy_modes
        for seed in optimizer_seeds
    ]
    shards = max(1, int(num_shards))
    index = int(shard_index)
    if index < 0 or index >= shards:
        raise ValueError(f"shard_index must be in [0, {shards - 1}], got {index}")
    return [cell for cell_index, cell in enumerate(cells) if cell_index % shards == index]


def scenario_optimizer_seed(training_replicate_seed: int, scenario: str) -> int:
    scenario_names = list(SCENARIOS)
    if scenario not in scenario_names:
        raise ValueError(f"unknown scenario: {scenario}")
    return int(training_replicate_seed) + 1009 * scenario_names.index(scenario)


def _experiment_matrix_coverage(
    rows: list[dict[str, Any]],
    *,
    expected_scenarios: list[str] | None = None,
    expected_policy_modes: list[str] | None = None,
    expected_replicate_seeds: list[int] | None = None,
    expected_eval_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Audit the observed scenario/policy/replicate/evaluation Cartesian grid."""

    expected_dimensions_declared = any(
        value is not None
        for value in (
            expected_scenarios,
            expected_policy_modes,
            expected_replicate_seeds,
            expected_eval_seeds,
        )
    )
    if not rows and not expected_dimensions_declared:
        return {
            "matrix_coverage_status": "not_run",
            "expected_evaluation_row_count": 0,
            "observed_evaluation_row_count": 0,
            "missing_evaluation_row_count": 0,
            "duplicate_evaluation_row_count": 0,
        }
    keys: list[tuple[str, str, str, str]] = []
    invalid_rows = 0
    for row in rows:
        key = (
            str(row.get("scenario", "")).strip(),
            str(row.get("policy_mode", row.get("baseline", ""))).strip(),
            str(row.get("training_replicate_seed", "")).strip(),
            str(row.get("seed", "")).strip(),
        )
        if not all(key):
            invalid_rows += 1
            continue
        keys.append(key)
    unique_keys = set(keys)
    scenarios = (
        {str(value) for value in expected_scenarios}
        if expected_scenarios is not None
        else {key[0] for key in unique_keys}
    )
    modes = (
        {str(value) for value in expected_policy_modes}
        if expected_policy_modes is not None
        else {key[1] for key in unique_keys}
    )
    replicates = (
        {str(int(value)) for value in expected_replicate_seeds}
        if expected_replicate_seeds is not None
        else {key[2] for key in unique_keys}
    )
    eval_seeds = (
        {str(int(value)) for value in expected_eval_seeds}
        if expected_eval_seeds is not None
        else {key[3] for key in unique_keys}
    )
    expected_keys = {
        (scenario, mode, replicate, seed)
        for scenario in scenarios
        for mode in modes
        for replicate in replicates
        for seed in eval_seeds
    }
    missing_keys = expected_keys - unique_keys
    unexpected_keys = unique_keys - expected_keys
    expected = len(expected_keys)
    missing = len(missing_keys)
    duplicates = max(0, len(keys) - len(unique_keys))
    status = (
        "complete"
        if invalid_rows == 0
        and missing == 0
        and not unexpected_keys
        and duplicates == 0
        else "incomplete"
    )
    return {
        "matrix_coverage_status": status,
        "expected_evaluation_row_count": int(expected),
        "observed_evaluation_row_count": int(len(unique_keys)),
        "missing_evaluation_row_count": int(missing),
        "unexpected_evaluation_row_count": int(len(unexpected_keys)),
        "duplicate_evaluation_row_count": int(duplicates),
        "invalid_evaluation_row_count": int(invalid_rows),
        "coverage_design_source": (
            "preregistered_expected_grid"
            if expected_dimensions_declared else "inferred_observed_grid"
        ),
    }


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


def learning_dynamics_summary(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit whether selected checkpoints actually improved over initialization."""

    modes = sorted({str(row.get("policy_mode", "")) for row in run_rows if row.get("policy_mode")})
    by_mode: list[dict[str, Any]] = []
    for mode in modes:
        group = [row for row in run_rows if str(row.get("policy_mode", "")) == mode]
        selected_iterations = [
            int(float(row.get("selected_checkpoint_iteration", -1))) for row in group
        ]
        gains = [float(row.get("validation_learning_gain", 0.0)) for row in group]
        trained_count = sum(iteration >= 0 for iteration in selected_iterations)
        fraction = float(trained_count / len(group)) if group else 0.0
        mean_gain = float(np.mean(gains)) if gains else 0.0
        by_mode.append({
            "policy_mode": mode,
            "cell_count": len(group),
            "trained_checkpoint_count": int(trained_count),
            "trained_checkpoint_fraction": fraction,
            "validation_learning_gain_mean": mean_gain,
            "status": (
                "supported"
                if fraction >= LEARNING_GATE_MIN_FRACTION and mean_gain > 0.0
                else "not_supported"
            ),
        })
    status = (
        "not_run"
        if not by_mode
        else "supported"
        if all(row["status"] == "supported" for row in by_mode)
        else "not_supported"
    )
    return {
        "learning_dynamics_status": status,
        "learning_gate_min_fraction": LEARNING_GATE_MIN_FRACTION,
        "learning_dynamics_by_policy": by_mode,
    }


def build_paired_checks(
    rows: list[dict[str, Any]],
    *,
    controls: tuple[str, ...] = CONFIRMATORY_CONTROLS,
    min_pairs: int = 10,
    include_scenario_strata: bool = True,
) -> list[dict[str, Any]]:
    if rows and any("training_replicate_seed" not in row for row in rows):
        raise ValueError(
            "strong learned-baseline rows must identify independent "
            "training_replicate_seed values; legacy eval-seed-only artifacts "
            "must be regenerated"
        )
    scenarios = sorted({
        str(row.get("scenario", ""))
        for row in rows
        if str(row.get("scenario", ""))
    })
    scopes: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("pooled_preregistered_stress", "all_preregistered", rows)
    ]
    if include_scenario_strata and len(scenarios) > 1:
        scopes.extend(
            (
                "scenario_stratum",
                scenario,
                [row for row in rows if str(row.get("scenario", "")) == scenario],
            )
            for scenario in scenarios
        )

    checks: list[dict[str, Any]] = []
    for inference_scope, scenario_label, scope_rows in scopes:
        for control in controls:
            for metric, lower_is_better in MAIN_METRICS:
                relevant = [
                    row for row in scope_rows
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
                training_protocol_valid = bool(
                    relevant
                    and {
                        str(row.get("training_path_protocol", "missing"))
                        for row in relevant
                    } == {TRAINING_PATH_PROTOCOL}
                    and {
                        str(row.get("checkpoint_selection_protocol", "missing"))
                        for row in relevant
                    } == {SELECTION_PROTOCOL}
                    and {
                        str(row.get("selection_objective_version", "missing"))
                        for row in relevant
                    } == {SELECTION_OBJECTIVE_VERSION}
                )
                stats = paired_delta_stats(
                    scope_rows,
                    variant_key="baseline",
                    pair_keys=("scenario", "training_replicate_seed", "seed"),
                    metric=metric,
                    treatment="freq_hrl",
                    control=control,
                    lower_is_better=lower_is_better,
                    cluster_keys=("training_replicate_seed",),
                )
                endpoint_class = (
                    "task_performance"
                    if metric == "total_return"
                    else "risk_adjusted"
                    if metric == "episode_information_ratio"
                    else "responsibility_separation"
                )
                checks.append({
                    "check": (
                        f"freq_hrl_vs_{control}_{metric}"
                        if inference_scope == "pooled_preregistered_stress"
                        else f"{scenario_label}__freq_hrl_vs_{control}_{metric}"
                    ),
                    **stats,
                    "inference_scope": inference_scope,
                    "scenario_stratum": scenario_label,
                    "metric_contract_valid": contract_valid,
                    "metric_contract_versions": contracts,
                    "training_protocol_valid": training_protocol_valid,
                    "practical_effect_threshold": float(
                        PRACTICAL_EFFECT_THRESHOLDS[metric]
                    ),
                    "practical_effect_threshold_source": (
                        PRACTICAL_EFFECT_THRESHOLD_SOURCE
                    ),
                    "multiplicity_family": (
                        f"strong_learned_{inference_scope}_{endpoint_class}"
                    ),
                    "baseline_class": (
                        "strong_learned_ppo"
                        if control in POLICY_MODES else "strong_learned_offpolicy"
                    ),
                    "confirmatory_analysis_version": CONFIRMATORY_ANALYSIS_VERSION,
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
        threshold = float(row.get("practical_effect_threshold", 0.0))
        if not bool(row.get("metric_contract_valid", False)):
            status = "invalid_legacy_metric_contract"
        elif not bool(row.get("training_protocol_valid", False)):
            status = "invalid_training_protocol"
        elif n_independent < int(min_pairs):
            status = "underpowered"
        elif (
            ci_low is not None
            and ci_low > threshold
            and bool(row.get("holm_reject", False))
        ):
            status = "supported"
        elif (
            ci_low is not None
            and ci_low > 0.0
            and bool(row.get("holm_reject", False))
        ):
            status = "statistically_supported_below_practical_threshold"
        elif improvement is not None and improvement > threshold:
            status = "positive_mixed"
        elif improvement is not None and improvement > 0.0:
            status = "positive_below_practical_threshold"
        elif improvement is not None and improvement <= 0.0:
            status = "not_supported"
        else:
            status = "inconclusive"
        row["status"] = status
        row["confirmatory_gate"] = (
            "min independent training replicates + cluster-bootstrap CI above "
            "the preregistered practical threshold + Holm-adjusted two-sided sign test"
        )
    return corrected


def _metric_evidence_status(
    checks: list[dict[str, Any]],
    controls: tuple[str, ...],
    *,
    metrics: tuple[str, ...],
    inference_scope: str = "pooled_preregistered_stress",
    scenario_stratum: str | None = None,
) -> str:
    if not controls:
        return "not_run"
    evidence: dict[tuple[str, str], str] = {}
    for row in checks:
        if str(row.get("inference_scope", "")) != str(inference_scope):
            continue
        if (
            scenario_stratum is not None
            and str(row.get("scenario_stratum", "")) != str(scenario_stratum)
        ):
            continue
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


def stress_stratum_evidence(
    checks: list[dict[str, Any]],
    controls: tuple[str, ...],
    *,
    metrics: tuple[str, ...] = ("total_return",),
) -> dict[str, str]:
    scenarios = sorted({
        str(row.get("scenario_stratum", ""))
        for row in checks
        if str(row.get("inference_scope", "")) == "scenario_stratum"
        and str(row.get("scenario_stratum", ""))
    })
    return {
        scenario: _metric_evidence_status(
            checks,
            controls,
            metrics=metrics,
            inference_scope="scenario_stratum",
            scenario_stratum=scenario,
        )
        for scenario in scenarios
    }


def validate_frozen_result_binding(
    rows: list[dict[str, Any]],
    *,
    expected_frozen_config_sha256: str = "",
    expected_selected: dict[str, dict[str, Any]] | None = None,
    expected_code_revision: str = "",
    expected_source_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Bind merged confirmatory rows back to the validated frozen config."""

    frozen_rows = [
        row for row in rows
        if str(row.get("hyperparameter_source", "")) == "frozen_nested_validation"
    ]
    if not frozen_rows:
        return {"status": "not_applicable_exploratory", "verified_row_count": 0}
    selected = expected_selected or {}
    digest = str(expected_frozen_config_sha256).strip().lower()
    revision = str(expected_code_revision).strip().lower()
    manifest = str(expected_source_manifest_sha256).strip().lower()
    if (
        not is_hex_digest(digest, length=64)
        or not selected
        or not is_hex_digest(revision, length=40)
        or not is_hex_digest(manifest, length=64)
    ):
        return {
            "status": "unverified_missing_frozen_config",
            "verified_row_count": 0,
        }

    for row in frozen_rows:
        mode = str(row.get("policy_mode", row.get("baseline", "")))
        entry = selected.get(mode)
        if not isinstance(entry, dict):
            raise ValueError(f"frozen config has no selected entry for {mode}")
        parameters = entry.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"frozen config has invalid parameters for {mode}")
        parameter_hash = canonical_hyperparameter_sha256(parameters)
        try:
            actual_parameters = json.loads(
                str(row.get("actual_hyperparameters_json", ""))
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid merged hyperparameters for {mode}") from exc
        expected_values = {
            "frozen_config_sha256": digest,
            "selected_candidate_id": str(entry.get("candidate_id", "")),
            "actual_hyperparameter_sha256": parameter_hash,
            "code_revision": revision,
            "source_manifest_sha256": manifest,
        }
        for field, expected in expected_values.items():
            actual = str(row.get(field, "")).strip().lower()
            if actual != str(expected).strip().lower():
                raise ValueError(
                    f"merged row disagrees with frozen config for {mode}: {field}"
                )
        if actual_parameters != parameters:
            raise ValueError(
                f"merged row parameters disagree with frozen config for {mode}"
            )
    return {
        "status": "verified",
        "verified_row_count": len(frozen_rows),
        "frozen_config_sha256": digest,
    }


def build_experiment_manifest(
    *,
    scenarios: list[str],
    policy_modes: list[str],
    train_seeds: list[int],
    validation_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    assets: int,
    iterations: int,
    optimizer_seed: int = 2026,
    optimizer_seeds: list[int] | None = None,
    min_pairs: int = 10,
    ppo_hidden_dim: int = 64,
    ppo_learning_rate: float = 3e-4,
    ppo_epochs: int = 4,
    ppo_minibatch_size: int = 512,
    ppo_init_log_std: float = -1.0,
    training_reward_scale: float = DEFAULT_TRAINING_REWARD_SCALE,
    offpolicy_hidden_dim: int = 64,
    offpolicy_learning_rate: float = 3e-4,
    offpolicy_replay_capacity: int = 100_000,
    offpolicy_warmup_steps: int = 256,
    offpolicy_batch_size: int = 64,
    offpolicy_updates_per_step: int = 1,
    confirmatory: bool = False,
    hyperparameter_source: str = "exploratory_unfrozen",
    frozen_config_sha256: str = "",
    selected_candidate_id: str = "",
    frozen_candidate_parameters_sha256: str = "",
    code_revision: str = "",
    source_manifest_sha256: str = "",
    source_identity_status: str = "unregistered_local",
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replicate_seeds = list(optimizer_seeds or [int(optimizer_seed)])
    cells = selected_experiment_cells(
        scenarios,
        policy_modes,
        replicate_seeds,
        shard_index=int(shard_index),
        num_shards=int(num_shards),
    )
    for scenario, mode, replicate_seed in cells:
        run_seed = scenario_optimizer_seed(replicate_seed, scenario)
        if mode in FLAT_PPO_MODES:
            trainer = "canonical_joint_flat_ppo_v1"
        elif mode in POLICY_MODES:
            trainer = "frequency_separated_smdp_ppo_v2"
        else:
            algorithm = "sac" if mode == "flat_sac" else "td3"
            trainer = f"flat_{algorithm}_twin_q_v1"
        output_dir = (
            "transit_hrl/results/strong_learned_baseline_validation_v2_cells/"
            f"{scenario}/{mode}/replicate_{int(replicate_seed)}"
        )
        rows.append({
            "scenario": scenario,
            "policy_mode": mode,
            "training_replicate_seed": int(replicate_seed),
            "train_seeds": " ".join(str(seed) for seed in train_seeds),
            "validation_seeds": " ".join(str(seed) for seed in validation_seeds),
            "eval_seeds": " ".join(str(seed) for seed in eval_seeds),
            "steps": int(steps),
            "assets": int(assets),
            "iterations": int(iterations),
            "trainer": trainer,
            "confirmatory": bool(confirmatory),
            "hyperparameter_source": str(hyperparameter_source),
            "frozen_config_sha256": str(frozen_config_sha256),
            "selected_candidate_id": str(selected_candidate_id),
            "frozen_candidate_parameters_sha256": str(
                frozen_candidate_parameters_sha256
            ),
            "code_revision": str(code_revision),
            "source_manifest_sha256": str(source_manifest_sha256),
            "source_identity_status": str(source_identity_status),
            "optimizer_seed": run_seed,
            "independent_unit": "training_replicate_seed",
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
            "command": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=transit_hrl python3 -m "
                "freq_hrl.experiments.trading.strong_learned_baseline_validation "
                f"--scenarios {scenario} --policy-modes {mode} "
                f"--steps {int(steps)} --assets {int(assets)} --iterations {int(iterations)} "
                f"--optimizer-seeds {int(replicate_seed)} "
                f"--min-pairs {int(min_pairs)} "
                "--train-seeds "
                + " ".join(str(seed) for seed in train_seeds)
                + " --validation-seeds "
                + " ".join(str(seed) for seed in validation_seeds)
                + " --eval-seeds "
                + " ".join(str(seed) for seed in eval_seeds)
                + f" --ppo-hidden-dim {int(ppo_hidden_dim)}"
                + f" --ppo-learning-rate {float(ppo_learning_rate)}"
                + f" --ppo-epochs {int(ppo_epochs)}"
                + f" --ppo-minibatch-size {int(ppo_minibatch_size)}"
                + f" --ppo-init-log-std {float(ppo_init_log_std)}"
                + f" --training-reward-scale {float(training_reward_scale)}"
                + f" --offpolicy-hidden-dim {int(offpolicy_hidden_dim)}"
                + f" --offpolicy-learning-rate {float(offpolicy_learning_rate)}"
                + f" --offpolicy-replay-capacity {int(offpolicy_replay_capacity)}"
                + f" --offpolicy-warmup-steps {int(offpolicy_warmup_steps)}"
                + f" --offpolicy-batch-size {int(offpolicy_batch_size)}"
                + f" --offpolicy-updates-per-step {int(offpolicy_updates_per_step)}"
                + (" --confirmatory" if confirmatory else "")
                + f" --hyperparameter-source {hyperparameter_source}"
                + f" --frozen-config-sha256 {frozen_config_sha256 or 'none'}"
                + f" --selected-candidate-id {selected_candidate_id or 'none'}"
                + " --frozen-candidate-parameters-sha256 "
                + (frozen_candidate_parameters_sha256 or "none")
                + f" --code-revision {code_revision or 'none'}"
                + f" --source-manifest-sha256 {source_manifest_sha256 or 'none'}"
                + f" --output-dir {output_dir}"
            ),
        })
    return rows


def _budget_statuses(
    rows: list[dict[str, Any]],
    parameter_budget: list[dict[str, Any]],
    sample_efficiency: list[dict[str, Any]],
) -> dict[str, Any]:
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
    ppo_capacity_ratios = [
        max(values) / max(min(values), 1)
        for values in ppo_counts_by_scenario.values()
        if values
    ]
    max_ppo_capacity_ratio = max(ppo_capacity_ratios, default=1.0)
    if not ppo_modes:
        ppo_parameter_status = "not_run"
    elif ppo_counts_by_scenario and all(
        len(values) == 1 for values in ppo_counts_by_scenario.values()
    ):
        ppo_parameter_status = "matched_exact"
    elif ppo_counts_by_scenario and max_ppo_capacity_ratio <= 1.05:
        ppo_parameter_status = "matched_within_5pct"
    else:
        ppo_parameter_status = "mismatch"

    ppo_trainers = {
        str(row.get("trainer", ""))
        for row in rows
        if str(row.get("policy_mode", row.get("baseline", ""))) in POLICY_MODES
        and str(row.get("trainer", "")).strip()
    }
    valid_ppo_trainers = {
        "canonical_joint_flat_ppo_v1",
        "frequency_separated_smdp_ppo_v2",
    }
    if not ppo_modes:
        ppo_trainer_status = "not_run"
    elif len(ppo_trainers) == 1:
        ppo_trainer_status = "matched_exact"
    elif ppo_trainers and ppo_trainers <= valid_ppo_trainers:
        ppo_trainer_status = "controlled_by_ppo_family"
    else:
        ppo_trainer_status = "mismatch"

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
        if mixed_algorithms and ppo_parameter_status in {
            "matched_exact", "matched_within_5pct"
        }
        else ppo_parameter_status
    )
    trainer_status = (
        "controlled_by_algorithm_family"
        if mixed_algorithms and ppo_trainer_status in {
            "matched_exact", "controlled_by_ppo_family"
        }
        else ppo_trainer_status
    )
    return {
        "ppo_parameter_budget_status": ppo_parameter_status,
        "ppo_max_parameter_ratio": float(max_ppo_capacity_ratio),
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
    optimizer_seeds: list[int] | None = None,
    validation_seeds: list[int] | None = None,
    ppo_hidden_dim: int = 64,
    ppo_learning_rate: float = 3e-4,
    ppo_epochs: int = 4,
    ppo_minibatch_size: int = 512,
    ppo_init_log_std: float = -1.0,
    training_reward_scale: float = DEFAULT_TRAINING_REWARD_SCALE,
    offpolicy_hidden_dim: int = 64,
    offpolicy_learning_rate: float = 3e-4,
    offpolicy_replay_capacity: int = 100_000,
    offpolicy_warmup_steps: int = 256,
    offpolicy_batch_size: int = 64,
    offpolicy_updates_per_step: int = 1,
    confirmatory: bool = False,
    hyperparameter_source: str = "exploratory_unfrozen",
    frozen_config_sha256: str = "",
    selected_candidate_id: str = "",
    frozen_candidate_parameters_sha256: str = "",
    code_revision: str = "",
    expected_source_manifest_sha256: str = "",
    shard_index: int = 0,
    num_shards: int = 1,
) -> dict[str, Any]:
    hyperparameter_protocol_status = validate_confirmatory_hyperparameter_metadata(
        confirmatory=bool(confirmatory),
        hyperparameter_source=str(hyperparameter_source),
        frozen_config_sha256=str(frozen_config_sha256),
        selected_candidate_id=str(selected_candidate_id),
        frozen_candidate_parameters_sha256=str(
            frozen_candidate_parameters_sha256
        ),
    )
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=str(code_revision),
        expected_source_manifest_sha256=str(expected_source_manifest_sha256),
        require_verified=bool(confirmatory),
    )
    source_revision = source_identity["code_revision"]
    source_manifest = source_identity["source_manifest_sha256"]
    source_identity_status = source_identity["source_identity_status"]
    if confirmatory and len(policy_modes) != 1:
        raise ValueError(
            "confirmatory cells must contain exactly one policy-specific frozen candidate"
        )
    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    parameter_budget: list[dict[str, Any]] = []
    sample_efficiency: list[dict[str, Any]] = []
    checkpoint_payloads: list[dict[str, Any]] = []
    rollout_seed_roots = validate_unique_seeds(
        train_seeds, role="rollout_seed_roots"
    )
    validation_seed_list = validate_unique_seeds(
        validation_seeds or DEFAULT_VALIDATION_SEEDS,
        role="validation_seeds",
    )
    validation_seed_list, heldout_test_seeds = validate_evaluation_seed_roles(
        validation_seed_list, eval_seeds
    )
    replicate_seeds = list(optimizer_seeds or [int(optimizer_seed)])
    cells = selected_experiment_cells(
        scenarios,
        policy_modes,
        replicate_seeds,
        shard_index=int(shard_index),
        num_shards=int(num_shards),
    )
    for scenario, mode, replicate_seed in cells:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")
        if mode not in ALL_POLICY_MODES:
            raise ValueError(f"unknown policy_mode: {mode}")
        actual_hyperparameters = policy_hyperparameters(
            mode,
            ppo_hidden_dim=int(ppo_hidden_dim),
            ppo_learning_rate=float(ppo_learning_rate),
            ppo_epochs=int(ppo_epochs),
            ppo_minibatch_size=int(ppo_minibatch_size),
            ppo_init_log_std=float(ppo_init_log_std),
            training_reward_scale=float(training_reward_scale),
            offpolicy_hidden_dim=int(offpolicy_hidden_dim),
            offpolicy_learning_rate=float(offpolicy_learning_rate),
            offpolicy_replay_capacity=int(offpolicy_replay_capacity),
            offpolicy_warmup_steps=int(offpolicy_warmup_steps),
            offpolicy_batch_size=int(offpolicy_batch_size),
            offpolicy_updates_per_step=int(offpolicy_updates_per_step),
        )
        actual_hyperparameter_sha256 = canonical_hyperparameter_sha256(
            actual_hyperparameters
        )
        if confirmatory and actual_hyperparameter_sha256 != str(
            frozen_candidate_parameters_sha256
        ).lower():
            raise ValueError(
                "actual training parameters do not match the frozen candidate"
            )
        start = time.perf_counter()
        run_seed = scenario_optimizer_seed(replicate_seed, scenario)
        if mode in POLICY_MODES:
            payload, heldout_rows, model = train_ppo_actor_critic(
                train_seeds=train_seeds,
                validation_seeds=validation_seed_list,
                eval_seeds=heldout_test_seeds,
                steps=int(steps),
                assets=int(assets),
                scenario=scenario,
                iterations=int(iterations),
                seed=run_seed,
                hidden_dim=int(ppo_hidden_dim),
                learning_rate=float(ppo_learning_rate),
                ppo_epochs=int(ppo_epochs),
                minibatch_size=int(ppo_minibatch_size),
                init_log_std=float(ppo_init_log_std),
                resample_training_paths=True,
                policy_mode=mode,
                use_handcrafted_frequency_prior=False,
                reward_scale=float(training_reward_scale),
            )
        else:
            payload, heldout_rows, model = train_flat_offpolicy_baseline(
                policy_mode=mode,
                train_seeds=train_seeds,
                validation_seeds=validation_seed_list,
                eval_seeds=heldout_test_seeds,
                steps=int(steps),
                assets=int(assets),
                scenario=scenario,
                iterations=int(iterations),
                seed=run_seed,
                hidden_dim=int(offpolicy_hidden_dim),
                learning_rate=float(offpolicy_learning_rate),
                replay_capacity=int(offpolicy_replay_capacity),
                warmup_steps=int(offpolicy_warmup_steps),
                batch_size=int(offpolicy_batch_size),
                updates_per_step=int(offpolicy_updates_per_step),
                resample_training_paths=True,
                reward_scale=float(training_reward_scale),
            )
        if (
            confirmatory
            and mode in POLICY_MODES
            and str(payload.get("capacity_match_status", ""))
            != "matched_within_5pct"
        ):
            raise ValueError(
                f"confirmatory PPO capacity mismatch for {mode}: "
                f"ratio={payload.get('capacity_ratio', 'missing')}"
            )
        payload["confirmatory"] = bool(confirmatory)
        payload["hyperparameter_source"] = str(hyperparameter_source)
        payload["hyperparameter_protocol_status"] = hyperparameter_protocol_status
        payload["frozen_config_sha256"] = str(frozen_config_sha256)
        payload["selected_candidate_id"] = str(selected_candidate_id)
        payload["actual_hyperparameters"] = dict(actual_hyperparameters)
        payload["actual_hyperparameter_sha256"] = actual_hyperparameter_sha256
        payload["learned_baseline_implementation_version"] = (
            LEARNED_BASELINE_IMPLEMENTATION_VERSION
        )
        payload["code_revision"] = source_revision
        payload["source_manifest_sha256"] = source_manifest
        payload["source_identity_status"] = source_identity_status
        elapsed = float(time.perf_counter() - start)
        params = count_parameters(model)
        checkpoint_file = (
            f"{scenario}__{mode}__replicate_{int(replicate_seed)}.pt"
        )
        checkpoint_payloads.append({
            "checkpoint_file": checkpoint_file,
            "payload": {
                "model_state_dict": model.state_dict(),
                "model_config": payload.get("config", {}),
                "trainer": payload.get("trainer", ""),
                "scenario": scenario,
                "policy_mode": mode,
                "training_replicate_seed": int(replicate_seed),
                "optimizer_seed": int(run_seed),
                "rollout_seed_roots": list(rollout_seed_roots),
                "validation_seeds": list(validation_seed_list),
                "heldout_test_seeds": list(heldout_test_seeds),
                "metric_contract_version": METRIC_CONTRACT_VERSION,
                "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
                "hyperparameter_source": str(hyperparameter_source),
                "hyperparameter_protocol_status": hyperparameter_protocol_status,
                "frozen_config_sha256": str(frozen_config_sha256),
                "selected_candidate_id": str(selected_candidate_id),
                "actual_hyperparameters": dict(actual_hyperparameters),
                "actual_hyperparameter_sha256": actual_hyperparameter_sha256,
                "learned_baseline_implementation_version": (
                    LEARNED_BASELINE_IMPLEMENTATION_VERSION
                ),
                "code_revision": source_revision,
                "source_manifest_sha256": source_manifest,
                "source_identity_status": source_identity_status,
                "selected_checkpoint_iteration": int(
                    payload.get("selected_checkpoint_iteration", -1)
                ),
                "initial_validation_score": float(
                    payload.get("initial_validation_score", 0.0)
                ),
                "validation_learning_gain": float(
                    payload.get("validation_learning_gain", 0.0)
                ),
            },
        })
        for row in heldout_rows:
            item = dict(row)
            item["scenario"] = scenario
            item["baseline"] = mode
            item["policy_mode"] = mode
            item["training_replicate_seed"] = int(replicate_seed)
            item["optimizer_seed"] = int(run_seed)
            item["independent_unit"] = "training_replicate_seed"
            item["trainer"] = payload["trainer"]
            item["training_path_protocol"] = str(
                payload.get("training_path_protocol", "missing")
            )
            item["checkpoint_selection_protocol"] = str(
                payload.get("checkpoint_selection_protocol", "missing")
            )
            item["selection_objective_version"] = str(
                payload.get("selection_objective_version", "missing")
            )
            item["confirmatory"] = bool(confirmatory)
            item["hyperparameter_source"] = str(hyperparameter_source)
            item["hyperparameter_protocol_status"] = hyperparameter_protocol_status
            item["frozen_config_sha256"] = str(frozen_config_sha256)
            item["selected_candidate_id"] = str(selected_candidate_id)
            item["actual_hyperparameters_json"] = json.dumps(
                actual_hyperparameters, sort_keys=True, separators=(",", ":")
            )
            item["actual_hyperparameter_sha256"] = actual_hyperparameter_sha256
            item["learned_baseline_implementation_version"] = (
                LEARNED_BASELINE_IMPLEMENTATION_VERSION
            )
            item["code_revision"] = source_revision
            item["source_manifest_sha256"] = source_manifest
            item["source_identity_status"] = source_identity_status
            item["rollout_seed_roots"] = " ".join(
                str(seed) for seed in rollout_seed_roots
            )
            item["validation_seeds"] = " ".join(
                str(seed) for seed in validation_seed_list
            )
            item["source_artifact"] = "strong_learned_baseline_validation"
            item["shard_index"] = int(shard_index)
            item["num_shards"] = int(num_shards)
            item["checkpoint_file"] = checkpoint_file
            rows.append(item)
        run_rows.append({
            "scenario": scenario,
            "policy_mode": mode,
            "training_replicate_seed": int(replicate_seed),
            "elapsed_sec": elapsed,
            "train_seed_count": len(train_seeds),
            "validation_seed_count": len(validation_seed_list),
            "eval_seed_count": len(heldout_test_seeds),
            "steps": int(steps),
            "iterations": int(iterations),
            "parameter_count": params,
            "trainer": payload["trainer"],
            "training_path_protocol": str(payload.get("training_path_protocol", "")),
            "checkpoint_selection_protocol": str(
                payload.get("checkpoint_selection_protocol", "")
            ),
            "selection_objective_version": str(
                payload.get("selection_objective_version", "")
            ),
            "confirmatory": bool(confirmatory),
            "hyperparameter_source": str(hyperparameter_source),
            "hyperparameter_protocol_status": hyperparameter_protocol_status,
            "frozen_config_sha256": str(frozen_config_sha256),
            "selected_candidate_id": str(selected_candidate_id),
            "actual_hyperparameters_json": json.dumps(
                actual_hyperparameters, sort_keys=True, separators=(",", ":")
            ),
            "actual_hyperparameter_sha256": actual_hyperparameter_sha256,
            "learned_baseline_implementation_version": (
                LEARNED_BASELINE_IMPLEMENTATION_VERSION
            ),
            "code_revision": source_revision,
            "source_manifest_sha256": source_manifest,
            "source_identity_status": source_identity_status,
            "optimizer_seed": run_seed,
            "gradient_updates_train": int(payload.get("gradient_updates_train", 0)),
            "environment_steps_validation": int(
                payload.get("environment_steps_validation", 0)
            ),
            "unique_training_path_count": int(
                payload.get("unique_training_path_count", 0)
            ),
            "checkpoint_file": checkpoint_file,
            "actor_optimizer_steps_train": int(payload.get("actor_optimizer_steps_train", 0)),
            "critic_optimizer_steps_train": int(payload.get("critic_optimizer_steps_train", 0)),
            "temperature_optimizer_steps_train": int(
                payload.get("temperature_optimizer_steps_train", 0)
            ),
            "best_score": float(payload.get("best_score", 0.0)),
            "initial_validation_score": float(
                payload.get("initial_validation_score", 0.0)
            ),
            "validation_learning_gain": float(
                payload.get("validation_learning_gain", 0.0)
            ),
            "selected_checkpoint_iteration": int(
                payload.get("selected_checkpoint_iteration", -1)
            ),
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
            training_replicate_seed=int(replicate_seed),
            optimizer_seed=int(run_seed),
            parameter_count=params,
            shard_index=int(shard_index),
            num_shards=int(num_shards),
            training_payload=payload,
        ))
        train_steps = int(payload.get(
            "environment_steps_train",
            len(train_seeds) * steps * max(1, int(iterations)),
        ))
        sample_efficiency.append({
            "scenario": scenario,
            "policy_mode": mode,
            "training_replicate_seed": int(replicate_seed),
            "optimizer_seed": int(run_seed),
            "environment_steps_train": train_steps,
            "environment_steps_validation": int(
                payload.get("environment_steps_validation", 0)
            ),
            "environment_steps_eval": int(len(heldout_test_seeds) * steps),
            "iterations": int(iterations),
            "best_score": float(payload.get("best_score", 0.0)),
            "initial_validation_score": float(
                payload.get("initial_validation_score", 0.0)
            ),
            "validation_learning_gain": float(
                payload.get("validation_learning_gain", 0.0)
            ),
            "selected_checkpoint_iteration": int(
                payload.get("selected_checkpoint_iteration", -1)
            ),
            "heldout_objective": float(np.mean([
                validation_utility(row) for row in heldout_rows
            ])) if heldout_rows else 0.0,
            "elapsed_sec": elapsed,
            "selection_metric": SELECTION_OBJECTIVE_VERSION,
            "confirmatory": bool(confirmatory),
            "hyperparameter_source": str(hyperparameter_source),
            "hyperparameter_protocol_status": hyperparameter_protocol_status,
            "frozen_config_sha256": str(frozen_config_sha256),
            "selected_candidate_id": str(selected_candidate_id),
            "actual_hyperparameters_json": json.dumps(
                actual_hyperparameters, sort_keys=True, separators=(",", ":")
            ),
            "actual_hyperparameter_sha256": actual_hyperparameter_sha256,
            "learned_baseline_implementation_version": (
                LEARNED_BASELINE_IMPLEMENTATION_VERSION
            ),
            "code_revision": source_revision,
            "source_manifest_sha256": source_manifest,
            "source_identity_status": source_identity_status,
            "training_path_protocol": str(payload.get("training_path_protocol", "")),
            "checkpoint_selection_protocol": str(
                payload.get("checkpoint_selection_protocol", "")
            ),
            "gradient_updates_train": int(payload.get("gradient_updates_train", 0)),
            "actor_optimizer_steps_train": int(payload.get("actor_optimizer_steps_train", 0)),
            "critic_optimizer_steps_train": int(payload.get("critic_optimizer_steps_train", 0)),
            "temperature_optimizer_steps_train": int(
                payload.get("temperature_optimizer_steps_train", 0)
            ),
            "algorithm_family": (
                "on_policy_joint_flat_ppo"
                if mode in FLAT_PPO_MODES else (
                    "on_policy_smdp_ppo" if mode in POLICY_MODES
                    else f"off_policy_{model.config.algorithm}"
                )
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
    stress_evidence = stress_stratum_evidence(
        checks,
        ppo_controls + offpolicy_controls,
        metrics=("total_return",),
    )
    budgets = _budget_statuses(rows, parameter_budget, sample_efficiency)
    learning_dynamics = learning_dynamics_summary(run_rows)
    coverage = _experiment_matrix_coverage(rows)
    ppo_modes_run = {
        str(row.get("policy_mode", "")) for row in rows
        if str(row.get("policy_mode", "")) in POLICY_MODES
    }
    ppo_baseline_status = (
        metric_status
        if set(POLICY_MODES) <= ppo_modes_run
        and budgets["ppo_parameter_budget_status"] in {
            "matched_exact", "matched_within_5pct"
        }
        and budgets["ppo_trainer_budget_status"] in {
            "matched_exact", "controlled_by_ppo_family"
        }
        else "partial_run_or_budget_mismatch"
    )
    if learning_dynamics["learning_dynamics_status"] != "supported":
        all_metric_status = "training_not_demonstrated"
        risk_adjusted_status = "training_not_demonstrated"
        responsibility_status = "training_not_demonstrated"
        ppo_baseline_status = "training_not_demonstrated"
    if hyperparameter_protocol_status != "frozen_validation_only":
        all_metric_status = "exploratory_unfrozen_hyperparameters"
        risk_adjusted_status = "exploratory_unfrozen_hyperparameters"
        responsibility_status = "exploratory_unfrozen_hyperparameters"
        ppo_baseline_status = "exploratory_unfrozen_hyperparameters"
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
            validation_seeds=validation_seed_list,
            eval_seeds=heldout_test_seeds,
            steps=int(steps),
            assets=int(assets),
            iterations=int(iterations),
            optimizer_seed=int(optimizer_seed),
            optimizer_seeds=replicate_seeds,
            min_pairs=int(min_pairs),
            ppo_hidden_dim=int(ppo_hidden_dim),
            ppo_learning_rate=float(ppo_learning_rate),
            ppo_epochs=int(ppo_epochs),
            ppo_minibatch_size=int(ppo_minibatch_size),
            ppo_init_log_std=float(ppo_init_log_std),
            training_reward_scale=float(training_reward_scale),
            offpolicy_hidden_dim=int(offpolicy_hidden_dim),
            offpolicy_learning_rate=float(offpolicy_learning_rate),
            offpolicy_replay_capacity=int(offpolicy_replay_capacity),
            offpolicy_warmup_steps=int(offpolicy_warmup_steps),
            offpolicy_batch_size=int(offpolicy_batch_size),
            offpolicy_updates_per_step=int(offpolicy_updates_per_step),
            confirmatory=bool(confirmatory),
            hyperparameter_source=str(hyperparameter_source),
            frozen_config_sha256=str(frozen_config_sha256),
            selected_candidate_id=str(selected_candidate_id),
            frozen_candidate_parameters_sha256=str(
                frozen_candidate_parameters_sha256
            ),
            code_revision=source_revision,
            source_manifest_sha256=source_manifest,
            source_identity_status=source_identity_status,
            shard_index=int(shard_index),
            num_shards=int(num_shards),
        ),
        "summary": {
            "rows": len(rows),
            "scenario_count": len(set(scenarios)),
            "selected_scenario_count": len({scenario for scenario, _, _ in cells}),
            "selected_pair_count": len({(scenario, mode) for scenario, mode, _ in cells}),
            "selected_cell_count": len(cells),
            "policy_modes": list(policy_modes),
            # Kept as a compatibility alias; these are rollout paths, not
            # independent statistical replications.
            "train_seed_count": len(rollout_seed_roots),
            "rollout_train_seed_count": len(rollout_seed_roots),
            "validation_seed_count": len(validation_seed_list),
            "eval_seed_count": len(heldout_test_seeds),
            "training_replicate_count": len(set(replicate_seeds)),
            "selected_training_replicate_count": len({seed for _, _, seed in cells}),
            "independent_unit": "training_replicate_seed",
            "training_path_protocol": TRAINING_PATH_PROTOCOL,
            "checkpoint_selection_protocol": SELECTION_PROTOCOL,
            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
            "confirmatory_analysis_version": CONFIRMATORY_ANALYSIS_VERSION,
            "practical_effect_thresholds": dict(PRACTICAL_EFFECT_THRESHOLDS),
            "practical_effect_threshold_source": (
                PRACTICAL_EFFECT_THRESHOLD_SOURCE
            ),
            "stress_stratum_total_return_status": stress_evidence,
            "training_reward_scale": float(training_reward_scale),
            "training_protocol_status": "valid",
            "confirmatory": bool(confirmatory),
            "hyperparameter_source": str(hyperparameter_source),
            "hyperparameter_protocol_status": hyperparameter_protocol_status,
            "frozen_config_sha256": str(frozen_config_sha256),
            "selected_candidate_id": str(selected_candidate_id),
            "frozen_candidate_parameters_sha256": str(
                frozen_candidate_parameters_sha256
            ),
            "learned_baseline_implementation_version": (
                LEARNED_BASELINE_IMPLEMENTATION_VERSION
            ),
            "code_revision": source_revision,
            "source_manifest_sha256": source_manifest,
            "source_identity_status": source_identity_status,
            "steps": int(steps),
            "assets": int(assets),
            "iterations": int(iterations),
            "ppo_strong_baseline_status": ppo_baseline_status,
            "ppo_metric_status": metric_status,
            "offpolicy_metric_status": offpolicy_metric_status,
            "strong_learned_baseline_evidence_status": all_metric_status,
            "risk_adjusted_evidence_status": risk_adjusted_status,
            "responsibility_evidence_status": responsibility_status,
            **learning_dynamics,
            **coverage,
            **budgets,
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        },
        "_checkpoint_payloads": checkpoint_payloads,
        "boundary": (
            "PPO-family baselines match active capacity within 5%, initialization "
            "seeds, environment-step budgets, and equal-size HPO search spaces; "
            "selected optimizer settings are reported per policy. Canonical flat "
            "PPO uses one primitive-rate joint action and one critic; generic HRL "
            "retains temporal abstraction. All raw baselines receive the same "
            "causal lag span. Flat SAC/TD3 use a single joint "
            "target/execution action. Cross-algorithm fairness is enforced by "
            "paired held-out seeds, equal environment-step budgets, the same "
            "environment/costs, and trading_metrics_v2; near-equal active capacity "
            "is claimed only inside the PPO family. Statistical uncertainty is "
            "clustered by independently initialized training replicate; held-out "
            "environment seeds are repeated measures inside that cluster."
        ),
    }


def merge_strong_learned_baseline_shards(
    input_dirs: list[Path],
    *,
    min_pairs: int,
    expected_scenarios: list[str] | None = None,
    expected_policy_modes: list[str] | None = None,
    expected_replicate_seeds: list[int] | None = None,
    expected_eval_seeds: list[int] | None = None,
    expected_frozen_config_sha256: str = "",
    expected_selected: dict[str, dict[str, Any]] | None = None,
    expected_code_revision: str = "",
    expected_source_manifest_sha256: str = "",
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
    learning_dynamics = learning_dynamics_summary(run_rows)
    coverage = _experiment_matrix_coverage(
        rows,
        expected_scenarios=expected_scenarios,
        expected_policy_modes=expected_policy_modes,
        expected_replicate_seeds=expected_replicate_seeds,
        expected_eval_seeds=expected_eval_seeds,
    )
    ppo_modes_run = {
        mode for mode in policy_modes if mode in POLICY_MODES
    }
    ppo_baseline_status = (
        metric_status
        if set(POLICY_MODES) <= ppo_modes_run
        and budgets["ppo_parameter_budget_status"] in {
            "matched_exact", "matched_within_5pct"
        }
        and budgets["ppo_trainer_budget_status"] in {
            "matched_exact", "controlled_by_ppo_family"
        }
        else "partial_run_or_budget_mismatch"
    )
    eval_seeds = {
        str(row.get("seed", "")) for row in rows
        if str(row.get("seed", "")).strip()
    }
    training_replicates = {
        str(row.get("training_replicate_seed", "")) for row in rows
        if str(row.get("training_replicate_seed", "")).strip()
    }
    selected_cells = {
        (
            str(row.get("scenario", "")),
            str(row.get("policy_mode", row.get("baseline", ""))),
            str(row.get("training_replicate_seed", "")),
        )
        for row in rows
        if str(row.get("training_replicate_seed", "")).strip()
    }
    rollout_train_seed_counts = {
        int(float(row["train_seed_count"]))
        for row in run_rows
        if str(row.get("train_seed_count", "")).strip()
    }
    rollout_train_seed_count = (
        next(iter(rollout_train_seed_counts))
        if len(rollout_train_seed_counts) == 1 else 0
    )
    validation_seed_counts = {
        int(float(row["validation_seed_count"]))
        for row in run_rows
        if str(row.get("validation_seed_count", "")).strip()
    }
    validation_seed_count = (
        next(iter(validation_seed_counts))
        if len(validation_seed_counts) == 1 else 0
    )
    training_protocols = {
        str(row.get("training_path_protocol", "")) for row in rows
        if str(row.get("training_path_protocol", "")).strip()
    }
    selection_protocols = {
        str(row.get("checkpoint_selection_protocol", "")) for row in rows
        if str(row.get("checkpoint_selection_protocol", "")).strip()
    }
    selection_objectives = {
        str(row.get("selection_objective_version", "")) for row in rows
        if str(row.get("selection_objective_version", "")).strip()
    }
    training_protocol_status = (
        "valid"
        if training_protocols == {TRAINING_PATH_PROTOCOL}
        and selection_protocols == {SELECTION_PROTOCOL}
        and selection_objectives == {SELECTION_OBJECTIVE_VERSION}
        else "invalid_or_mixed"
    )
    hyperparameter_sources = {
        str(row.get("hyperparameter_source", "")) for row in rows
        if str(row.get("hyperparameter_source", "")).strip()
    }
    frozen_hashes = {
        str(row.get("frozen_config_sha256", "")).lower() for row in rows
        if str(row.get("frozen_config_sha256", "")).strip()
    }
    implementation_versions = {
        str(row.get("learned_baseline_implementation_version", ""))
        for row in rows
        if str(row.get("learned_baseline_implementation_version", "")).strip()
    }
    code_revisions = {
        str(row.get("code_revision", "")).strip().lower()
        for row in rows
        if str(row.get("code_revision", "")).strip()
    }
    source_manifests = {
        str(row.get("source_manifest_sha256", "")).strip().lower()
        for row in rows
        if str(row.get("source_manifest_sha256", "")).strip()
    }
    if len(code_revisions) > 1 or len(source_manifests) > 1:
        raise ValueError(
            "learned-baseline shards mix code revisions or source manifests"
        )
    source_rows_complete = bool(rows) and all(
        str(row.get("source_identity_status", "")) == "verified"
        and is_hex_digest(row.get("code_revision"), length=40)
        and is_hex_digest(row.get("source_manifest_sha256"), length=64)
        for row in rows
    )
    source_identity_status = (
        "verified"
        if source_rows_complete
        and len(code_revisions) == 1
        and len(source_manifests) == 1
        else "unregistered_or_incomplete"
    )
    selected_candidates_present = all(
        str(row.get("selected_candidate_id", "")).strip() for row in rows
    )
    candidate_parameter_rows_valid = bool(rows)
    for row in rows:
        try:
            parameters = json.loads(str(row.get("actual_hyperparameters_json", "")))
            recorded_hash = str(
                row.get("actual_hyperparameter_sha256", "")
            ).lower()
            if (
                not isinstance(parameters, dict)
                or canonical_hyperparameter_sha256(parameters) != recorded_hash
            ):
                candidate_parameter_rows_valid = False
                break
        except (TypeError, ValueError, json.JSONDecodeError):
            candidate_parameter_rows_valid = False
            break
    freeze_binding = validate_frozen_result_binding(
        rows,
        expected_frozen_config_sha256=expected_frozen_config_sha256,
        expected_selected=expected_selected,
        expected_code_revision=expected_code_revision,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
    )
    valid_digest = (
        len(frozen_hashes) == 1
        and len(next(iter(frozen_hashes))) == 64
        and all(
            char in "0123456789abcdef" for char in next(iter(frozen_hashes))
        )
    )
    if (
        hyperparameter_sources == {"frozen_nested_validation"}
        and valid_digest
        and implementation_versions == {LEARNED_BASELINE_IMPLEMENTATION_VERSION}
        and selected_candidates_present
        and candidate_parameter_rows_valid
        and source_identity_status == "verified"
        and freeze_binding["status"] == "verified"
    ):
        hyperparameter_protocol_status = "frozen_validation_only"
    elif hyperparameter_sources in (set(), {"exploratory_unfrozen"}):
        hyperparameter_protocol_status = "exploratory_unfrozen"
    else:
        hyperparameter_protocol_status = "invalid_or_mixed"
    if coverage["matrix_coverage_status"] != "complete":
        all_metric_status = "incomplete_matrix"
        risk_adjusted_status = "incomplete_matrix"
        responsibility_status = "incomplete_matrix"
        ppo_baseline_status = "partial_run_or_budget_mismatch"
    if training_protocol_status != "valid":
        all_metric_status = "invalid_training_protocol"
        risk_adjusted_status = "invalid_training_protocol"
        responsibility_status = "invalid_training_protocol"
        ppo_baseline_status = "invalid_training_protocol"
    if (
        learning_dynamics["learning_dynamics_status"] != "supported"
        and coverage["matrix_coverage_status"] == "complete"
        and training_protocol_status == "valid"
    ):
        all_metric_status = "training_not_demonstrated"
        risk_adjusted_status = "training_not_demonstrated"
        responsibility_status = "training_not_demonstrated"
        ppo_baseline_status = "training_not_demonstrated"
    if (
        hyperparameter_sources == {"frozen_nested_validation"}
        and source_identity_status != "verified"
    ):
        all_metric_status = "unverified_source_identity"
        risk_adjusted_status = "unverified_source_identity"
        responsibility_status = "unverified_source_identity"
        ppo_baseline_status = "unverified_source_identity"
    elif hyperparameter_protocol_status != "frozen_validation_only":
        all_metric_status = "exploratory_unfrozen_hyperparameters"
        risk_adjusted_status = "exploratory_unfrozen_hyperparameters"
        responsibility_status = "exploratory_unfrozen_hyperparameters"
        ppo_baseline_status = "exploratory_unfrozen_hyperparameters"
    stress_evidence = stress_stratum_evidence(
        checks,
        ppo_controls + offpolicy_controls,
        metrics=("total_return",),
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
            "selected_cell_count": len(selected_cells),
            "policy_modes": policy_modes,
            "shard_count": len(input_dirs),
            "train_seed_count": int(rollout_train_seed_count),
            "rollout_train_seed_count": int(rollout_train_seed_count),
            "validation_seed_count": int(validation_seed_count),
            "eval_seed_count": len(eval_seeds),
            "training_replicate_count": len(training_replicates),
            "selected_training_replicate_count": len(training_replicates),
            "independent_unit": "training_replicate_seed",
            "training_protocol_status": training_protocol_status,
            "hyperparameter_protocol_status": hyperparameter_protocol_status,
            "hyperparameter_source": (
                next(iter(hyperparameter_sources))
                if len(hyperparameter_sources) == 1 else "mixed_or_missing"
            ),
            "frozen_config_sha256": (
                next(iter(frozen_hashes))
                if len(frozen_hashes) == 1 else "mixed_or_missing"
            ),
            "source_identity_status": source_identity_status,
            "code_revision": (
                next(iter(code_revisions))
                if len(code_revisions) == 1 else "mixed_or_missing"
            ),
            "source_manifest_sha256": (
                next(iter(source_manifests))
                if len(source_manifests) == 1 else "mixed_or_missing"
            ),
            "learned_baseline_implementation_version": (
                next(iter(implementation_versions))
                if len(implementation_versions) == 1 else "mixed_or_missing"
            ),
            "training_path_protocol": (
                next(iter(training_protocols))
                if len(training_protocols) == 1 else "mixed_or_missing"
            ),
            "checkpoint_selection_protocol": (
                next(iter(selection_protocols))
                if len(selection_protocols) == 1 else "mixed_or_missing"
            ),
            "selection_objective_version": (
                next(iter(selection_objectives))
                if len(selection_objectives) == 1 else "mixed_or_missing"
            ),
            "confirmatory_analysis_version": CONFIRMATORY_ANALYSIS_VERSION,
            "practical_effect_thresholds": dict(PRACTICAL_EFFECT_THRESHOLDS),
            "practical_effect_threshold_source": (
                PRACTICAL_EFFECT_THRESHOLD_SOURCE
            ),
            "frozen_result_binding_status": freeze_binding["status"],
            "frozen_result_binding_verified_rows": int(
                freeze_binding["verified_row_count"]
            ),
            "stress_stratum_total_return_status": stress_evidence,
            "ppo_strong_baseline_status": ppo_baseline_status,
            "ppo_metric_status": metric_status,
            "offpolicy_metric_status": offpolicy_metric_status,
            "strong_learned_baseline_evidence_status": all_metric_status,
            "risk_adjusted_evidence_status": risk_adjusted_status,
            "responsibility_evidence_status": responsibility_status,
            **learning_dynamics,
            **coverage,
            **budgets,
            "merge_status": "merged",
        },
        "boundary": (
            "Merged learned-baseline shards. PPO comparisons match trainable "
            "capacity and HPO search budget; selected optimizer settings are "
            "reported per policy. SAC/TD3 use their native twin-Q architectures "
            "under the same paired evaluation and environment-step budget; "
            "cross-family parameter equality is not claimed. Statistical uncertainty is "
            "clustered by independently initialized training replicate; held-out "
            "environment seeds remain repeated measures within each replicate."
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
    checkpoint_dir = output_dir / "checkpoints"
    for item in payload.get("_checkpoint_payloads", []):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(item["payload"], checkpoint_dir / item["checkpoint_file"])
    serializable_payload = {
        key: value for key, value in payload.items() if not key.startswith("_")
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(serializable_payload, f, indent=2)
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
        f"- independent training replicates: "
        f"`{payload['summary']['training_replicate_count']}`",
        f"- rollout train seeds per replicate: "
        f"`{payload['summary']['rollout_train_seed_count']}`",
        f"- validation seeds: `{payload['summary'].get('validation_seed_count', 0)}`",
        f"- eval seeds: `{payload['summary']['eval_seed_count']}`",
        f"- matrix coverage: `{payload['summary']['matrix_coverage_status']}`",
        "",
        "| check | scope | stratum | status | metric | threshold | paired rows | train reps | delta | CI95 low | CI95 high | win rate | Holm p |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["paired_checks"]:
        delta = float(row["delta_mean"])
        ci_low = float(row["delta_ci95_low"])
        ci_high = float(row["delta_ci95_high"])
        win_rate = float(row["win_rate"])
        holm_p = float(row["holm_adjusted_p_value"])
        lines.append(
            f"| {row['check']} | {row.get('inference_scope', 'missing')} "
            f"| {row.get('scenario_stratum', 'missing')} | {row['status']} "
            f"| {row['metric']} | {float(row.get('practical_effect_threshold', 0.0)):.4f} "
            f"| {row['n_common']} | {row['n_independent']} | {delta:+.4f} "
            f"| {ci_low:+.4f} | {ci_high:+.4f} "
            f"| {win_rate:.2f} | {holm_p:.4f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--policy-modes", nargs="+", choices=ALL_POLICY_MODES, default=list(DEFAULT_POLICY_MODES))
    parser.add_argument(
        "--train-seeds", type=int, nargs="+",
        default=list(DEFAULT_ROLLOUT_SEED_ROOTS),
    )
    parser.add_argument(
        "--validation-seeds", type=int, nargs="+",
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVAL_SEEDS),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument(
        "--optimizer-seed",
        type=int,
        default=None,
        help="Deprecated single-replicate override for smoke tests.",
    )
    parser.add_argument(
        "--optimizer-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_OPTIMIZER_SEEDS),
        help="Independent policy-training initialization seeds.",
    )
    parser.add_argument("--min-pairs", type=int, default=10)
    parser.add_argument("--ppo-hidden-dim", type=int, default=64)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-minibatch-size", type=int, default=512)
    parser.add_argument("--ppo-init-log-std", type=float, default=-1.0)
    parser.add_argument(
        "--training-reward-scale",
        type=float,
        default=DEFAULT_TRAINING_REWARD_SCALE,
    )
    parser.add_argument("--offpolicy-hidden-dim", type=int, default=64)
    parser.add_argument("--offpolicy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--offpolicy-replay-capacity", type=int, default=100_000)
    parser.add_argument("--offpolicy-warmup-steps", type=int, default=2048)
    parser.add_argument("--offpolicy-batch-size", type=int, default=64)
    parser.add_argument("--offpolicy-updates-per-step", type=int, default=1)
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument(
        "--hyperparameter-source",
        choices=("exploratory_unfrozen", "frozen_nested_validation"),
        default="exploratory_unfrozen",
    )
    parser.add_argument("--frozen-config-sha256", default="")
    parser.add_argument("--selected-candidate-id", default="")
    parser.add_argument("--frozen-candidate-parameters-sha256", default="")
    parser.add_argument("--code-revision", default="")
    parser.add_argument("--source-manifest-sha256", default="")
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
        replicate_seeds = (
            [int(args.optimizer_seed)]
            if args.optimizer_seed is not None
            else [int(seed) for seed in args.optimizer_seeds]
        )
        payload = run_strong_learned_baseline_validation(
            scenarios=list(args.scenarios),
            policy_modes=list(args.policy_modes),
            train_seeds=list(args.train_seeds),
            validation_seeds=list(args.validation_seeds),
            eval_seeds=list(args.eval_seeds),
            steps=int(args.steps),
            assets=int(args.assets),
            iterations=int(args.iterations),
            optimizer_seed=int(replicate_seeds[0]),
            optimizer_seeds=replicate_seeds,
            min_pairs=int(args.min_pairs),
            ppo_hidden_dim=int(args.ppo_hidden_dim),
            ppo_learning_rate=float(args.ppo_learning_rate),
            ppo_epochs=int(args.ppo_epochs),
            ppo_minibatch_size=int(args.ppo_minibatch_size),
            ppo_init_log_std=float(args.ppo_init_log_std),
            training_reward_scale=float(args.training_reward_scale),
            offpolicy_hidden_dim=int(args.offpolicy_hidden_dim),
            offpolicy_learning_rate=float(args.offpolicy_learning_rate),
            offpolicy_replay_capacity=int(args.offpolicy_replay_capacity),
            offpolicy_warmup_steps=int(args.offpolicy_warmup_steps),
            offpolicy_batch_size=int(args.offpolicy_batch_size),
            offpolicy_updates_per_step=int(args.offpolicy_updates_per_step),
            confirmatory=bool(args.confirmatory),
            hyperparameter_source=str(args.hyperparameter_source),
            frozen_config_sha256=str(args.frozen_config_sha256),
            selected_candidate_id=str(args.selected_candidate_id),
            frozen_candidate_parameters_sha256=str(
                args.frozen_candidate_parameters_sha256
            ),
            code_revision=str(args.code_revision),
            expected_source_manifest_sha256=str(
                args.source_manifest_sha256
            ),
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
