#!/usr/bin/env python3
"""Fail-fast validator for the submission-grade FreqDuet v4 contract."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner_v3 import load_config


REFERENCE = "F_freqduet_protocol_v4_main_hiro.yaml"
REFERENCE_CONFIGS = [
    REFERENCE,
    "F_freqduet_protocol_v4_csac_hiro.yaml",
    "F_freqduet_protocol_v4_nofreq_hiro.yaml",
    "F_freqduet_protocol_v4_rawhistory_hiro.yaml",
    "F_freqduet_protocol_v4_allfreq_hiro.yaml",
    "F_freqduet_protocol_v4_nopromotion_hiro.yaml",
    "F_freqduet_protocol_v4_noleakage_hiro.yaml",
    "F_freqduet_protocol_v4_nodriftfb_hiro.yaml",
    "F_freqduet_protocol_v4_noprior_hiro.yaml",
    "F_freqduet_protocol_v4_continuous_holding_hiro.yaml",
    "F_freqduet_protocol_v4_nolowercontext_hiro.yaml",
]


def _get(config, path):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


CORE_REQUIRED = {
    "protocol.version": "freqduet-eval-v4",
    "randomness.mode": "isolated_streams_v4",
    "env.fleet_inventory_mode": "fixed_pool",
    "env.upper_fleet_state_mode": "fixed_pool_readiness_v4",
    "env.observation_contract": "deployable_apc_avl_v4",
    "env.headway_reward_mode": "forward_event_only",
    "upper.temperature_contract": "bounded_log_parameter_v4",
    "upper.timetable_planner.terminal_schedule_mode": "exact_headway_curve",
    "upper.timetable_planner.terminal_dispatch": True,
    "upper.timetable_planner.terminal_headway_floor.enable": False,
    "upper.interval_credit.enable": True,
    "lower.observation_contract": "deployable_apc_avl_v4",
    "lower.headway_reward_mode": "forward_event_only",
    "lower.unobserved_action_mode": "zero",
    "lower.temperature_contract": "bounded_log_parameter_v4",
    "lower.entropy_action_coordinates": "normalized_unit_interval",
    "lower.cost_limit_semantics": "per_decision_rate",
    "lower.state_encoder.input_schema": "causal_forward_v4",
    "reward_attribution.upper_wait_weight": 0.0,
    "coupling.tpc.target_distribution": "bounded_logistic_normal_v4",
    "training.decouple_init_seeds": True,
    "training.checkpoint_contract": "exact_training_state_v4",
}


SPLIT_FREQUENCY_REQUIRED = {
    "frequency.enable": True,
    "frequency.method": "harmonic",
    "frequency.upper_features": True,
    "frequency.lower_features": True,
    "frequency.forecast_mode": "causal",
    "frequency.observation_source": "apc_boardings",
    "reward_attribution.enable": True,
    "reward_attribution.assignment_mode": "frozen_passenger",
    "upper.interval_credit.wait_ownership": "frozen_low_frequency",
}


ROLE_REQUIRED = {
    "physical_causal_frequency_owned_main": {
        "frequency.upper_mode": "low",
        "frequency.lower_mode": "high",
        "frequency.use_historical_prior": True,
        "frequency.promotion.enable": True,
        "frequency.drift_feedback.enable": True,
        "leakage.enable": True,
    },
    "candidate_load_weighted_holding": {
        "frequency.upper_mode": "low",
        "frequency.lower_mode": "high",
        "frequency.use_historical_prior": True,
        "frequency.promotion.enable": True,
        "frequency.drift_feedback.enable": True,
        "leakage.enable": True,
        "lower.load_weighted_holding.enable": True,
        "lower.load_weighted_holding.source": "observation_load",
        "lower.load_weighted_holding.action_norm_s": 45.0,
        "lower.load_weighted_holding.load_clip": 1.0,
    },
    "optimizer_ablation_standard_constrained_sac": {
        "frequency.upper_mode": "low",
        "frequency.lower_mode": "high",
    },
    "ablation_no_frequency_state_allocation": {
        "frequency.upper_mode": "all",
        "frequency.lower_mode": "all",
    },
    "ablation_no_promotion": {"frequency.promotion.enable": False},
    "ablation_no_leakage_regularizer": {"leakage.enable": False},
    "ablation_no_drift_feedback": {
        "frequency.drift_feedback.enable": False,
    },
    "ablation_no_historical_prior": {
        "frequency.use_historical_prior": False,
    },
    "ablation_continuous_holding_action": {},
    "ablation_no_causal_lower_context": {
        "frequency.lower_context.enable": False,
    },
    "ablation_no_frequency_system": {},
    "ablation_raw_causal_history": {},
}


ROLE_ALLOWED_DIFFS = {
    "candidate_load_weighted_holding": {
        "lower.load_weighted_holding.enable",
        "lower.load_weighted_holding.source",
        "lower.load_weighted_holding.reward_weight",
        "lower.load_weighted_holding.action_norm_s",
        "lower.load_weighted_holding.load_clip",
    },
    "optimizer_ablation_standard_constrained_sac": {
        "upper.algorithm_id", "upper.critic_aggregation",
        "upper.ensemble_size", "upper.resac_beta", "upper.beta_ood",
        "upper.weight_reg", "lower.algorithm_id",
        "lower.critic_aggregation", "lower.ensemble_size",
        "lower.resac_beta", "lower.beta_ood", "lower.weight_reg",
    },
    "ablation_no_frequency_system": {
        "protocol.frequency_contract", "upper.state_dim",
        "upper.interval_credit.wait_ownership", "frequency.enable",
        "frequency.upper_features", "frequency.lower_features",
        "frequency.use_historical_prior", "frequency.promotion.enable",
        "frequency.drift_feedback.enable", "reward_attribution.enable",
        "leakage.enable",
    },
    "ablation_raw_causal_history": {
        "protocol.frequency_contract",
        "upper.interval_credit.wait_ownership", "frequency.method",
        "frequency.od_features", "frequency.upper_history_bins",
        "frequency.lower_history_bins", "frequency.promotion.enable",
        "frequency.drift_feedback.enable", "reward_attribution.enable",
    },
    "ablation_no_frequency_state_allocation": {
        "frequency.upper_mode", "frequency.lower_mode",
    },
    "ablation_no_promotion": {"frequency.promotion.enable"},
    "ablation_no_leakage_regularizer": {"leakage.enable"},
    "ablation_no_drift_feedback": {"frequency.drift_feedback.enable"},
    "ablation_no_historical_prior": {"frequency.use_historical_prior"},
    "ablation_continuous_holding_action": {
        "protocol.action_contract", "lower.action_bins",
    },
    "ablation_no_causal_lower_context": {
        "protocol.lower_context_contract", "frequency.lower_context.enable",
    },
}


COMMON_ALLOWED_DIFFS = {"_name", "protocol.role"}


def _validate_expected(config, expected, name, errors):
    for path, wanted in expected.items():
        observed = _get(config, path)
        if observed != wanted:
            errors.append(
                f"{name}: {path}={observed!r}, expected {wanted!r}")


def _flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(_flatten(item, path))
        else:
            result[path] = item
    return result


def _validate_single_axis(config, reference, name, errors):
    role = _get(config, "protocol.role")
    allowed = COMMON_ALLOWED_DIFFS | ROLE_ALLOWED_DIFFS.get(role, set())
    current_flat = _flatten(config)
    reference_flat = _flatten(reference)
    differences = {
        path for path in current_flat.keys() | reference_flat.keys()
        if current_flat.get(path) != reference_flat.get(path)
    }
    unexpected = sorted(differences - allowed)
    if unexpected:
        errors.append(
            f"{name}: confounded ablation changes {unexpected!r}")


def validate_config(config, name="config"):
    errors = []
    _validate_expected(config, CORE_REQUIRED, name, errors)
    role = _get(config, "protocol.role")
    if role not in ROLE_REQUIRED:
        errors.append(f"{name}: unsupported protocol.role {role!r}")
    else:
        _validate_expected(config, ROLE_REQUIRED[role], name, errors)

    frequency_contract = _get(config, "protocol.frequency_contract")
    if frequency_contract == "split_owned":
        _validate_expected(config, SPLIT_FREQUENCY_REQUIRED, name, errors)
    elif frequency_contract == "raw_history_control":
        _validate_expected(config, {
            "frequency.enable": True,
            "frequency.method": "raw_history",
            "frequency.upper_features": True,
            "frequency.lower_features": True,
            "frequency.forecast_mode": "causal",
            "frequency.observation_source": "apc_boardings",
            "frequency.promotion.enable": False,
            "frequency.drift_feedback.enable": False,
            "reward_attribution.enable": False,
            "upper.interval_credit.wait_ownership": "all_wait_legacy",
        }, name, errors)
    elif frequency_contract == "no_frequency_control":
        _validate_expected(config, {
            "frequency.enable": False,
            "frequency.upper_features": False,
            "frequency.lower_features": False,
            "frequency.use_historical_prior": False,
            "frequency.promotion.enable": False,
            "frequency.drift_feedback.enable": False,
            "reward_attribution.enable": False,
            "upper.interval_credit.wait_ownership": "all_wait_legacy",
            "upper.state_dim": 15,
            "leakage.enable": False,
        }, name, errors)
    else:
        errors.append(
            f"{name}: unsupported frequency contract {frequency_contract!r}")

    action_contract = _get(config, "protocol.action_contract")
    action_bins = _get(config, "lower.action_bins")
    if action_contract == "discrete_holding_v4":
        expected_bins = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]
        if action_bins != expected_bins:
            errors.append(
                f"{name}: lower.action_bins={action_bins!r}, "
                f"expected {expected_bins!r}")
    elif action_contract == "continuous_holding_ablation":
        if action_bins is not None:
            errors.append(f"{name}: continuous action ablation has action bins")
    else:
        errors.append(f"{name}: unsupported action contract {action_contract!r}")

    context_contract = _get(config, "protocol.lower_context_contract")
    if context_contract == "causal_rich_v4":
        _validate_expected(config, {
            "frequency.lower_context.enable": True,
            "frequency.lower_context.features": [
                "load", "capacity", "queue", "speed_residual",
                "shock_age", "schedule_slack",
            ],
        }, name, errors)
    elif context_contract == "no_rich_context_ablation":
        _validate_expected(config, {
            "frequency.lower_context.enable": False,
        }, name, errors)
    else:
        errors.append(
            f"{name}: unsupported lower-context contract {context_contract!r}")
    load_hold_enabled = bool(_get(
        config, "lower.load_weighted_holding.enable"))
    if load_hold_enabled:
        reward_weight = float(_get(
            config, "lower.load_weighted_holding.reward_weight") or 0.0)
        if reward_weight <= 0.0:
            errors.append(
                f"{name}: enabled load-weighted holding needs positive weight")
        if "load" not in (_get(
                config, "frequency.lower_context.features") or []):
            errors.append(
                f"{name}: load-weighted holding needs causal APC load context")
    if _get(config, "coupling.tpc.enable") and _get(
            config, "coupling.tpc.target_distribution"
    ) != "bounded_logistic_normal_v4":
        errors.append(f"{name}: enabled TPC must use bounded logistic-normal")
    algorithm_contracts = {
        (
            "pessimistic_ensemble_sac_v4",
            "pessimistic_ensemble_sac_lagrangian_v4",
        ): ("ensemble_mean_lcb", 10),
        (
            "standard_sac_v4",
            "standard_sac_lagrangian_v4",
        ): ("twin_min", 2),
    }
    algorithm_pair = (
        _get(config, "upper.algorithm_id"),
        _get(config, "lower.algorithm_id"),
    )
    expected_optimizer = algorithm_contracts.get(algorithm_pair)
    if expected_optimizer is None:
        errors.append(f"{name}: unsupported algorithm pair {algorithm_pair!r}")
    else:
        aggregation, ensemble_size = expected_optimizer
        for level in ("upper", "lower"):
            if _get(config, f"{level}.critic_aggregation") != aggregation:
                errors.append(
                    f"{name}: {level}.critic_aggregation must be {aggregation}")
            if int(_get(config, f"{level}.ensemble_size") or 0) != ensemble_size:
                errors.append(
                    f"{name}: {level}.ensemble_size must be {ensemble_size}")
        if aggregation == "twin_min":
            for level in ("upper", "lower"):
                for field in ("resac_beta", "beta_ood", "weight_reg"):
                    if float(_get(config, f"{level}.{field}") or 0.0) != 0.0:
                        errors.append(
                            f"{name}: {level}.{field} must be zero for twin-min")

    for level in ("upper", "lower"):
        alpha = float(_get(config, f"{level}.initial_alpha") or 0.0)
        minimum = float(_get(config, f"{level}.minimum_alpha") or 0.0)
        maximum = float(_get(config, f"{level}.maximum_alpha") or 0.0)
        if not minimum <= alpha <= maximum:
            errors.append(
                f"{name}: {level} initial alpha is outside its bounds")
    return errors


def validate_all(config_names=None):
    names = list(config_names or REFERENCE_CONFIGS)
    errors = []
    reference = load_config(ROOT / "configs_freqduet" / REFERENCE)
    for name in names:
        path = ROOT / "configs_freqduet" / name
        if not path.exists():
            errors.append(f"missing v4 config: {path}")
            continue
        config = load_config(path)
        errors.extend(validate_config(config, name=name))
        if name != REFERENCE:
            _validate_single_axis(config, reference, name, errors)
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=REFERENCE_CONFIGS)
    args = parser.parse_args()
    errors = validate_all(args.configs)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"validated {len(args.configs)} FreqDuet v4 config(s)")


if __name__ == "__main__":
    main()
