#!/usr/bin/env python3
"""Fail-fast validator for the journey-feasible FreqDuet v5 screen."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frequency.demand_frequency import DemandFrequencyTracker
from runner_v3 import load_config


REFERENCE = "F_freqduet_protocol_v5_main_hiro.yaml"
REFERENCE_CONFIGS = [
    REFERENCE,
    "F_freqduet_protocol_v5_nofreq_hiro.yaml",
    "F_freqduet_protocol_v5_rawhistory_hiro.yaml",
    "F_freqduet_protocol_v5_allfreq_hiro.yaml",
    "F_freqduet_protocol_v5_upperonly_hiro.yaml",
    "F_freqduet_protocol_v5_loweronly_hiro.yaml",
    "F_freqduet_protocol_v5_nobudget_hiro.yaml",
    "F_freqduet_protocol_v5_noguard_hiro.yaml",
    "F_freqduet_protocol_v5_noloadcost_hiro.yaml",
    "F_freqduet_protocol_v5_waitonlycredit_hiro.yaml",
    "F_freqduet_protocol_v5_csac_hiro.yaml",
]


def _get(config, path):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


CORE_REQUIRED = {
    "protocol.version": "freqduet-eval-v5",
    "protocol.action_contract": "causal_discrete_holding_v5",
    "protocol.objective_contract": "restricted_passenger_journey_v5",
    "objective.primary_endpoint": "restricted_total_journey_horizon_min",
    "randomness.mode": "isolated_streams_v4",
    "env.fleet_inventory_mode": "fixed_pool",
    "env.upper_fleet_state_mode": "fixed_pool_readiness_v4",
    "env.observation_contract": "deployable_apc_avl_v4",
    "env.headway_reward_mode": "forward_event_only",
    "upper.fleet_mode": "fixed",
    "upper.transition_stream_mode": "planner_key",
    "upper.credit_assignment.system_reward_mode": "none",
    "upper.credit_assignment.gap_credit_mode": "none",
    "upper.credit_assignment.reliability_reward_mode": "uniform",
    "upper.credit_assignment.reliability_reward_weight": 5.0,
    "upper.timetable_planner.terminal_schedule_mode": "exact_headway_curve",
    "upper.timetable_planner.terminal_dispatch": True,
    "upper.timetable_planner.terminal_headway_floor.enable": False,
    "upper.timetable_planner.basis_per_direction": 2,
    "upper.timetable_planner.shared_directions": True,
    "upper.timetable_planner.plan_all_directions": False,
    "upper.timetable_planner.headway_budget_mode": "zero_sum_delta_v5",
    "upper.timetable_planner.coefficient_parameterization": (
        "antisymmetric_linear_v5"),
    "upper.timetable_planner.promotion_replan": False,
    "upper.interval_credit.enable": True,
    "upper.interval_credit.assignment_mode": "additive",
    "upper.interval_credit.wait_ownership": "all_wait_legacy",
    "upper.interval_credit.wait_reference_min": 20.0,
    "upper.interval_credit.onboard_reference_min": 20.0,
    "upper.interval_credit.dispatch_backlog_reference_trips": 1.0,
    "upper.interval_credit.weights.wait": 1.0,
    "upper.interval_credit.weights.onboard": 1.0,
    "upper.interval_credit.weights.dispatch_backlog": 0.5,
    "upper.interval_credit.weights.headway": 0.25,
    "upper.interval_credit.weights.fleet": 0.25,
    "lower.observation_contract": "deployable_apc_avl_v4",
    "lower.headway_reward_mode": "forward_event_only",
    "lower.unobserved_action_mode": "zero",
    "lower.state_encoder.input_schema": "causal_forward_v4",
    "lower.terminal_action_mode": "transition",
    "lower.trip_boundary_mode": "reset",
    "lower.holding_action_trace_mode": "all_decisions",
    "lower.causal_holding_guard.enable": True,
    "lower.causal_holding_guard.max_deficit_fraction": 1.0,
    "lower.causal_holding_guard.minimum_deficit_s": 0.0,
    "lower.load_weighted_holding.enable": True,
    "lower.load_weighted_holding.source": "observation_load",
    "lower.load_weighted_holding.reward_weight": 0.03,
    "frequency.forecast_mode": "causal",
    "frequency.observation_source": "apc_boardings",
    "frequency.promotion.enable": False,
    "frequency.drift_feedback.enable": False,
    "reward_attribution.enable": False,
    "leakage.enable": False,
    "training.decouple_init_seeds": True,
    "training.checkpoint_contract": "exact_training_state_v4",
}


ROLE_REQUIRED = {
    "journey_feasible_frequency_main": {},
    "ablation_no_frequency_features_v5": {
        "protocol.frequency_contract": "no_frequency_features_v5",
        "frequency.enable": False,
        "frequency.upper_features": False,
        "frequency.lower_features": False,
        "frequency.use_historical_prior": False,
    },
    "ablation_raw_causal_history_v5": {
        "protocol.frequency_contract": "raw_history_features_v5",
        "frequency.enable": True,
        "frequency.method": "raw_history",
        "frequency.od_features": False,
        "frequency.upper_history_bins": 6,
        "frequency.lower_history_bins": 4,
    },
    "ablation_all_bands_both_layers_v5": {
        "frequency.upper_mode": "all",
        "frequency.lower_mode": "all",
    },
    "ablation_upper_low_only_v5": {
        "frequency.upper_features": True,
        "frequency.lower_features": False,
    },
    "ablation_lower_high_only_v5": {
        "frequency.upper_features": False,
        "frequency.lower_features": True,
        "frequency.replace_upper_demand_with_low": False,
    },
    "ablation_no_headway_budget_v5": {
        "upper.timetable_planner.headway_budget_mode": "free",
    },
    "ablation_no_causal_holding_guard_v5": {
        "lower.causal_holding_guard.enable": False,
    },
    "ablation_no_load_holding_cost_v5": {
        "lower.load_weighted_holding.enable": False,
        "lower.load_weighted_holding.reward_weight": 0.0,
    },
    "ablation_wait_only_upper_credit_v5": {
        "upper.interval_credit.weights.onboard": 0.0,
        "upper.interval_credit.weights.dispatch_backlog": 0.0,
    },
    "optimizer_ablation_standard_constrained_sac_v5": {},
}


ROLE_ALLOWED_DIFFS = {
    "ablation_no_frequency_features_v5": {
        "protocol.frequency_contract",
        "frequency.enable",
        "frequency.upper_features",
        "frequency.lower_features",
        "frequency.use_historical_prior",
    },
    "ablation_raw_causal_history_v5": {
        "protocol.frequency_contract",
        "frequency.method",
        "frequency.od_features",
        "frequency.upper_history_bins",
        "frequency.lower_history_bins",
    },
    "ablation_all_bands_both_layers_v5": {
        "frequency.upper_mode", "frequency.lower_mode",
    },
    "ablation_upper_low_only_v5": {"frequency.lower_features"},
    "ablation_lower_high_only_v5": {
        "frequency.upper_features", "frequency.replace_upper_demand_with_low",
    },
    "ablation_no_headway_budget_v5": {
        "upper.timetable_planner.headway_budget_mode",
    },
    "ablation_no_causal_holding_guard_v5": {
        "lower.causal_holding_guard.enable",
    },
    "ablation_no_load_holding_cost_v5": {
        "lower.load_weighted_holding.enable",
        "lower.load_weighted_holding.reward_weight",
    },
    "ablation_wait_only_upper_credit_v5": {
        "upper.interval_credit.weights.onboard",
        "upper.interval_credit.weights.dispatch_backlog",
    },
    "optimizer_ablation_standard_constrained_sac_v5": {
        "upper.algorithm_id", "upper.critic_aggregation",
        "upper.ensemble_size", "upper.resac_beta", "upper.beta_ood",
        "upper.weight_reg", "lower.algorithm_id",
        "lower.critic_aggregation", "lower.ensemble_size",
        "lower.resac_beta", "lower.beta_ood", "lower.weight_reg",
    },
}


COMMON_ALLOWED_DIFFS = {"_name", "protocol.role"}
SPLIT_ROLES = set(ROLE_REQUIRED) - {"ablation_no_frequency_features_v5"}


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
        errors.append(f"{name}: confounded ablation changes {unexpected!r}")


def _validate_optimizer(config, name, errors):
    pairs = {
        (
            "pessimistic_ensemble_sac_v4",
            "pessimistic_ensemble_sac_lagrangian_v4",
        ): ("ensemble_mean_lcb", 10),
        (
            "standard_sac_v4",
            "standard_sac_lagrangian_v4",
        ): ("twin_min", 2),
    }
    pair = (_get(config, "upper.algorithm_id"), _get(config, "lower.algorithm_id"))
    contract = pairs.get(pair)
    if contract is None:
        errors.append(f"{name}: unsupported optimizer pair {pair!r}")
        return
    aggregation, ensemble_size = contract
    for level in ("upper", "lower"):
        if _get(config, f"{level}.critic_aggregation") != aggregation:
            errors.append(f"{name}: {level} aggregation must be {aggregation}")
        if int(_get(config, f"{level}.ensemble_size") or 0) != ensemble_size:
            errors.append(
                f"{name}: {level} ensemble_size must be {ensemble_size}")
        if aggregation == "twin_min":
            for field in ("resac_beta", "beta_ood", "weight_reg"):
                if float(_get(config, f"{level}.{field}") or 0.0) != 0.0:
                    errors.append(
                        f"{name}: {level}.{field} must be zero for twin-min")


def validate_config(config, name="config"):
    errors = []
    role = _get(config, "protocol.role")
    core = dict(CORE_REQUIRED)
    for path in ROLE_ALLOWED_DIFFS.get(role, set()):
        core.pop(path, None)
    _validate_expected(config, core, name, errors)
    if role not in ROLE_REQUIRED:
        errors.append(f"{name}: unsupported protocol.role {role!r}")
        return errors
    _validate_expected(config, ROLE_REQUIRED[role], name, errors)

    if role in SPLIT_ROLES and role != "ablation_raw_causal_history_v5":
        split_expected = {
            "protocol.frequency_contract": "split_features_v5",
            "frequency.enable": True,
            "frequency.method": "harmonic",
            "frequency.upper_features": True,
            "frequency.lower_features": True,
            "frequency.upper_mode": "low",
            "frequency.lower_mode": "high",
            "frequency.use_historical_prior": True,
        }
        for path in ROLE_ALLOWED_DIFFS.get(role, set()):
            split_expected.pop(path, None)
        _validate_expected(config, split_expected, name, errors)
    if _get(config, "lower.action_bins") != [
            0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]:
        errors.append(f"{name}: lower action bins violate the v5 contract")
    _validate_optimizer(config, name, errors)
    return errors


def _validate_matched_history_dimensions(configs, errors):
    main = DemandFrequencyTracker.from_config(configs[REFERENCE])
    raw_name = "F_freqduet_protocol_v5_rawhistory_hiro.yaml"
    raw = DemandFrequencyTracker.from_config(configs[raw_name])
    if main.upper_feature_dim != raw.upper_feature_dim:
        errors.append(
            "raw-history upper feature dimension is not matched to harmonic")
    if main.lower_feature_dim != raw.lower_feature_dim:
        errors.append(
            "raw-history lower feature dimension is not matched to harmonic")


def validate_all(config_names=None):
    names = list(config_names or REFERENCE_CONFIGS)
    errors = []
    reference = load_config(ROOT / "configs_freqduet" / REFERENCE)
    loaded = {REFERENCE: reference}
    for name in names:
        path = ROOT / "configs_freqduet" / name
        if not path.exists():
            errors.append(f"missing v5 config: {path}")
            continue
        config = load_config(path)
        loaded[name] = config
        errors.extend(validate_config(config, name=name))
        if name != REFERENCE:
            _validate_single_axis(config, reference, name, errors)
    raw_name = "F_freqduet_protocol_v5_rawhistory_hiro.yaml"
    if REFERENCE in loaded and raw_name in loaded:
        _validate_matched_history_dimensions(loaded, errors)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", default=REFERENCE_CONFIGS)
    args = parser.parse_args()
    errors = validate_all(args.configs)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"validated {len(args.configs)} FreqDuet v5 config(s)")


if __name__ == "__main__":
    main()
