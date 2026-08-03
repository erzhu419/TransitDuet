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


def _get(config, path):
    value = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


REQUIRED = {
    "protocol.version": "freqduet-eval-v4",
    "randomness.mode": "isolated_streams_v4",
    "env.fleet_inventory_mode": "fixed_pool",
    "env.upper_fleet_state_mode": "fixed_pool_readiness_v4",
    "env.observation_contract": "deployable_apc_avl_v4",
    "env.headway_reward_mode": "forward_event_only",
    "upper.algorithm_id": "pessimistic_ensemble_sac_v4",
    "upper.temperature_contract": "bounded_log_parameter_v4",
    "upper.timetable_planner.terminal_schedule_mode": "exact_headway_curve",
    "upper.timetable_planner.terminal_dispatch": True,
    "upper.timetable_planner.terminal_headway_floor.enable": False,
    "upper.interval_credit.enable": True,
    "upper.interval_credit.wait_ownership": "frozen_low_frequency",
    "lower.algorithm_id": "pessimistic_ensemble_sac_lagrangian_v4",
    "lower.observation_contract": "deployable_apc_avl_v4",
    "lower.headway_reward_mode": "forward_event_only",
    "lower.unobserved_action_mode": "zero",
    "lower.temperature_contract": "bounded_log_parameter_v4",
    "lower.entropy_action_coordinates": "normalized_unit_interval",
    "lower.cost_limit_semantics": "per_decision_rate",
    "lower.state_encoder.input_schema": "causal_forward_v4",
    "frequency.forecast_mode": "causal",
    "frequency.observation_source": "apc_boardings",
    "reward_attribution.assignment_mode": "frozen_passenger",
    "reward_attribution.upper_wait_weight": 0.0,
    "coupling.tpc.target_distribution": "bounded_logistic_normal_v4",
    "training.checkpoint_contract": "exact_training_state_v4",
}


def validate_config(config, name="config"):
    errors = []
    for path, expected in REQUIRED.items():
        observed = _get(config, path)
        if observed != expected:
            errors.append(
                f"{name}: {path}={observed!r}, expected {expected!r}")
    if not bool(_get(config, "frequency.enable")):
        errors.append(f"{name}: frequency.enable must be true")
    if not bool(_get(config, "reward_attribution.enable")):
        errors.append(f"{name}: reward_attribution.enable must be true")
    if _get(config, "coupling.tpc.enable") and _get(
            config, "coupling.tpc.target_distribution"
    ) != "bounded_logistic_normal_v4":
        errors.append(f"{name}: enabled TPC must use bounded logistic-normal")
    return errors


def validate_all(config_names=None):
    names = list(config_names or [REFERENCE])
    errors = []
    for name in names:
        path = ROOT / "configs_freqduet" / name
        if not path.exists():
            errors.append(f"missing v4 config: {path}")
            continue
        errors.extend(validate_config(load_config(path), name=name))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=[REFERENCE])
    args = parser.parse_args()
    errors = validate_all(args.configs)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f"validated {len(args.configs)} FreqDuet v4 config(s)")


if __name__ == "__main__":
    main()
