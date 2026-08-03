#!/usr/bin/env python3
"""Validate the protocol-v3 physical-contract selection matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs_freqduet"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runner_v3 import load_config


REFERENCE = "F_freqduet_protocol_v3_compact_b30_hiro.yaml"
VARIANT_DIFFS = {
    "F_freqduet_protocol_v3_global3_b30_hiro.yaml": {
        "_name", "protocol.role",
        "upper.timetable_planner.basis_per_direction",
        "upper.timetable_planner.shared_directions",
        "upper.timetable_planner.plan_all_directions",
    },
    "F_freqduet_protocol_v3_local4_b30_hiro.yaml": {
        "_name", "protocol.role",
        "upper.timetable_planner.basis_per_direction",
    },
    "F_freqduet_protocol_v3_compact_b15_hiro.yaml": {
        "_name", "protocol.role", "leakage.lower_drift_budget_s",
    },
    "F_freqduet_protocol_v3_compact_b60_hiro.yaml": {
        "_name", "protocol.role", "leakage.lower_drift_budget_s",
    },
    "F_freqduet_protocol_v3_compact_b90_hiro.yaml": {
        "_name", "protocol.role", "leakage.lower_drift_budget_s",
    },
    "F_freqduet_protocol_v3_compact_rolling_b30_hiro.yaml": {
        "_name", "protocol.role", "leakage.lower_drift_signal_mode",
    },
    "F_freqduet_protocol_v3_compact_nolowerdrift_b30_hiro.yaml": {
        "_name", "protocol.role", "leakage.lower_drift_penalty",
        "leakage.lower_drift_cost_weight",
    },
    "F_freqduet_protocol_v3_compact_nodriftfb_b30_hiro.yaml": {
        "_name", "protocol.role", "frequency.drift_feedback.enable",
    },
    "F_freqduet_protocol_v3_compact_spatial_b30_hiro.yaml": {
        "_name", "protocol.role", "lower.headway_state_mode",
    },
    "F_freqduet_protocol_v3_compact_inferredtarget_b30_hiro.yaml": {
        "_name", "protocol.role", "lower.state_encoder.input_schema",
    },
    "F_freqduet_protocol_v3_compact_observedtrain_b30_hiro.yaml": {
        "_name", "protocol.role", "objective.wait_metric",
    },
    "F_freqduet_protocol_v3_compact_legacycredit_b30_hiro.yaml": {
        "_name", "protocol.role",
        "upper.credit_assignment.system_reward_mode",
        "upper.credit_assignment.system_reward_weight",
        "upper.credit_assignment.gap_credit_mode",
        "upper.credit_assignment.gap_credit_weight",
        "upper.credit_assignment.gap_credit_clip",
        "upper.credit_assignment.reliability_reward_mode",
        "upper.credit_assignment.reliability_reward_weight",
        "upper.interval_credit.enable",
    },
}


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(flatten(item, path))
    return result


def changed_paths(reference: dict, candidate: dict) -> set[str]:
    left = flatten(reference)
    right = flatten(candidate)
    return {
        key for key in set(left) | set(right)
        if left.get(key) != right.get(key)
    }


def validate_common_contract(config: dict, source: str) -> None:
    checks = {
        "protocol.version": "freqduet-eval-v3",
        "upper.transition_stream_mode": "planner_key",
        "upper.holding_state.source": "trip_lifecycle",
        "upper.holding_state.episode_local": True,
        "lower.trip_boundary_mode": "reset",
        "lower.terminal_action_mode": "transition",
        "lower.holding_action_trace_mode": "all_decisions",
        "lower.unobserved_action_mode": "zero",
        "frequency.forecast_mode": "causal",
        "fixed_expert_selector.enable": False,
    }
    values = flatten(config)
    for path, expected in checks.items():
        if values.get(path) != expected:
            raise ValueError(
                f"{source}: {path}={values.get(path)!r}, expected {expected!r}")


def validate_all() -> dict[str, Any]:
    reference = load_config(CONFIG_DIR / REFERENCE)
    validate_common_contract(reference, REFERENCE)
    validated = [Path(REFERENCE).stem]
    for filename, allowed in VARIANT_DIFFS.items():
        candidate = load_config(CONFIG_DIR / filename)
        validate_common_contract(candidate, filename)
        actual = changed_paths(reference, candidate)
        if actual != allowed:
            raise ValueError(
                f"{filename}: unexpected resolved differences; "
                f"actual={sorted(actual)}, expected={sorted(allowed)}")
        validated.append(Path(filename).stem)
    return {
        "status": "valid",
        "protocol_version": "freqduet-eval-v3",
        "reference": Path(REFERENCE).stem,
        "configs": validated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = validate_all()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
