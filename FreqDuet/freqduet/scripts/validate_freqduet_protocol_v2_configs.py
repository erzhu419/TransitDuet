#!/usr/bin/env python3
"""Validate protocol-v2 main, ablations, and structural candidates."""

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


MAIN = "F_freqduet_protocol_v2_main_hiro.yaml"
ABLATIONS = {
    "F_freqduet_protocol_v2_nofreq_hiro.yaml": {
        "upper.state_dim",
        "frequency.enable",
        "frequency.use_historical_prior",
        "frequency.upper_features",
        "frequency.lower_features",
        "frequency.promotion.enable",
        "frequency.drift_feedback.enable",
        "reward_attribution.enable",
        "leakage.enable",
    },
    "F_freqduet_protocol_v2_noprior_hiro.yaml": {
        "frequency.use_historical_prior",
    },
    "F_freqduet_protocol_v2_nopromotion_hiro.yaml": {
        "frequency.promotion.enable",
    },
    "F_freqduet_protocol_v2_noleakage_hiro.yaml": {
        "leakage.enable",
    },
    "F_freqduet_protocol_v2_nodriftfb_hiro.yaml": {
        "frequency.drift_feedback.enable",
    },
    "F_freqduet_protocol_v2_allfreq_hiro.yaml": {
        "frequency.upper_mode",
        "frequency.lower_mode",
    },
    "F_freqduet_protocol_v2_rawhistory_hiro.yaml": {
        "frequency.method",
        "frequency.upper_history_bins",
        "frequency.lower_history_bins",
        "frequency.od_features",
        "frequency.promotion.enable",
        "frequency.drift_feedback.enable",
        "reward_attribution.enable",
    },
}
STRUCTURAL = [
    "F_freqduet_protocol_v2_upperdisc_hiro.yaml",
    "F_freqduet_protocol_v2_upperhist_hiro.yaml",
    "F_freqduet_protocol_v2_upperdisc_hist_hiro.yaml",
    "F_freqduet_protocol_v2_uppercompact_hiro.yaml",
    "F_freqduet_protocol_v2_harmonicnb_hiro.yaml",
]
STRUCTURAL_ENABLED_ADDITIONS = {
    "F_freqduet_protocol_v2_upperdisc_hiro.yaml": set(),
    "F_freqduet_protocol_v2_upperhist_hiro.yaml": {
        "upper.state_history.enable",
    },
    "F_freqduet_protocol_v2_upperdisc_hist_hiro.yaml": {
        "upper.state_history.enable",
    },
    "F_freqduet_protocol_v2_uppercompact_hiro.yaml": set(),
    "F_freqduet_protocol_v2_harmonicnb_hiro.yaml": set(),
}
DOMAINS = [
    "F_freqduet_protocol_v2_gen_highnoise_main_hiro.yaml",
    "F_freqduet_protocol_v2_gen_odshift_main_hiro.yaml",
    "F_freqduet_protocol_v2_gen_rushshift_main_hiro.yaml",
]
MAIN_ENABLED_PATHS = {
    "frequency.enable",
    "frequency.promotion.enable",
    "frequency.drift_feedback.enable",
    "reward_attribution.enable",
    "leakage.enable",
    "upper.timetable_planner.enable",
    "upper.timetable_planner.terminal_headway_floor.enable",
}
ABLATION_DISABLED_PATHS = {
    "F_freqduet_protocol_v2_nofreq_hiro.yaml": {
        "frequency.enable",
        "frequency.promotion.enable",
        "frequency.drift_feedback.enable",
        "reward_attribution.enable",
        "leakage.enable",
    },
    "F_freqduet_protocol_v2_noprior_hiro.yaml": set(),
    "F_freqduet_protocol_v2_nopromotion_hiro.yaml": {
        "frequency.promotion.enable",
    },
    "F_freqduet_protocol_v2_noleakage_hiro.yaml": {
        "leakage.enable",
    },
    "F_freqduet_protocol_v2_nodriftfb_hiro.yaml": {
        "frequency.drift_feedback.enable",
    },
    "F_freqduet_protocol_v2_allfreq_hiro.yaml": set(),
    "F_freqduet_protocol_v2_rawhistory_hiro.yaml": {
        "frequency.promotion.enable",
        "frequency.drift_feedback.enable",
        "reward_attribution.enable",
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


def changed_paths(base: dict[str, Any], other: dict[str, Any]) -> set[str]:
    left = flatten(base)
    right = flatten(other)
    ignored = {"_name", "protocol.role"}
    return {
        key for key in set(left) | set(right)
        if key not in ignored and left.get(key) != right.get(key)
    }


def load(name: str) -> dict[str, Any]:
    return load_config(str(CONFIG_DIR / name))


def validate_learned_main(cfg: dict[str, Any], name: str) -> None:
    if cfg.get("protocol", {}).get("version") != "freqduet-eval-v2":
        raise AssertionError(f"{name}: protocol version is not v2")
    if cfg.get("env", {}).get("effective_trip_num") != "all":
        raise AssertionError(f"{name}: full timetable is not enabled")
    upper = cfg.get("upper", {})
    if upper.get("action_override", {}).get("enable", False):
        raise AssertionError(f"{name}: upper action override is enabled")
    for selector in [
        "counterfactual_action_selector",
        "snapshot_value_selector",
        "snapshot_action_value_selector",
        "residual_value_selector",
    ]:
        if upper.get(selector, {}).get("enable", False):
            raise AssertionError(f"{name}: {selector} is enabled")
    fixed = cfg.get("fixed_expert_selector", {})
    if fixed.get("enable", False):
        raise AssertionError(f"{name}: fixed expert selector is enabled")
    planner = upper.get("timetable_planner", {})
    if not planner.get("terminal_dispatch", False):
        raise AssertionError(f"{name}: executable terminal dispatch is disabled")
    if not (
        float(planner.get("terminal_shift_min_s", 0.0)) < 0.0
        < float(planner.get("terminal_shift_max_s", 0.0))
    ):
        raise AssertionError(f"{name}: terminal shift must allow advance and delay")
    weights = cfg.get("objective", {}).get("weights", {})
    if float(weights.get("incomplete_service", 0.0)) <= 0.0:
        raise AssertionError(
            f"{name}: incomplete-service penalty is not positive")


def enabled_paths(cfg: dict[str, Any]) -> set[str]:
    return {
        path for path, value in flatten(cfg).items()
        if path.endswith(".enable") and bool(value)
    }


def validate_enabled_modules(
    cfg: dict[str, Any], name: str, expected: set[str]
) -> None:
    actual = enabled_paths(cfg)
    if actual != expected:
        raise AssertionError(
            f"{name}: enabled-module mismatch; expected={sorted(expected)} "
            f"actual={sorted(actual)}")


def validate_all() -> dict[str, Any]:
    main = load(MAIN)
    validate_learned_main(main, MAIN)
    validate_enabled_modules(main, MAIN, MAIN_ENABLED_PATHS)
    ablation_diffs = {}
    for name, expected in ABLATIONS.items():
        cfg = load(name)
        validate_learned_main(cfg, name)
        actual = changed_paths(main, cfg)
        if actual != expected:
            raise AssertionError(
                f"{name}: confounded diff; expected={sorted(expected)} "
                f"actual={sorted(actual)}")
        ablation_diffs[name] = sorted(actual)
        validate_enabled_modules(
            cfg,
            name,
            MAIN_ENABLED_PATHS - ABLATION_DISABLED_PATHS[name],
        )
    for name in STRUCTURAL + DOMAINS:
        cfg = load(name)
        validate_learned_main(cfg, name)
        validate_enabled_modules(
            cfg,
            name,
            MAIN_ENABLED_PATHS
            | STRUCTURAL_ENABLED_ADDITIONS.get(name, set()),
        )
    return {
        "protocol_version": "freqduet-eval-v2",
        "main": MAIN,
        "ablations": ablation_diffs,
        "structural_candidates": STRUCTURAL,
        "domain_training_configs": DOMAINS,
        "status": "valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    result = validate_all()
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        destination = Path(args.json_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
