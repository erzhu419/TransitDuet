#!/usr/bin/env python3
"""Fail-closed semantic validator for the locked FreqDuet V6 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_freqduet_protocol_v2_matrix import (  # noqa: E402
    config_name,
    resolved_config,
    scenario_contract,
)


LOCKED_CONFIGS = [
    f"F_freqduet_protocol_v6_{name}_hiro"
    for name in (
        "main", "nofreq", "rawhistory", "allfreq", "upperonly",
        "loweronly", "swapped", "nobudget", "noguard",
        "noloadcost", "waitonlycredit", "csac",
    )
]
CONFIRMATION_CONFIGS = [
    "F_freqduet_protocol_v6_avlctx_hiro",
    "F_freqduet_protocol_v6_avlbal_w4_hiro",
    "F_freqduet_protocol_v6_avlcompact_hiro",
    "F_freqduet_protocol_v6_avlcompact_w2_hiro",
    "F_freqduet_protocol_v6_avlcompact_w4_hiro",
]
PROMOTED_CONFIGS = [
    "F_freqduet_protocol_v6_confirmed_main_hiro",
]
REGULARITY_POLICY_CONFIGS = [
    f"F_freqduet_protocol_v6_{prefix}_c{limit}_hiro"
    for prefix in ("actiondual", "w2actiondual")
    for limit in ("0005", "0010", "0020")
]
CONDITIONAL_ENTROPY_CONFIGS = [
    f"F_freqduet_protocol_v6_w2adent_e{fraction}_c{limit}_hiro"
    for limit in ("0010", "0020")
    for fraction in ("25", "50", "75")
]
NORMALIZED_REGULARITY_CONFIGS = [
    f"F_freqduet_protocol_v6_w2adnorm_l{initial}_e{fraction}_c{limit}_hiro"
    for limit, fraction in (("0010", "50"), ("0020", "25"))
    for initial in ("005", "010", "020")
]
ZERO_HOLD_REGRET_CONFIGS = [
    f"F_freqduet_protocol_v6_w2adregret_l{initial}_e25_r{limit}_hiro"
    for initial in ("001", "005")
    for limit in ("00025", "0005", "0010")
]
CAPACITY_GAIN_CONFIGS = [
    f"F_freqduet_protocol_v6_w2adcapgain_l001_e25_r00025_w{weight}_x{exponent}_hiro"
    for exponent in ("1", "2")
    for weight in ("0005", "0010", "0020")
]
EFFICIENCY_GAIN_CONFIGS = [
    f"F_freqduet_protocol_v6_w2adeffgain_l001_e25_r00025_w{weight}_b{penalty}_hiro"
    for penalty in ("05", "10", "20")
    for weight in ("0025", "0030", "0035")
]
FLEET_EFFICIENCY_GAIN_CONFIGS = [
    f"F_freqduet_protocol_v6_w2adfleetgain_l001_e25_r00025_"
    f"w{weight}_b{penalty}_p{exponent}_hiro"
    for exponent in ("1", "2")
    for penalty in ("05", "10")
    for weight in ("0020", "0025", "0030")
]
TARGET_PRESERVING_GAIN_CONFIGS = [
    f"F_freqduet_protocol_v6_w2adtpgain_l001_e25_r00025_"
    f"w{weight}_b{penalty}_t{target_exponent}_hiro"
    for target_exponent in ("0", "1")
    for penalty in ("05", "10")
    for weight in ("0020", "0025", "0030")
]
EXPERIMENTAL_CONFIGS = [
    "F_freqduet_protocol_v6_maskguard_hiro",
    "F_freqduet_protocol_v6_maskguard_nofreq_hiro",
    *[
        f"F_freqduet_protocol_v6_{kind}_c{limit}_{rate}_hiro"
        for kind in (
            "softdual", "softreg_w025", "softreg_w05", "softreg_w1")
        for limit in ("035", "030")
        for rate in ("l3e4", "l1e3")
    ],
    "F_freqduet_protocol_v6_departctx_hiro",
    *[
        f"F_freqduet_protocol_v6_{kind}_w{weight}_hiro"
        for kind in ("fwdadv", "avlbal")
        for weight in ("05", "1", "2", "4")
        if not (kind == "avlbal" and weight == "4")
    ],
    *[
        f"F_freqduet_protocol_v6_avlcompact_w{weight}_hiro"
        for weight in ("6", "8")
    ],
    *REGULARITY_POLICY_CONFIGS,
    *CONDITIONAL_ENTROPY_CONFIGS,
    *NORMALIZED_REGULARITY_CONFIGS,
    *ZERO_HOLD_REGRET_CONFIGS,
    *CAPACITY_GAIN_CONFIGS,
    *EFFICIENCY_GAIN_CONFIGS,
    *FLEET_EFFICIENCY_GAIN_CONFIGS,
    *TARGET_PRESERVING_GAIN_CONFIGS,
]


def validate(
    configs: list[str], *, allow_experimental: bool = False
) -> dict[str, object]:
    names = [config_name(value) for value in configs]
    if len(names) != len(set(names)):
        raise ValueError("V6 configs must be unique")
    allowed = set(LOCKED_CONFIGS).union(
        CONFIRMATION_CONFIGS, PROMOTED_CONFIGS)
    if allow_experimental:
        allowed.update(EXPERIMENTAL_CONFIGS)
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"unregistered V6 configs: {unknown}")
    historical_main = "F_freqduet_protocol_v6_main_hiro"
    promoted_mains = sorted(set(names).intersection(PROMOTED_CONFIGS))
    if len(promoted_mains) > 1:
        raise ValueError("V6 matrix includes multiple promoted main configs")
    if promoted_mains:
        canonical_main = promoted_mains[0]
    elif historical_main in names:
        canonical_main = historical_main
    else:
        raise ValueError(
            "V6 matrix must include a historical or promoted main config")

    resolved = {name: resolved_config(name) for name in names}
    scenario_hashes = set()
    for name, config in resolved.items():
        protocol = config.get("protocol", {}) or {}
        frequency = config.get("frequency", {}) or {}
        timetable = (config.get("upper", {}) or {}).get(
            "timetable_planner", {}) or {}
        guard = (config.get("lower", {}) or {}).get(
            "causal_holding_guard", {}) or {}
        lower = config.get("lower", {}) or {}
        regularity = lower.get(
            "causal_departure_regularity", {}) or {}
        regularity_policy = lower.get(
            "causal_regularity_policy", {}) or {}
        lower_context = (frequency.get("lower_context", {}) or {})
        required = {
            "protocol.version": (
                protocol.get("version"), "freqduet-eval-v6"),
            "protocol.objective_contract": (
                protocol.get("objective_contract"),
                "realized_restricted_passenger_journey_v6"),
            "frequency.forecast_mode": (
                frequency.get("forecast_mode"), "causal"),
            "frequency.observation_source": (
                frequency.get("observation_source"), "apc_boardings"),
            "timetable.terminal_schedule_mode": (
                timetable.get("terminal_schedule_mode"),
                "exact_headway_curve"),
            "guard.evidence_mode": (
                guard.get("evidence_mode"), "pre_action_departure_v6"),
        }
        mismatches = {
            key: {"observed": observed, "expected": expected}
            for key, (observed, expected) in required.items()
            if observed != expected
        }
        if mismatches:
            raise ValueError(f"{name}: V6 contract mismatch {mismatches}")
        if not bool(timetable.get("terminal_dispatch")):
            raise ValueError(f"{name}: executable terminal dispatch is disabled")
        if (bool(regularity.get("enable"))
                and regularity.get("evidence_mode")
                != "pre_action_departure_v6"):
            raise ValueError(
                f"{name}: soft regularity uses non-causal evidence")
        objective_mode = str(
            regularity.get("objective_mode", "cmdp_absolute"))
        if bool(regularity.get("enable")) and objective_mode in {
                "forward_incremental_reward",
                "avl_two_sided_incremental_reward"}:
            if bool(guard.get("enable")):
                raise ValueError(
                    f"{name}: incremental regularity must preserve noguard "
                    "action semantics")
            features = set(lower_context.get("features", []))
            forward_features = {
                "departure_gap_norm", "departure_gap_valid"}
            raw_two_sided_features = forward_features.union({
                "avl_follower_gap_norm", "avl_follower_gap_valid"})
            compact_two_sided_features = {
                "regularity_hold_target_norm",
                "regularity_hold_target_valid",
            }
            if objective_mode == "avl_two_sided_incremental_reward":
                causal_state_present = (
                    raw_two_sided_features.issubset(features)
                    or compact_two_sided_features.issubset(features)
                )
                required_description = (
                    f"raw={sorted(raw_two_sided_features)} or "
                    f"compact={sorted(compact_two_sided_features)}")
            else:
                causal_state_present = forward_features.issubset(features)
                required_description = sorted(forward_features)
            if not causal_state_present:
                raise ValueError(
                    f"{name}: incremental regularity lacks causal state "
                    f"features; requires {required_description}")
        if bool(regularity_policy.get("enable")):
            features = set(lower_context.get("features", []))
            compact_features = {
                "regularity_hold_target_norm",
                "regularity_hold_target_valid",
            }
            if bool(guard.get("enable")):
                raise ValueError(
                    f"{name}: regularity policy must preserve noguard "
                    "action semantics")
            if regularity_policy.get(
                    "evidence_mode") != "compact_causal_target_v7":
                raise ValueError(
                    f"{name}: regularity policy uses non-causal evidence")
            expected_policy_mode = (
                "analytic_two_sided_target_preserving_gain_regret_dual_v6"
                if name in TARGET_PRESERVING_GAIN_CONFIGS
                else
                "analytic_two_sided_fleet_efficiency_gain_regret_dual_v5"
                if name in FLEET_EFFICIENCY_GAIN_CONFIGS
                else
                "analytic_two_sided_efficiency_gain_regret_dual_v4"
                if name in EFFICIENCY_GAIN_CONFIGS
                else
                "analytic_two_sided_capacity_gain_regret_dual_v3"
                if name in CAPACITY_GAIN_CONFIGS
                else "analytic_two_sided_zero_hold_regret_dual_v2"
                if name in ZERO_HOLD_REGRET_CONFIGS
                else "analytic_two_sided_target_dual_v1")
            if regularity_policy.get("mode") != expected_policy_mode:
                raise ValueError(
                    f"{name}: regularity policy objective is not locked")
            if not compact_features.issubset(features):
                raise ValueError(
                    f"{name}: regularity policy lacks compact causal state")
            gain_configs = (
                CAPACITY_GAIN_CONFIGS
                + EFFICIENCY_GAIN_CONFIGS
                + FLEET_EFFICIENCY_GAIN_CONFIGS
                + TARGET_PRESERVING_GAIN_CONFIGS)
            if name in gain_configs:
                gain = regularity_policy.get(
                    "capacity_gated_gain", {}) or {}
                if "capacity" not in features:
                    raise ValueError(
                        f"{name}: capacity gain lacks causal capacity state")
                expected_gain_mode = (
                    "positive_zero_hold_target_preserving_gain_v4"
                    if name in TARGET_PRESERVING_GAIN_CONFIGS
                    else
                    "positive_zero_hold_fleet_efficiency_gain_v3"
                    if name in FLEET_EFFICIENCY_GAIN_CONFIGS
                    else
                    "positive_zero_hold_efficiency_gain_v2"
                    if name in EFFICIENCY_GAIN_CONFIGS
                    else "positive_zero_hold_gain_v1")
                if (gain.get("enable") is not True
                        or gain.get("mode") != expected_gain_mode):
                    raise ValueError(
                        f"{name}: capacity gain contract is not locked")
                allowed_weights = (
                    {0.02, 0.025, 0.03}
                    if name in (
                        FLEET_EFFICIENCY_GAIN_CONFIGS
                        + TARGET_PRESERVING_GAIN_CONFIGS)
                    else
                    {0.025, 0.03, 0.035}
                    if name in EFFICIENCY_GAIN_CONFIGS
                    else {0.005, 0.01, 0.02})
                if float(gain.get("weight", -1.0)) not in allowed_weights:
                    raise ValueError(
                        f"{name}: capacity gain weight is not registered")
                if float(gain.get("gain_scale", -1.0)) != 0.002:
                    raise ValueError(
                        f"{name}: capacity gain scale is not locked")
                allowed_exponents = (
                    {1.0} if name in (
                        EFFICIENCY_GAIN_CONFIGS
                        + FLEET_EFFICIENCY_GAIN_CONFIGS
                        + TARGET_PRESERVING_GAIN_CONFIGS)
                    else {1.0, 2.0})
                if (float(gain.get("capacity_exponent", -1.0))
                        not in allowed_exponents):
                    raise ValueError(
                        f"{name}: capacity gain exponent is not registered")
                expected_penalties = (
                    {0.5, 1.0}
                    if name in FLEET_EFFICIENCY_GAIN_CONFIGS
                    else
                    {0.5, 1.0, 2.0}
                    if name in EFFICIENCY_GAIN_CONFIGS else {0.0})
                if (float(gain.get("action_efficiency_penalty", 0.0))
                        not in expected_penalties):
                    raise ValueError(
                        f"{name}: action efficiency penalty is not registered")
                if name in (
                        FLEET_EFFICIENCY_GAIN_CONFIGS
                        + TARGET_PRESERVING_GAIN_CONFIGS):
                    if "fleet_utilization" not in features:
                        raise ValueError(
                            f"{name}: fleet efficiency lacks causal fleet state")
                    if float(gain.get("fleet_pressure_start", -1.0)) != 0.9:
                        raise ValueError(
                            f"{name}: fleet pressure start is not registered")
                    pressure_exponents = (
                        {1.0} if name in TARGET_PRESERVING_GAIN_CONFIGS
                        else {1.0, 2.0})
                    if (float(gain.get("fleet_pressure_full", -1.0)) != 1.0
                            or float(gain.get(
                                "fleet_pressure_exponent", -1.0))
                            not in pressure_exponents):
                        raise ValueError(
                            f"{name}: fleet pressure contract is not locked")
                if name in TARGET_PRESERVING_GAIN_CONFIGS:
                    if float(gain.get(
                            "opportunity_cost_penalty", -1.0)) not in {
                                0.5, 1.0}:
                        raise ValueError(
                            f"{name}: opportunity-cost penalty is not registered")
                    if float(gain.get(
                            "target_pressure_exponent", -1.0)) not in {
                                0.0, 1.0}:
                        raise ValueError(
                            f"{name}: target pressure exponent is not registered")
                if (float(regularity_policy.get("cost_limit", -1.0))
                        != 0.00025
                        or float(regularity_policy.get(
                            "initial_lambda", -1.0)) != 0.01):
                    raise ValueError(
                        f"{name}: capacity gain changed the V13 anchor")
            if not lower.get("action_bins"):
                raise ValueError(
                    f"{name}: regularity policy requires discrete actions")
            if not bool((lower.get("state_encoder", {}) or {}).get("enable")):
                raise ValueError(
                    f"{name}: regularity policy requires physical encoding")
            cost_limit = float(regularity_policy.get("cost_limit", -1.0))
            cost_cap = float(regularity_policy.get("cost_cap", -1.0))
            if not (0.0 <= cost_limit < cost_cap):
                raise ValueError(
                    f"{name}: invalid regularity policy cost contract")
            constraint_scale_mode = regularity_policy.get(
                "constraint_scale_mode", "raw_cost_v1")
            if constraint_scale_mode not in {
                    "raw_cost_v1", "cost_limit_ratio_v1"}:
                raise ValueError(
                    f"{name}: invalid regularity constraint scale mode")
            if (constraint_scale_mode == "cost_limit_ratio_v1"
                    and cost_limit <= 0.0):
                raise ValueError(
                    f"{name}: normalized regularity constraint needs a "
                    "positive cost limit")
            entropy = regularity_policy.get(
                "conditional_entropy", {}) or {}
            if bool(entropy.get("enable")):
                if entropy.get(
                        "mode") != "evidence_split_temperature_v1":
                    raise ValueError(
                        f"{name}: conditional entropy mode is not locked")
                target_fraction = float(
                    entropy.get("target_fraction", -1.0))
                if not (0.0 <= target_fraction < 0.98):
                    raise ValueError(
                        f"{name}: invalid conditional entropy target")
                entropy_lr = float(entropy.get("lr", -1.0))
                alpha_min = float(entropy.get("minimum_alpha", -1.0))
                alpha_max = float(entropy.get("maximum_alpha", -1.0))
                alpha_initial = float(entropy.get("initial_alpha", -1.0))
                if not (entropy_lr > 0.0 and 0.0 < alpha_min
                        <= alpha_initial <= alpha_max):
                    raise ValueError(
                        f"{name}: invalid conditional entropy optimizer")
        scenario_hashes.add(str(scenario_contract(name)["sha256"]))

    main = resolved[canonical_main]
    frequency = main["frequency"]
    timetable = main["upper"]["timetable_planner"]
    credit = main["upper"]["interval_credit"]
    main_checks = {
        "harmonic": frequency.get("method") == "harmonic",
        "historical_prior": bool(frequency.get("use_historical_prior")),
        "upper_low": frequency.get("upper_mode") == "low",
        "lower_high": frequency.get("lower_mode") == "high",
        "rolling_budget": (
            timetable.get("headway_budget_mode")
            == "rolling_zero_sum_delta_v6"),
        "budget_matches_replan": (
            float(timetable.get("headway_budget_window_s", -1.0))
            == float(timetable.get("replan_interval_s", -2.0))),
        "additive_credit": credit.get("assignment_mode") == "additive",
        "onboard_credit": float(credit["weights"].get("onboard", 0.0)) > 0,
        "backlog_credit": (
            float(credit["weights"].get("dispatch_backlog", 0.0)) > 0),
        "no_directional_half_fleet_penalty": (
            float(credit["weights"].get("fleet", -1.0)) == 0.0),
    }
    if not all(main_checks.values()):
        raise ValueError(f"V6 main contract failed: {main_checks}")
    if len(scenario_hashes) != 1:
        raise ValueError("V6 configs do not share one exogenous scenario contract")
    return {
        "status": "valid",
        "protocol_version": "freqduet-eval-v6",
        "configs": names,
        "canonical_main": canonical_main,
        "scenario_contract_sha256": next(iter(scenario_hashes)),
        "main_checks": main_checks,
        "experimental_configs": sorted(
            set(names).intersection(EXPERIMENTAL_CONFIGS)),
        "confirmation_configs": sorted(
            set(names).intersection(CONFIRMATION_CONFIGS)),
        "promoted_configs": sorted(
            set(names).intersection(PROMOTED_CONFIGS)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Permit registered exploratory configs without promoting them.",
    )
    parser.add_argument("configs", nargs="*", default=LOCKED_CONFIGS)
    args = parser.parse_args()
    print(json.dumps(validate(
        args.configs,
        allow_experimental=args.allow_experimental,
    ), indent=2))


if __name__ == "__main__":
    main()
