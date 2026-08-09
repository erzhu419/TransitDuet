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
]


def validate(
    configs: list[str], *, allow_experimental: bool = False
) -> dict[str, object]:
    names = [config_name(value) for value in configs]
    if len(names) != len(set(names)):
        raise ValueError("V6 configs must be unique")
    allowed = set(LOCKED_CONFIGS).union(CONFIRMATION_CONFIGS)
    if allow_experimental:
        allowed.update(EXPERIMENTAL_CONFIGS)
    unknown = sorted(set(names) - allowed)
    if unknown:
        raise ValueError(f"unregistered V6 configs: {unknown}")
    if "F_freqduet_protocol_v6_main_hiro" not in names:
        raise ValueError("V6 matrix must include the locked main config")

    resolved = {name: resolved_config(name) for name in names}
    scenario_hashes = set()
    for name, config in resolved.items():
        protocol = config.get("protocol", {}) or {}
        frequency = config.get("frequency", {}) or {}
        timetable = (config.get("upper", {}) or {}).get(
            "timetable_planner", {}) or {}
        guard = (config.get("lower", {}) or {}).get(
            "causal_holding_guard", {}) or {}
        regularity = (config.get("lower", {}) or {}).get(
            "causal_departure_regularity", {}) or {}
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
        scenario_hashes.add(str(scenario_contract(name)["sha256"]))

    main = resolved["F_freqduet_protocol_v6_main_hiro"]
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
        "scenario_contract_sha256": next(iter(scenario_hashes)),
        "main_checks": main_checks,
        "experimental_configs": sorted(
            set(names).intersection(EXPERIMENTAL_CONFIGS)),
        "confirmation_configs": sorted(
            set(names).intersection(CONFIRMATION_CONFIGS)),
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
