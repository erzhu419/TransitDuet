#!/usr/bin/env python3
"""Audit the preregistered V21 HF regularity-gain floor screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_capacity_gain_screen import (
    CURRENT_MAIN,
    HARD_MAIN,
    MATCHED_CONTEXT,
    REFERENCE,
    SAME_ENTROPY,
    V13_ANCHOR,
    _finite,
    _mean,
    _paired_delta,
    _rows,
)


V13_ZERO_HOLD_ADVANTAGE = (
    "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_qadv0_hiro"
)
V20_SCALAR_B080 = (
    "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_"
    "paxdual_b080_hiro"
)
V20_QADV_B080 = (
    "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_"
    "paxdual_b080_qadv0_hiro"
)
FLOOR_ONLY_SPECS = [
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_"
        "f050_h000_qadv0_hiro",
        0.50,
        0.00,
    ),
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_"
        "f040_h025_qadv0_hiro",
        0.40,
        0.25,
    ),
]
CANDIDATE_SPECS = [
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_"
        "f050_h000_pax080_qadv0_hiro",
        0.50,
        0.00,
    ),
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_"
        "f030_h030_pax080_qadv0_hiro",
        0.30,
        0.30,
    ),
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_"
        "f040_h025_pax080_qadv0_hiro",
        0.40,
        0.25,
    ),
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_"
        "f050_h020_pax080_qadv0_hiro",
        0.50,
        0.20,
    ),
]
FLOOR_CONFIGS = [name for name, _, _ in FLOOR_ONLY_SPECS + CANDIDATE_SPECS]
CANDIDATES = [name for name, _, _ in CANDIDATE_SPECS]
CONFIGS = [
    HARD_MAIN,
    REFERENCE,
    MATCHED_CONTEXT,
    CURRENT_MAIN,
    SAME_ENTROPY,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    V20_SCALAR_B080,
    V20_QADV_B080,
    *[name for name, _, _ in FLOOR_ONLY_SPECS],
    *CANDIDATES,
]
TRAIN_SEEDS = [26013, 26031, 26053, 26077]
EVAL_SEEDS = [59017, 59041, 59059, 59083]
PRIORITY = [
    CANDIDATE_SPECS[1][0],
    CANDIDATE_SPECS[2][0],
    CANDIDATE_SPECS[3][0],
    CANDIDATE_SPECS[0][0],
]


def _paired_metrics(
    per_eval: pd.DataFrame,
    candidate: str,
    reference: str,
) -> tuple[dict[str, float], list[int]]:
    deltas: dict[str, float] = {}
    counts: list[int] = []
    for label, metric in {
        "headway_cv": "headway_cv",
        "journey_min": "restricted_total_journey_horizon_min",
    }.items():
        deltas[label], count = _paired_delta(
            per_eval, candidate, reference, metric)
        counts.append(count)
    return deltas, counts


def _floor_mechanism_checks(
    rows: pd.DataFrame,
    *,
    base_fraction: float,
    hf_increment: float,
    passenger_enabled: bool,
) -> dict[str, bool]:
    numeric = lambda column: pd.to_numeric(rows[column], errors="coerce")
    policy_frozen = numeric("lower_policy_frozen")
    critic_frozen = numeric("lower_critic_frozen")
    upper_frozen = numeric("upper_policy_frozen")
    adjustment = numeric("lower_causal_guard_adjustment_mean_s")
    evidence = numeric("lower_regularity_policy_evidence_valid_mean")
    regret = numeric("lower_regularity_policy_action_regret_mean")
    regularity_lambda = numeric("lower_regularity_lambda")
    required = numeric(
        "lower_regularity_gain_floor_required_fraction_mean")
    hf_pressure = numeric("lower_regularity_gain_floor_hf_pressure_mean")
    expected_gain = numeric(
        "lower_regularity_gain_floor_expected_gain_fraction_mean")
    selected_gain = numeric(
        "lower_regularity_gain_floor_selected_gain_fraction_mean")
    expected_shortfall = numeric(
        "lower_regularity_gain_floor_expected_shortfall_mean")
    selected_shortfall = numeric(
        "lower_regularity_gain_floor_selected_shortfall_mean")
    eligible = numeric("lower_regularity_gain_floor_eligible_mean")
    actor_floor_fields = [
        numeric("lower_regularity_gain_floor_actor_required_fraction_mean"),
        numeric("lower_regularity_gain_floor_actor_hf_energy_mean"),
        numeric("lower_regularity_gain_floor_actor_hf_pressure_mean"),
        numeric(
            "lower_regularity_gain_floor_actor_expected_gain_fraction_mean"),
        numeric(
            "lower_regularity_gain_floor_actor_expected_shortfall_mean"),
        numeric("lower_regularity_gain_floor_actor_eligible_fraction"),
    ]
    passenger_flag = numeric(
        "lower_regularity_passenger_holding_enabled")
    passenger_expected = numeric(
        "lower_regularity_passenger_expected_cost_mean")
    passenger_selected = numeric(
        "lower_regularity_passenger_selected_cost_mean")
    passenger_load = numeric("lower_regularity_passenger_load_mean")
    passenger_lambda = numeric("lower_regularity_passenger_lambda")

    passenger_contract = bool(
        (passenger_flag == float(passenger_enabled)).all())
    passenger_budget = True
    passenger_telemetry = True
    passenger_dual = True
    if passenger_enabled:
        passenger_contract = bool(
            passenger_contract
            and (rows[
                "lower_regularity_passenger_holding_mode"].astype(str)
                == "causal_apc_person_delay_dual_v1").all()
            and (rows[
                "lower_regularity_passenger_constraint_scale_mode"].astype(str)
                == "cost_limit_ratio_v1").all()
            and np.allclose(
                numeric("lower_regularity_passenger_initial_lambda"), 0.01)
            and np.allclose(
                numeric("lower_regularity_passenger_cost_limit"), 0.08)
            and np.allclose(
                numeric("lower_regularity_passenger_scaled_limit"), 1.0))
        passenger_budget = bool(
            _finite(passenger_expected)
            and (passenger_expected >= 0.0).all()
            and (passenger_expected <= 0.08 + 1e-12).all())
        passenger_telemetry = bool(
            _finite(passenger_selected) and (passenger_selected >= 0.0).all()
            and _finite(passenger_load) and (passenger_load > 0.0).all()
            and (passenger_load <= 1.0 + 1e-12).all())
        passenger_dual = bool(
            _finite(passenger_lambda)
            and (passenger_lambda >= 0.0001 - 1e-12).all()
            and (passenger_lambda <= 2.0 + 1e-12).all())
    else:
        passenger_contract = bool(
            passenger_contract
            and (passenger_expected.abs() <= 1e-12).all()
            and (passenger_selected.abs() <= 1e-12).all()
            and (passenger_load.abs() <= 1e-12).all())

    return {
        "zero_hold_advantage_critic_locked": bool(
            (rows["lower_discrete_critic"].astype(str)
             == "zero_hold_advantage").all()),
        "frozen_evaluation_locked": bool(
            _finite(policy_frozen) and (policy_frozen == 1.0).all()
            and _finite(critic_frozen) and (critic_frozen == 1.0).all()
            and _finite(upper_frozen) and (upper_frozen == 1.0).all()
            and all(_finite(field) and (field.abs() <= 1e-12).all()
                    for field in actor_floor_fields)),
        "zero_execution_adjustment": bool(
            _finite(adjustment) and (adjustment.abs() <= 1e-12).all()),
        "gain_floor_contract_locked": bool(
            (numeric("lower_regularity_policy_enabled") == 1.0).all()
            and (rows["lower_regularity_policy_mode"].astype(str)
                 == "analytic_two_sided_hf_gain_floor_dual_v8").all()
            and (rows[
                "lower_regularity_policy_constraint_cost_mode"].astype(str)
                 == "hf_relative_gain_shortfall_v3").all()
            and (rows[
                "lower_regularity_policy_constraint_scale_mode"].astype(str)
                 == "cost_limit_ratio_v1").all()
            and np.allclose(
                numeric("lower_regularity_policy_initial_lambda"), 0.01)
            and np.allclose(
                numeric("lower_regularity_policy_cost_limit"), 0.05)
            and np.allclose(
                numeric("lower_regularity_policy_scaled_limit"), 1.0)
            and (numeric("lower_regularity_gain_floor_enabled") == 1.0).all()
            and (rows["lower_regularity_gain_floor_mode"].astype(str)
                 == "causal_hf_relative_gain_floor_v1").all()
            and np.allclose(
                numeric("lower_regularity_gain_floor_base_fraction"),
                base_fraction)
            and np.allclose(
                numeric("lower_regularity_gain_floor_hf_increment"),
                hf_increment)
            and np.allclose(
                numeric("lower_regularity_gain_floor_hf_energy_scale"), 0.04)
            and np.allclose(
                numeric("lower_regularity_gain_floor_hf_energy_exponent"),
                1.0)),
        "passive_floor_telemetry_active": bool(
            _finite(eligible) and (eligible > 0.0).all()
            and (eligible <= 1.0 + 1e-12).all()
            and _finite(hf_pressure) and (hf_pressure >= 0.0).all()
            and (hf_pressure <= 1.0 + 1e-12).all()
            and _finite(required)
            and np.allclose(
                required,
                base_fraction + hf_increment * hf_pressure,
                atol=1e-6,
                rtol=0.0)
            and _finite(expected_gain) and (expected_gain >= 0.0).all()
            and (expected_gain <= 1.0 + 1e-12).all()
            and _finite(selected_gain) and (selected_gain >= 0.0).all()
            and (selected_gain <= 1.0 + 1e-12).all()
            and _finite(selected_shortfall)
            and (selected_shortfall >= 0.0).all()),
        "hf_conditioning_active_when_registered": bool(
            hf_increment == 0.0
            or (_finite(hf_pressure)
                and (hf_pressure > 0.0).all()
                and float(hf_pressure.max() - hf_pressure.min()) > 1e-8)),
        "expected_floor_budget_satisfied_every_rollout": bool(
            _finite(expected_shortfall)
            and (expected_shortfall >= 0.0).all()
            and (expected_shortfall <= 0.05 + 1e-12).all()),
        "regularity_dual_finite_and_bounded": bool(
            _finite(regularity_lambda)
            and (regularity_lambda >= 0.0001 - 1e-12).all()
            and (regularity_lambda <= 2.0 + 1e-12).all()),
        "regret_limit_satisfied_every_rollout": bool(
            _finite(regret) and (regret <= 0.00025 + 1e-12).all()),
        "causal_evidence_coverage": bool(
            _finite(evidence) and (evidence >= 0.50).all()),
        "passenger_contract_locked": passenger_contract,
        "expected_passenger_budget_satisfied_every_rollout": passenger_budget,
        "passive_passenger_telemetry_active": passenger_telemetry,
        "passenger_dual_finite_and_bounded": passenger_dual,
    }


def evaluate_gain_floor_screen(aggregate_dir: Path) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    paths = {
        name: aggregate_dir / filename
        for name, filename in {
            "manifest": "matrix_manifest.json",
            "per_eval": "frozen_per_eval.csv",
            "summary": "frozen_summary.csv",
            "paired": "frozen_paired_deltas.csv",
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V21 aggregate artifacts: {missing}")

    manifest = json.loads(paths["manifest"].read_text())
    per_eval = pd.read_csv(paths["per_eval"])
    paired = pd.read_csv(paths["paired"])
    expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
    expected_rollouts = len(CONFIGS) * expected_pairs
    strict_checks = {
        "strict_complete": manifest.get("strict_complete") is True,
        "run_manifests_verified": manifest.get(
            "run_manifests_verified") is True,
        "common_random_numbers_verified": manifest.get(
            "common_random_numbers_verified") is True,
        "exploratory_stage": (
            manifest.get("stage") == "exploratory"
            and manifest.get("independent_confirmation") is False),
        "exact_configs": manifest.get("configs") == CONFIGS,
        "exact_train_seeds": manifest.get("train_seeds") == TRAIN_SEEDS,
        "exact_eval_seeds": manifest.get("eval_seeds") == EVAL_SEEDS,
        "forty_training_episodes": (
            manifest.get("train_episodes") == 40
            and manifest.get("checkpoint_ep") == 39),
        "reference_is_v13_anchor": manifest.get("reference") == V13_ANCHOR,
        "source_is_clean": manifest.get(
            "run_git_provenance", {}).get("tracked_dirty") is False,
        "expected_rollouts": (
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts),
        "unique_rollouts": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
    }
    if not all(strict_checks.values()):
        raise ValueError(f"V21 strict checks failed: {strict_checks}")

    required_columns = {
        "config", "train_seed", "eval_seed", "headway_cv",
        "restricted_total_journey_horizon_min", "holding_vehicle_seconds",
        "fleet_denied_dispatch_events", "lower_action_mean",
        "lower_discrete_critic", "lower_policy_frozen",
        "lower_critic_frozen", "upper_policy_frozen",
        "lower_causal_guard_adjustment_mean_s",
        "lower_regularity_policy_enabled", "lower_regularity_policy_mode",
        "lower_regularity_policy_constraint_cost_mode",
        "lower_regularity_policy_constraint_scale_mode",
        "lower_regularity_policy_initial_lambda",
        "lower_regularity_policy_cost_limit",
        "lower_regularity_policy_scaled_limit",
        "lower_regularity_policy_action_regret_mean",
        "lower_regularity_policy_evidence_valid_mean",
        "lower_regularity_lambda",
        "lower_regularity_gain_floor_enabled",
        "lower_regularity_gain_floor_mode",
        "lower_regularity_gain_floor_base_fraction",
        "lower_regularity_gain_floor_hf_increment",
        "lower_regularity_gain_floor_hf_energy_scale",
        "lower_regularity_gain_floor_hf_energy_exponent",
        "lower_regularity_gain_floor_actor_required_fraction_mean",
        "lower_regularity_gain_floor_actor_hf_energy_mean",
        "lower_regularity_gain_floor_actor_hf_pressure_mean",
        "lower_regularity_gain_floor_actor_expected_gain_fraction_mean",
        "lower_regularity_gain_floor_actor_expected_shortfall_mean",
        "lower_regularity_gain_floor_actor_eligible_fraction",
        "lower_regularity_gain_floor_required_fraction_mean",
        "lower_regularity_gain_floor_hf_pressure_mean",
        "lower_regularity_gain_floor_expected_gain_fraction_mean",
        "lower_regularity_gain_floor_selected_gain_fraction_mean",
        "lower_regularity_gain_floor_expected_shortfall_mean",
        "lower_regularity_gain_floor_selected_shortfall_mean",
        "lower_regularity_gain_floor_eligible_mean",
        "lower_regularity_passenger_holding_enabled",
        "lower_regularity_passenger_holding_mode",
        "lower_regularity_passenger_constraint_scale_mode",
        "lower_regularity_passenger_initial_lambda",
        "lower_regularity_passenger_cost_limit",
        "lower_regularity_passenger_scaled_limit",
        "lower_regularity_passenger_expected_cost_mean",
        "lower_regularity_passenger_selected_cost_mean",
        "lower_regularity_passenger_load_mean",
        "lower_regularity_passenger_lambda",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(f"V21 metrics are missing: {missing_columns}")

    nonfloor_contract_locked = all(
        (pd.to_numeric(
            _rows(per_eval, config)["lower_regularity_gain_floor_enabled"],
            errors="coerce") == 0.0).all()
        for config in CONFIGS if config not in FLOOR_CONFIGS
    )
    floor_control_results = []
    for config, base_fraction, hf_increment in FLOOR_ONLY_SPECS:
        rows = _rows(per_eval, config)
        checks = _floor_mechanism_checks(
            rows,
            base_fraction=base_fraction,
            hf_increment=hf_increment,
            passenger_enabled=False,
        )
        checks["nonfloor_controls_locked"] = nonfloor_contract_locked
        floor_control_results.append({
            "config": config,
            "base_fraction": base_fraction,
            "hf_increment": hf_increment,
            "mechanism_checks": checks,
            "passes": all(checks.values()),
        })

    reference_holding = _mean(
        per_eval, REFERENCE, "holding_vehicle_seconds")
    reference_denied = _mean(
        per_eval, REFERENCE, "fleet_denied_dispatch_events")
    hard_holding = _mean(per_eval, HARD_MAIN, "holding_vehicle_seconds")
    hard_denied = _mean(
        per_eval, HARD_MAIN, "fleet_denied_dispatch_events")
    holding_limit = reference_holding + 0.10 * max(
        hard_holding - reference_holding, 0.0)
    denied_limit = reference_denied + 0.10 * max(
        hard_denied - reference_denied, 0.0)

    candidate_results = []
    controls = (
        ("noguard", REFERENCE),
        ("current", CURRENT_MAIN),
        ("v11", SAME_ENTROPY),
    )
    for candidate, base_fraction, hf_increment in CANDIDATE_SPECS:
        rows = _rows(per_eval, candidate)
        if len(rows) != expected_pairs:
            raise ValueError(f"incomplete V21 candidate: {candidate}")
        aggregate_pair = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == V13_ANCHOR)]
        if len(aggregate_pair) != 1 or int(
                aggregate_pair.iloc[0]["n_pairs"]) != expected_pairs:
            raise ValueError(f"incomplete V21 aggregate pair: {candidate}")

        v13_delta, pair_counts = _paired_metrics(
            per_eval, candidate, V13_ANCHOR)
        qadv_delta, counts = _paired_metrics(
            per_eval, candidate, V13_ZERO_HOLD_ADVANTAGE)
        pair_counts.extend(counts)
        v20_delta, counts = _paired_metrics(
            per_eval, candidate, V20_QADV_B080)
        pair_counts.extend(counts)
        comparisons = {}
        for label, control in controls:
            comparisons[label], counts = _paired_metrics(
                per_eval, candidate, control)
            pair_counts.extend(counts)

        numeric = lambda column: pd.to_numeric(rows[column], errors="coerce")
        candidate_holding = float(numeric(
            "holding_vehicle_seconds").mean())
        candidate_denied = float(numeric(
            "fleet_denied_dispatch_events").mean())
        candidate_action = float(numeric("lower_action_mean").mean())
        mechanism_checks = _floor_mechanism_checks(
            rows,
            base_fraction=base_fraction,
            hf_increment=hf_increment,
            passenger_enabled=True,
        )
        mechanism_checks.update({
            "paired_rollouts_complete": all(
                count == expected_pairs for count in pair_counts),
            "nonfloor_controls_locked": nonfloor_contract_locked,
            "floor_only_controls_valid": all(
                result["passes"] for result in floor_control_results),
        })
        outcome_checks = {
            "journey_improves_v13_scalar_anchor": (
                v13_delta["journey_min"] <= -0.05),
            "cv_improves_v13_scalar_anchor": (
                v13_delta["headway_cv"] <= -0.001),
            "holding_does_not_increase_vs_v13": (
                candidate_holding <= _mean(
                    per_eval, V13_ANCHOR, "holding_vehicle_seconds")),
            "denied_does_not_increase_vs_v13": (
                candidate_denied <= _mean(
                    per_eval, V13_ANCHOR,
                    "fleet_denied_dispatch_events")),
            "action_does_not_increase_vs_v13": (
                candidate_action <= _mean(
                    per_eval, V13_ANCHOR, "lower_action_mean")),
            "journey_improves_unbounded_qadv": (
                qadv_delta["journey_min"] <= -0.05),
            "cv_improves_unbounded_qadv": (
                qadv_delta["headway_cv"] <= -0.001),
            "cv_recovers_v20_passenger_collapse": (
                v20_delta["headway_cv"] <= -0.020),
            "journey_noninferior_to_v20_passenger": (
                v20_delta["journey_min"] <= 0.20),
            "journey_beats_noguard": (
                comparisons["noguard"]["journey_min"] <= -0.25),
            "cv_beats_noguard": (
                comparisons["noguard"]["headway_cv"] <= -0.030),
            "journey_beats_current_main": (
                comparisons["current"]["journey_min"] <= -0.50),
            "cv_noninferior_to_current_main": (
                comparisons["current"]["headway_cv"] <= 0.005),
            "journey_beats_v11": (
                comparisons["v11"]["journey_min"] <= -0.25),
            "cv_noninferior_to_v11": (
                comparisons["v11"]["headway_cv"] <= 0.005),
            "historical_holding_limit": candidate_holding <= holding_limit,
            "historical_denied_limit": candidate_denied <= denied_limit,
        }
        candidate_results.append({
            "config": candidate,
            "base_fraction": base_fraction,
            "hf_increment": hf_increment,
            "mechanism_checks": mechanism_checks,
            "outcome_checks": outcome_checks,
            "passes": (
                all(mechanism_checks.values())
                and all(outcome_checks.values())),
            "v13_delta": v13_delta,
            "unbounded_qadv_delta": qadv_delta,
            "v20_passenger_delta": v20_delta,
            "control_deltas": comparisons,
            "holding_vehicle_seconds_mean": candidate_holding,
            "fleet_denied_dispatch_events_mean": candidate_denied,
            "lower_action_mean_s": candidate_action,
            "expected_floor_shortfall_max": float(numeric(
                "lower_regularity_gain_floor_expected_shortfall_mean").max()),
            "expected_passenger_cost_max": float(numeric(
                "lower_regularity_passenger_expected_cost_mean").max()),
        })

    passing = {
        result["config"] for result in candidate_results if result["passes"]}
    selected = next((name for name in PRIORITY if name in passing), None)
    return {
        "gate_version": "freqduet-v21-hf-regularity-gain-floor-screen-v1",
        "status": (
            "exploratory_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "strict_checks": strict_checks,
        "resource_limits": {
            "holding_vehicle_seconds_mean_max": holding_limit,
            "fleet_denied_dispatch_events_mean_max": denied_limit,
        },
        "floor_control_results": floor_control_results,
        "candidate_results": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_gain_floor_screen(args.aggregate_dir)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    if args.require_pass and result["selected_for_confirmation"] is None:
        raise SystemExit("V21 gate found no promotion candidate")


if __name__ == "__main__":
    main()
