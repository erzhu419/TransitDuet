#!/usr/bin/env python3
"""Audit the preregistered V22 aggregate-gain optimizer screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_capacity_gain_screen import (
    CURRENT_MAIN,
    REFERENCE,
    V13_ANCHOR,
    _finite,
    _mean,
    _paired_delta,
    _rows,
)


V13_ZERO_HOLD_ADVANTAGE = (
    "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_qadv0_hiro"
)
V20_QADV_B080 = (
    "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_"
    "paxdual_b080_qadv0_hiro"
)
V21_RELATIVE_LOG_ANCHOR = (
    "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_f030_h030_"
    "pax080_qadv0_hiro"
)
FACTORIAL_SPECS = [
    (V21_RELATIVE_LOG_ANCHOR, "relative", "log_adam_v1", 0.0, False),
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_f030_h030_"
        "pax080_qadv0_pdual_hiro",
        "relative", "projected_violation_v1", 0.0, True,
    ),
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_f030_h030_"
        "pax080_qadv0_al05_hiro",
        "relative", "log_adam_v1", 0.5, True,
    ),
    (
        "F_freqduet_protocol_v6_w2adgfloor_l001_e25_c005_f030_h030_"
        "pax080_qadv0_pdual_al05_hiro",
        "relative", "projected_violation_v1", 0.5, True,
    ),
    (
        "F_freqduet_protocol_v6_w2adaggfloor_l001_e25_c005_f030_h030_"
        "pax080_qadv0_hiro",
        "aggregate", "log_adam_v1", 0.0, True,
    ),
    (
        "F_freqduet_protocol_v6_w2adaggfloor_l001_e25_c005_f030_h030_"
        "pax080_qadv0_pdual_hiro",
        "aggregate", "projected_violation_v1", 0.0, True,
    ),
    (
        "F_freqduet_protocol_v6_w2adaggfloor_l001_e25_c005_f030_h030_"
        "pax080_qadv0_al05_hiro",
        "aggregate", "log_adam_v1", 0.5, True,
    ),
    (
        "F_freqduet_protocol_v6_w2adaggfloor_l001_e25_c005_f030_h030_"
        "pax080_qadv0_pdual_al05_hiro",
        "aggregate", "projected_violation_v1", 0.5, True,
    ),
]
FACTORIAL_CONFIGS = [spec[0] for spec in FACTORIAL_SPECS]
PROMOTION_CANDIDATES = [spec[0] for spec in FACTORIAL_SPECS if spec[4]]
CONFIGS = [
    CURRENT_MAIN,
    REFERENCE,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    V20_QADV_B080,
    *FACTORIAL_CONFIGS,
]
TRAIN_SEEDS = [27013, 27031, 27053, 27077]
EVAL_SEEDS = [60017, 60041, 60059, 60083]
PRIORITY = [
    FACTORIAL_SPECS[7][0],
    FACTORIAL_SPECS[5][0],
    FACTORIAL_SPECS[6][0],
    FACTORIAL_SPECS[4][0],
    FACTORIAL_SPECS[3][0],
    FACTORIAL_SPECS[1][0],
    FACTORIAL_SPECS[2][0],
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


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _ratio_identity(
    numerator: pd.Series,
    denominator: pd.Series,
    ratio: pd.Series,
) -> bool:
    if not (_finite(numerator) and _finite(denominator) and _finite(ratio)):
        return False
    numerator_np = numerator.to_numpy(dtype=float)
    denominator_np = denominator.to_numpy(dtype=float)
    expected = np.divide(
        numerator_np,
        denominator_np,
        out=np.zeros_like(numerator_np),
        where=denominator_np > 1e-12,
    )
    return bool(np.allclose(
        ratio.to_numpy(dtype=float), expected, atol=2e-6, rtol=2e-5))


def _mechanism_checks(
    rows: pd.DataFrame,
    *,
    allocation: str,
    dual_update_mode: str,
    augmented_rho: float,
) -> dict[str, bool]:
    expected_policy_mode = (
        "analytic_two_sided_hf_aggregate_gain_floor_dual_v9"
        if allocation == "aggregate"
        else "analytic_two_sided_hf_gain_floor_dual_v8")
    expected_cost_mode = (
        "hf_aggregate_gain_shortfall_v4"
        if allocation == "aggregate"
        else "hf_relative_gain_shortfall_v3")
    expected_floor_mode = (
        "causal_hf_aggregate_gain_floor_v2"
        if allocation == "aggregate"
        else "causal_hf_relative_gain_floor_v1")

    evidence = _numeric(
        rows, "lower_regularity_policy_evidence_valid_mean")
    regret = _numeric(rows, "lower_regularity_policy_action_regret_mean")
    regularity_lambda = _numeric(rows, "lower_regularity_lambda")
    passenger_lambda = _numeric(rows, "lower_regularity_passenger_lambda")
    hf_pressure = _numeric(
        rows, "lower_regularity_gain_floor_hf_pressure_mean")
    required_fraction = _numeric(
        rows, "lower_regularity_gain_floor_required_fraction_mean")
    expected_gain_fraction = _numeric(
        rows, "lower_regularity_gain_floor_expected_gain_fraction_mean")
    selected_gain_fraction = _numeric(
        rows, "lower_regularity_gain_floor_selected_gain_fraction_mean")
    expected_relative_shortfall = _numeric(
        rows, "lower_regularity_gain_floor_expected_shortfall_mean")
    selected_relative_shortfall = _numeric(
        rows, "lower_regularity_gain_floor_selected_shortfall_mean")
    required_gain = _numeric(
        rows, "lower_regularity_gain_floor_required_gain_mean")
    expected_absolute_shortfall = _numeric(
        rows,
        "lower_regularity_gain_floor_expected_absolute_shortfall_mean",
    )
    selected_absolute_shortfall = _numeric(
        rows,
        "lower_regularity_gain_floor_selected_absolute_shortfall_mean",
    )
    expected_aggregate_ratio = _numeric(
        rows,
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
    )
    selected_aggregate_ratio = _numeric(
        rows,
        "lower_regularity_gain_floor_selected_aggregate_shortfall_ratio",
    )
    eligible = _numeric(rows, "lower_regularity_gain_floor_eligible_mean")
    passenger_expected = _numeric(
        rows, "lower_regularity_passenger_expected_cost_mean")
    passenger_selected = _numeric(
        rows, "lower_regularity_passenger_selected_cost_mean")
    passenger_load = _numeric(
        rows, "lower_regularity_passenger_load_mean")

    actor_fields = [
        _numeric(rows, column)
        for column in (
            "lower_regularity_gain_floor_actor_required_fraction_mean",
            "lower_regularity_gain_floor_actor_hf_energy_mean",
            "lower_regularity_gain_floor_actor_hf_pressure_mean",
            "lower_regularity_gain_floor_actor_expected_gain_fraction_mean",
            "lower_regularity_gain_floor_actor_expected_shortfall_mean",
            "lower_regularity_gain_floor_actor_required_gain_mean",
            "lower_regularity_gain_floor_actor_expected_absolute_shortfall_mean",
            "lower_regularity_gain_floor_actor_aggregate_shortfall_ratio",
            "lower_regularity_gain_floor_actor_eligible_fraction",
            "lower_regularity_policy_augmented_penalty",
            "lower_regularity_passenger_actor_augmented_penalty",
        )
    ]
    regularity_budget = (
        expected_aggregate_ratio
        if allocation == "aggregate"
        else expected_relative_shortfall)

    return {
        "zero_hold_advantage_critic_locked": bool(
            (rows["lower_discrete_critic"].astype(str)
             == "zero_hold_advantage").all()),
        "frozen_evaluation_locked": bool(
            (_numeric(rows, "lower_policy_frozen") == 1.0).all()
            and (_numeric(rows, "lower_critic_frozen") == 1.0).all()
            and (_numeric(rows, "upper_policy_frozen") == 1.0).all()
            and all(_finite(field) and (field.abs() <= 1e-12).all()
                    for field in actor_fields)),
        "zero_execution_adjustment": bool(
            _finite(_numeric(
                rows, "lower_causal_guard_adjustment_mean_s"))
            and (_numeric(
                rows,
                "lower_causal_guard_adjustment_mean_s").abs()
                 <= 1e-12).all()),
        "gain_floor_contract_locked": bool(
            (_numeric(rows, "lower_regularity_policy_enabled") == 1.0).all()
            and (rows["lower_regularity_policy_mode"].astype(str)
                 == expected_policy_mode).all()
            and (rows[
                "lower_regularity_policy_constraint_cost_mode"].astype(str)
                 == expected_cost_mode).all()
            and (rows[
                "lower_regularity_policy_constraint_scale_mode"].astype(str)
                 == "cost_limit_ratio_v1").all()
            and (rows[
                "lower_regularity_policy_dual_update_mode"].astype(str)
                 == dual_update_mode).all()
            and np.allclose(_numeric(
                rows,
                "lower_regularity_policy_augmented_lagrangian_rho"),
                augmented_rho)
            and np.allclose(_numeric(
                rows, "lower_regularity_policy_initial_lambda"), 0.01)
            and np.allclose(_numeric(
                rows, "lower_regularity_policy_cost_limit"), 0.05)
            and np.allclose(_numeric(
                rows, "lower_regularity_policy_scaled_limit"), 1.0)
            and (_numeric(
                rows, "lower_regularity_gain_floor_enabled") == 1.0).all()
            and (rows["lower_regularity_gain_floor_mode"].astype(str)
                 == expected_floor_mode).all()
            and np.allclose(_numeric(
                rows, "lower_regularity_gain_floor_base_fraction"), 0.30)
            and np.allclose(_numeric(
                rows, "lower_regularity_gain_floor_hf_increment"), 0.30)
            and np.allclose(_numeric(
                rows, "lower_regularity_gain_floor_hf_energy_scale"), 0.04)
            and np.allclose(_numeric(
                rows,
                "lower_regularity_gain_floor_hf_energy_exponent"), 1.0)),
        "passenger_contract_locked": bool(
            (_numeric(
                rows, "lower_regularity_passenger_holding_enabled")
             == 1.0).all()
            and (rows[
                "lower_regularity_passenger_holding_mode"].astype(str)
                 == "causal_apc_person_delay_dual_v1").all()
            and (rows[
                "lower_regularity_passenger_constraint_scale_mode"].astype(str)
                 == "cost_limit_ratio_v1").all()
            and (rows[
                "lower_regularity_passenger_dual_update_mode"].astype(str)
                 == dual_update_mode).all()
            and np.allclose(_numeric(
                rows,
                "lower_regularity_passenger_augmented_lagrangian_rho"),
                augmented_rho)
            and np.allclose(_numeric(
                rows, "lower_regularity_passenger_initial_lambda"), 0.01)
            and np.allclose(_numeric(
                rows, "lower_regularity_passenger_cost_limit"), 0.08)
            and np.allclose(_numeric(
                rows, "lower_regularity_passenger_scaled_limit"), 1.0)),
        "passive_relative_floor_arithmetic": bool(
            _finite(hf_pressure)
            and (hf_pressure >= 0.0).all()
            and (hf_pressure <= 1.0 + 1e-12).all()
            and _finite(required_fraction)
            and np.allclose(
                required_fraction, 0.30 + 0.30 * hf_pressure,
                atol=1e-6, rtol=0.0)
            and _finite(expected_gain_fraction)
            and (expected_gain_fraction >= 0.0).all()
            and (expected_gain_fraction <= 1.0 + 1e-12).all()
            and _finite(selected_gain_fraction)
            and (selected_gain_fraction >= 0.0).all()
            and (selected_gain_fraction <= 1.0 + 1e-12).all()
            and _finite(expected_relative_shortfall)
            and (expected_relative_shortfall >= 0.0).all()
            and _finite(selected_relative_shortfall)
            and (selected_relative_shortfall >= 0.0).all()),
        "passive_aggregate_floor_arithmetic": bool(
            _finite(eligible) and (eligible > 0.0).all()
            and (eligible <= 1.0 + 1e-12).all()
            and _finite(required_gain) and (required_gain > 0.0).all()
            and _finite(expected_absolute_shortfall)
            and (expected_absolute_shortfall >= 0.0).all()
            and (expected_absolute_shortfall
                 <= required_gain + 1e-8).all()
            and _finite(selected_absolute_shortfall)
            and (selected_absolute_shortfall >= 0.0).all()
            and (selected_absolute_shortfall
                 <= required_gain + 1e-8).all()
            and _ratio_identity(
                expected_absolute_shortfall,
                required_gain,
                expected_aggregate_ratio)
            and _ratio_identity(
                selected_absolute_shortfall,
                required_gain,
                selected_aggregate_ratio)),
        "hf_conditioning_active": bool(
            (hf_pressure > 0.0).all()
            and float(hf_pressure.max() - hf_pressure.min()) > 1e-8),
        "regularity_budget_satisfied_every_rollout": bool(
            _finite(regularity_budget)
            and (regularity_budget >= 0.0).all()
            and (regularity_budget <= 0.05 + 1e-12).all()),
        "passenger_budget_satisfied_every_rollout": bool(
            _finite(passenger_expected)
            and (passenger_expected >= 0.0).all()
            and (passenger_expected <= 0.08 + 1e-12).all()),
        "passenger_telemetry_active": bool(
            _finite(passenger_selected)
            and (passenger_selected >= 0.0).all()
            and _finite(passenger_load)
            and (passenger_load > 0.0).all()
            and (passenger_load <= 1.0 + 1e-12).all()),
        "independent_duals_finite_and_bounded": bool(
            _finite(regularity_lambda)
            and (regularity_lambda >= 0.0001 - 1e-12).all()
            and (regularity_lambda <= 2.0 + 1e-12).all()
            and _finite(passenger_lambda)
            and (passenger_lambda >= 0.0001 - 1e-12).all()
            and (passenger_lambda <= 2.0 + 1e-12).all()),
        "regret_limit_satisfied_every_rollout": bool(
            _finite(regret) and (regret <= 0.00025 + 1e-12).all()),
        "causal_evidence_coverage": bool(
            _finite(evidence) and (evidence >= 0.50).all()),
    }


def evaluate_aggregate_gain_screen(aggregate_dir: Path) -> dict[str, object]:
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
        raise FileNotFoundError(f"missing V22 aggregate artifacts: {missing}")

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
        raise ValueError(f"V22 strict checks failed: {strict_checks}")

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
        "lower_regularity_policy_dual_update_mode",
        "lower_regularity_policy_augmented_lagrangian_rho",
        "lower_regularity_policy_initial_lambda",
        "lower_regularity_policy_cost_limit",
        "lower_regularity_policy_scaled_limit",
        "lower_regularity_policy_action_regret_mean",
        "lower_regularity_policy_evidence_valid_mean",
        "lower_regularity_policy_augmented_penalty",
        "lower_regularity_lambda", "lower_regularity_gain_floor_enabled",
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
        "lower_regularity_gain_floor_actor_required_gain_mean",
        "lower_regularity_gain_floor_actor_expected_absolute_shortfall_mean",
        "lower_regularity_gain_floor_actor_aggregate_shortfall_ratio",
        "lower_regularity_gain_floor_actor_eligible_fraction",
        "lower_regularity_gain_floor_required_fraction_mean",
        "lower_regularity_gain_floor_hf_pressure_mean",
        "lower_regularity_gain_floor_expected_gain_fraction_mean",
        "lower_regularity_gain_floor_selected_gain_fraction_mean",
        "lower_regularity_gain_floor_expected_shortfall_mean",
        "lower_regularity_gain_floor_selected_shortfall_mean",
        "lower_regularity_gain_floor_required_gain_mean",
        "lower_regularity_gain_floor_expected_absolute_shortfall_mean",
        "lower_regularity_gain_floor_selected_absolute_shortfall_mean",
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
        "lower_regularity_gain_floor_selected_aggregate_shortfall_ratio",
        "lower_regularity_gain_floor_eligible_mean",
        "lower_regularity_passenger_holding_enabled",
        "lower_regularity_passenger_holding_mode",
        "lower_regularity_passenger_constraint_scale_mode",
        "lower_regularity_passenger_dual_update_mode",
        "lower_regularity_passenger_augmented_lagrangian_rho",
        "lower_regularity_passenger_initial_lambda",
        "lower_regularity_passenger_cost_limit",
        "lower_regularity_passenger_scaled_limit",
        "lower_regularity_passenger_expected_cost_mean",
        "lower_regularity_passenger_selected_cost_mean",
        "lower_regularity_passenger_load_mean",
        "lower_regularity_passenger_actor_augmented_penalty",
        "lower_regularity_passenger_lambda",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(f"V22 metrics are missing: {missing_columns}")

    nonfactor_contract_locked = all(
        (_numeric(
            _rows(per_eval, config),
            "lower_regularity_gain_floor_enabled") == 0.0).all()
        for config in CONFIGS if config not in FACTORIAL_CONFIGS
    )
    candidate_results = []
    for config, allocation, dual_update_mode, augmented_rho, promotable in (
            FACTORIAL_SPECS):
        rows = _rows(per_eval, config)
        if len(rows) != expected_pairs:
            raise ValueError(f"incomplete V22 factorial row: {config}")
        aggregate_pair = paired.loc[
            (paired["candidate"] == config)
            & (paired["reference"] == V13_ANCHOR)]
        if len(aggregate_pair) != 1 or int(
                aggregate_pair.iloc[0]["n_pairs"]) != expected_pairs:
            raise ValueError(f"incomplete V22 aggregate pair: {config}")

        v13_delta, pair_counts = _paired_metrics(
            per_eval, config, V13_ANCHOR)
        qadv_delta, counts = _paired_metrics(
            per_eval, config, V13_ZERO_HOLD_ADVANTAGE)
        pair_counts.extend(counts)
        v20_delta, counts = _paired_metrics(
            per_eval, config, V20_QADV_B080)
        pair_counts.extend(counts)
        current_delta, counts = _paired_metrics(
            per_eval, config, CURRENT_MAIN)
        pair_counts.extend(counts)
        noguard_delta, counts = _paired_metrics(
            per_eval, config, REFERENCE)
        pair_counts.extend(counts)

        mechanism_checks = _mechanism_checks(
            rows,
            allocation=allocation,
            dual_update_mode=dual_update_mode,
            augmented_rho=augmented_rho,
        )
        mechanism_checks.update({
            "paired_rollouts_complete": all(
                count == expected_pairs for count in pair_counts),
            "nonfactor_controls_locked": nonfactor_contract_locked,
        })
        candidate_holding = float(_numeric(
            rows, "holding_vehicle_seconds").mean())
        candidate_denied = float(_numeric(
            rows, "fleet_denied_dispatch_events").mean())
        candidate_action = float(_numeric(
            rows, "lower_action_mean").mean())
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
                noguard_delta["journey_min"] <= -0.25),
            "cv_beats_noguard": (
                noguard_delta["headway_cv"] <= -0.030),
            "journey_beats_confirmed_main": (
                current_delta["journey_min"] <= -0.50),
            "cv_noninferior_to_confirmed_main": (
                current_delta["headway_cv"] <= 0.005),
        }
        regularity_budget_column = (
            "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio"
            if allocation == "aggregate"
            else "lower_regularity_gain_floor_expected_shortfall_mean")
        candidate_results.append({
            "config": config,
            "allocation": allocation,
            "dual_update_mode": dual_update_mode,
            "augmented_lagrangian_rho": augmented_rho,
            "promotion_eligible": promotable,
            "mechanism_checks": mechanism_checks,
            "outcome_checks": outcome_checks,
            "passes": bool(
                promotable
                and all(mechanism_checks.values())
                and all(outcome_checks.values())),
            "v13_delta": v13_delta,
            "unbounded_qadv_delta": qadv_delta,
            "v20_passenger_delta": v20_delta,
            "confirmed_main_delta": current_delta,
            "noguard_delta": noguard_delta,
            "holding_vehicle_seconds_mean": candidate_holding,
            "fleet_denied_dispatch_events_mean": candidate_denied,
            "lower_action_mean_s": candidate_action,
            "regularity_cost_max": float(_numeric(
                rows, regularity_budget_column).max()),
            "passenger_cost_max": float(_numeric(
                rows,
                "lower_regularity_passenger_expected_cost_mean").max()),
        })

    passing = {
        result["config"] for result in candidate_results if result["passes"]
    }
    selected = next((name for name in PRIORITY if name in passing), None)
    return {
        "gate_version": "freqduet-v22-aggregate-gain-optimizer-screen-v1",
        "status": (
            "exploratory_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "strict_checks": strict_checks,
        "candidate_results": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_aggregate_gain_screen(args.aggregate_dir)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    if args.require_pass and result["selected_for_confirmation"] is None:
        raise SystemExit("V22 gate found no promotion candidate")


if __name__ == "__main__":
    main()
