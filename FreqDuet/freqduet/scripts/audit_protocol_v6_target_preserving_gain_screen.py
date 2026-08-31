#!/usr/bin/env python3
"""Audit the preregistered V17 target-preserving efficiency-gain screen."""

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


V14_ANCHOR = (
    "F_freqduet_protocol_v6_w2adcapgain_l001_e25_r00025_w0020_x1_hiro"
)
V15_NEAR = (
    "F_freqduet_protocol_v6_w2adeffgain_l001_e25_r00025_w0025_b10_hiro"
)
V16_NEAR = (
    "F_freqduet_protocol_v6_w2adfleetgain_l001_e25_r00025_w0030_b05_p1_hiro"
)
WEIGHTS = (("0020", 0.02), ("0025", 0.025), ("0030", 0.03))
PENALTIES = (("05", 0.5), ("10", 1.0))
TARGET_EXPONENTS = (("0", 0.0), ("1", 1.0))
PRESSURE_START = 0.9
PRESSURE_FULL = 1.0
PRESSURE_EXPONENT = 1.0
CANDIDATE_SPECS = [
    (
        "F_freqduet_protocol_v6_w2adtpgain_l001_e25_r00025_"
        f"w{weight_name}_b{penalty_name}_t{target_name}_hiro",
        weight,
        penalty,
        target_exponent,
    )
    for target_name, target_exponent in TARGET_EXPONENTS
    for penalty_name, penalty in PENALTIES
    for weight_name, weight in WEIGHTS
]
CANDIDATES = [name for name, _, _, _ in CANDIDATE_SPECS]
CONFIGS = [
    HARD_MAIN,
    REFERENCE,
    MATCHED_CONTEXT,
    CURRENT_MAIN,
    SAME_ENTROPY,
    V13_ANCHOR,
    V14_ANCHOR,
    V15_NEAR,
    V16_NEAR,
    *CANDIDATES,
]
TRAIN_SEEDS = [21013, 21031, 21053, 21077]
EVAL_SEEDS = [54017, 54041, 54059, 54083]
PRIORITY = [
    "F_freqduet_protocol_v6_w2adtpgain_l001_e25_r00025_"
    f"w{weight}_b{penalty}_t{target}_hiro"
    for weight in ("0020", "0025", "0030")
    for penalty in ("05", "10")
    for target in ("1", "0")
]


def evaluate_target_preserving_gain_screen(
    aggregate_dir: Path,
) -> dict[str, object]:
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
        raise FileNotFoundError(f"missing V17 aggregate artifacts: {missing}")

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
        raise ValueError(f"V17 strict checks failed: {strict_checks}")

    required_columns = {
        "config", "train_seed", "eval_seed", "headway_cv",
        "restricted_total_journey_horizon_min", "holding_vehicle_seconds",
        "fleet_denied_dispatch_events", "lower_action_mean",
        "lower_causal_guard_adjustment_mean_s",
        "lower_regularity_policy_enabled", "lower_regularity_policy_mode",
        "lower_regularity_policy_constraint_scale_mode",
        "lower_regularity_policy_initial_lambda",
        "lower_regularity_policy_cost_limit",
        "lower_regularity_policy_scaled_limit",
        "lower_regularity_policy_action_regret_mean",
        "lower_regularity_policy_evidence_valid_mean",
        "lower_regularity_policy_abs_error_mean_s",
        "lower_regularity_policy_capacity_gain_enabled",
        "lower_regularity_policy_capacity_gain_mode",
        "lower_regularity_policy_capacity_gain_weight",
        "lower_regularity_policy_capacity_gain_scale",
        "lower_regularity_policy_capacity_exponent",
        "lower_regularity_policy_action_efficiency_penalty",
        "lower_regularity_policy_opportunity_cost_penalty",
        "lower_regularity_policy_fleet_pressure_start",
        "lower_regularity_policy_fleet_pressure_full",
        "lower_regularity_policy_fleet_pressure_exponent",
        "lower_regularity_policy_target_pressure_exponent",
        "lower_regularity_policy_capacity_gain_mean",
        "lower_regularity_policy_scaled_capacity_gain_mean",
        "lower_regularity_policy_capacity_gain_bonus",
        "lower_regularity_policy_capacity_gate_mean",
        "lower_regularity_policy_action_efficiency_gate_mean",
        "lower_regularity_policy_fleet_utilization_mean",
        "lower_regularity_policy_fleet_pressure_mean",
        "lower_regularity_policy_target_pressure_mean",
        "lower_regularity_policy_actor_target_pressure_mean",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(f"V17 metrics are missing: {missing_columns}")

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
    v14_holding = _mean(per_eval, V14_ANCHOR, "holding_vehicle_seconds")
    v14_denied = _mean(
        per_eval, V14_ANCHOR, "fleet_denied_dispatch_events")
    v14_action = _mean(per_eval, V14_ANCHOR, "lower_action_mean")
    v16_error = _mean(
        per_eval, V16_NEAR,
        "lower_regularity_policy_abs_error_mean_s")

    candidate_results = []
    for candidate, weight, penalty, target_exponent in CANDIDATE_SPECS:
        rows = _rows(per_eval, candidate)
        anchor_pair = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == V13_ANCHOR)]
        if len(rows) != expected_pairs or len(anchor_pair) != 1:
            raise ValueError(f"incomplete V17 candidate: {candidate}")
        anchor_pair = anchor_pair.iloc[0]

        numeric = lambda column: pd.to_numeric(rows[column], errors="coerce")
        adjustment = numeric("lower_causal_guard_adjustment_mean_s")
        enabled = numeric("lower_regularity_policy_enabled")
        policy_mode = rows["lower_regularity_policy_mode"].astype(str)
        scale_mode = rows[
            "lower_regularity_policy_constraint_scale_mode"].astype(str)
        initial = numeric("lower_regularity_policy_initial_lambda")
        cost_limit = numeric("lower_regularity_policy_cost_limit")
        scaled_limit = numeric("lower_regularity_policy_scaled_limit")
        action_regret = numeric("lower_regularity_policy_action_regret_mean")
        evidence = numeric("lower_regularity_policy_evidence_valid_mean")
        abs_error = numeric("lower_regularity_policy_abs_error_mean_s")
        gain_enabled = numeric(
            "lower_regularity_policy_capacity_gain_enabled")
        gain_mode = rows[
            "lower_regularity_policy_capacity_gain_mode"].astype(str)
        gain_weight = numeric("lower_regularity_policy_capacity_gain_weight")
        gain_scale = numeric("lower_regularity_policy_capacity_gain_scale")
        gain_exponent = numeric("lower_regularity_policy_capacity_exponent")
        action_penalty = numeric(
            "lower_regularity_policy_action_efficiency_penalty")
        opportunity_penalty = numeric(
            "lower_regularity_policy_opportunity_cost_penalty")
        pressure_start = numeric(
            "lower_regularity_policy_fleet_pressure_start")
        pressure_full = numeric(
            "lower_regularity_policy_fleet_pressure_full")
        pressure_exponent = numeric(
            "lower_regularity_policy_fleet_pressure_exponent")
        observed_target_exponent = numeric(
            "lower_regularity_policy_target_pressure_exponent")
        realized_gain = numeric(
            "lower_regularity_policy_capacity_gain_mean")
        realized_scaled_gain = numeric(
            "lower_regularity_policy_scaled_capacity_gain_mean")
        realized_bonus = numeric(
            "lower_regularity_policy_capacity_gain_bonus")
        realized_capacity_gate = numeric(
            "lower_regularity_policy_capacity_gate_mean")
        realized_efficiency = numeric(
            "lower_regularity_policy_action_efficiency_gate_mean")
        fleet_utilization = numeric(
            "lower_regularity_policy_fleet_utilization_mean")
        fleet_pressure = numeric(
            "lower_regularity_policy_fleet_pressure_mean")
        target_pressure = numeric(
            "lower_regularity_policy_target_pressure_mean")
        actor_target_pressure = numeric(
            "lower_regularity_policy_actor_target_pressure_mean")

        anchor_cv = float(anchor_pair["delta_headway_cv_mean"])
        anchor_cv_ci_high = float(anchor_pair["delta_headway_cv_ci_high"])
        anchor_journey = float(anchor_pair[
            "delta_restricted_total_journey_horizon_min_mean"])
        comparisons = {}
        for label, control in (
                ("noguard", REFERENCE),
                ("current", CURRENT_MAIN),
                ("v11", SAME_ENTROPY),
                ("v14", V14_ANCHOR),
                ("v15", V15_NEAR),
                ("v16", V16_NEAR)):
            comparisons[(label, "cv")] = _paired_delta(
                per_eval, candidate, control, "headway_cv")
            comparisons[(label, "journey")] = _paired_delta(
                per_eval, candidate, control,
                "restricted_total_journey_horizon_min")
        pairs_complete = (
            int(anchor_pair["n_pairs"]) == expected_pairs
            and all(count == expected_pairs
                    for _, count in comparisons.values()))

        candidate_holding = float(numeric(
            "holding_vehicle_seconds").mean())
        candidate_denied = float(numeric(
            "fleet_denied_dispatch_events").mean())
        candidate_action = float(numeric("lower_action_mean").mean())
        minimum_efficiency = 1.0 / (1.0 + penalty)
        if target_exponent == 0.0:
            target_signal_valid = bool(
                np.allclose(target_pressure, 1.0)
                and np.allclose(actor_target_pressure, 1.0))
        else:
            target_signal_valid = bool(
                _finite(target_pressure)
                and (target_pressure > 0.0).all()
                and (target_pressure < 1.0).all()
                and _finite(actor_target_pressure)
                and (actor_target_pressure > 0.0).all()
                and (actor_target_pressure < 1.0).all())

        mechanism_checks = {
            "paired_rollouts_complete": pairs_complete,
            "zero_execution_adjustment": bool(
                _finite(adjustment) and (adjustment.abs() <= 1e-12).all()),
            "target_preserving_gain_mode_locked": bool(
                (enabled == 1.0).all()
                and (policy_mode
                     == "analytic_two_sided_target_preserving_gain_regret_dual_v6"
                    ).all()),
            "dimensionless_regret_constraint": bool(
                (scale_mode == "cost_limit_ratio_v1").all()
                and np.allclose(initial, 0.01)
                and np.allclose(cost_limit, 0.00025)
                and np.allclose(scaled_limit, 1.0)),
            "target_preserving_gain_contract_locked": bool(
                (gain_enabled == 1.0).all()
                and (gain_mode
                     == "positive_zero_hold_target_preserving_gain_v4").all()
                and np.allclose(gain_weight, weight)
                and np.allclose(gain_scale, 0.002)
                and np.allclose(gain_exponent, 1.0)
                and np.allclose(action_penalty, 0.0)
                and np.allclose(opportunity_penalty, penalty)
                and np.allclose(pressure_start, PRESSURE_START)
                and np.allclose(pressure_full, PRESSURE_FULL)
                and np.allclose(pressure_exponent, PRESSURE_EXPONENT)
                and np.allclose(
                    observed_target_exponent, target_exponent)),
            "fleet_signal_active_in_every_rollout": bool(
                _finite(fleet_utilization)
                and (fleet_utilization > 0.0).all()
                and (fleet_utilization <= 1.0).all()
                and _finite(fleet_pressure)
                and (fleet_pressure > 0.0).all()
                and (fleet_pressure < 1.0).all()),
            "target_signal_matches_registered_exponent": (
                target_signal_valid),
            "realized_gain_active_in_every_rollout": bool(
                _finite(realized_gain) and (realized_gain > 0.0).all()),
            "capacity_gate_active_in_every_rollout": bool(
                _finite(realized_capacity_gate)
                and (realized_capacity_gate > 0.0).all()
                and (realized_capacity_gate <= 1.0).all()),
            "state_scalar_gate_active_in_every_rollout": bool(
                _finite(realized_efficiency)
                and (realized_efficiency >= minimum_efficiency - 1e-7).all()
                and (realized_efficiency < 1.0).all()),
            "gain_arithmetic_verified": bool(
                _finite(realized_scaled_gain)
                and _finite(realized_bonus)
                and np.allclose(
                    realized_scaled_gain, realized_gain / 0.002,
                    atol=3e-6, rtol=1e-6)
                and np.allclose(
                    realized_bonus, weight * realized_scaled_gain,
                    atol=1e-7, rtol=1e-6)),
            "regret_limit_satisfied_every_rollout": bool(
                _finite(action_regret)
                and (action_regret <= 0.00025 + 1e-12).all()),
            "causal_evidence_coverage": bool(
                _finite(evidence) and (evidence >= 0.50).all()),
        }

        delta = lambda label, metric: comparisons[(label, metric)][0]
        outcome_checks = {
            "cv_improves_v13_with_ci": (
                anchor_cv <= -0.010 and anchor_cv_ci_high < 0.0),
            "journey_preserved_vs_v13": anchor_journey <= 0.20,
            "journey_beats_noguard": delta("noguard", "journey") <= -0.25,
            "cv_beats_noguard": delta("noguard", "cv") <= -0.030,
            "journey_beats_current_main": (
                delta("current", "journey") <= -0.50),
            "cv_noninferior_to_current_main": (
                delta("current", "cv") <= 0.005),
            "journey_beats_v11": delta("v11", "journey") <= -0.25,
            "cv_noninferior_to_v11": delta("v11", "cv") <= 0.005,
            "journey_improves_v14": delta("v14", "journey") <= -0.05,
            "cv_improves_v14": delta("v14", "cv") <= -0.001,
            "journey_noninferior_to_v15": (
                delta("v15", "journey") <= 0.02),
            "cv_improves_v15": delta("v15", "cv") <= -0.001,
            "journey_noninferior_to_v16": (
                delta("v16", "journey") <= 0.02),
            "cv_improves_v16": delta("v16", "cv") <= -0.001,
            "target_error_improves_v16": float(abs_error.mean()) < v16_error,
            "holding_does_not_increase_vs_v14": (
                candidate_holding <= v14_holding),
            "denied_does_not_increase_vs_v14": (
                candidate_denied <= v14_denied),
            "action_does_not_increase_vs_v14": (
                candidate_action <= v14_action),
            "historical_holding_limit": candidate_holding <= holding_limit,
            "historical_denied_limit": candidate_denied <= denied_limit,
        }
        candidate_results.append({
            "config": candidate,
            "expected_gain_weight": weight,
            "expected_opportunity_cost_penalty": penalty,
            "expected_target_pressure_exponent": target_exponent,
            "mechanism_checks": mechanism_checks,
            "outcome_checks": outcome_checks,
            "passes": (
                all(mechanism_checks.values())
                and all(outcome_checks.values())),
            "anchor_delta_headway_cv_mean": anchor_cv,
            "anchor_delta_headway_cv_ci_high": anchor_cv_ci_high,
            "anchor_delta_journey_min_mean": anchor_journey,
            "v14_delta_headway_cv_mean": delta("v14", "cv"),
            "v14_delta_journey_min_mean": delta("v14", "journey"),
            "v15_delta_headway_cv_mean": delta("v15", "cv"),
            "v15_delta_journey_min_mean": delta("v15", "journey"),
            "v16_delta_headway_cv_mean": delta("v16", "cv"),
            "v16_delta_journey_min_mean": delta("v16", "journey"),
            "holding_vehicle_seconds_mean": candidate_holding,
            "fleet_denied_dispatch_events_mean": candidate_denied,
            "lower_action_mean_s": candidate_action,
            "target_abs_error_mean_s": float(abs_error.mean()),
            "action_regret_mean": float(action_regret.mean()),
            "causal_evidence_min": float(evidence.min()),
            "fleet_utilization_mean": float(fleet_utilization.mean()),
            "fleet_pressure_mean": float(fleet_pressure.mean()),
            "target_pressure_mean": float(target_pressure.mean()),
            "realized_state_scalar_gate_mean": float(
                realized_efficiency.mean()),
            "realized_efficiency_gain_mean": float(realized_gain.mean()),
        })

    passing = {
        result["config"] for result in candidate_results if result["passes"]}
    selected = next((name for name in PRIORITY if name in passing), None)
    return {
        "gate_version": "freqduet-v17-target-preserving-gain-screen-v1",
        "status": (
            "exploratory_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "strict_checks": strict_checks,
        "resource_limits": {
            "holding_vehicle_seconds_mean_max": holding_limit,
            "fleet_denied_dispatch_events_mean_max": denied_limit,
        },
        "candidate_results": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_target_preserving_gain_screen(args.aggregate_dir)
    out = args.out or args.aggregate_dir / "target_preserving_gain_screen.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and result[
            "status"] != "exploratory_candidate_selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
