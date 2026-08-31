#!/usr/bin/env python3
"""Audit the preregistered V15 holding-efficiency gain screen."""

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
WEIGHTS = (("0025", 0.025), ("0030", 0.03), ("0035", 0.035))
PENALTIES = (("05", 0.5), ("10", 1.0), ("20", 2.0))
CANDIDATE_SPECS = [
    (
        "F_freqduet_protocol_v6_w2adeffgain_l001_e25_r00025_"
        f"w{weight_name}_b{penalty_name}_hiro",
        weight,
        penalty,
    )
    for penalty_name, penalty in PENALTIES
    for weight_name, weight in WEIGHTS
]
CANDIDATES = [name for name, _, _ in CANDIDATE_SPECS]
CONFIGS = [
    HARD_MAIN,
    REFERENCE,
    MATCHED_CONTEXT,
    CURRENT_MAIN,
    SAME_ENTROPY,
    V13_ANCHOR,
    V14_ANCHOR,
    *CANDIDATES,
]
TRAIN_SEEDS = [19013, 19031, 19053, 19077]
EVAL_SEEDS = [52017, 52041, 52059, 52083]
PRIORITY = [
    "F_freqduet_protocol_v6_w2adeffgain_l001_e25_r00025_"
    f"w{weight}_b{penalty}_hiro"
    for weight in ("0025", "0030", "0035")
    for penalty in ("05", "10", "20")
]


def evaluate_efficiency_gain_screen(
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
        raise FileNotFoundError(f"missing V15 aggregate artifacts: {missing}")

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
        raise ValueError(f"V15 strict checks failed: {strict_checks}")

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
        "lower_regularity_policy_capacity_gain_enabled",
        "lower_regularity_policy_capacity_gain_mode",
        "lower_regularity_policy_capacity_gain_weight",
        "lower_regularity_policy_capacity_gain_scale",
        "lower_regularity_policy_capacity_exponent",
        "lower_regularity_policy_action_efficiency_penalty",
        "lower_regularity_policy_capacity_gain_mean",
        "lower_regularity_policy_scaled_capacity_gain_mean",
        "lower_regularity_policy_capacity_gain_bonus",
        "lower_regularity_policy_capacity_gate_mean",
        "lower_regularity_policy_action_efficiency_gate_mean",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(f"V15 metrics are missing: {missing_columns}")

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
    for candidate, expected_weight, expected_penalty in CANDIDATE_SPECS:
        rows = _rows(per_eval, candidate)
        anchor_pair = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == V13_ANCHOR)]
        if len(rows) != expected_pairs or len(anchor_pair) != 1:
            raise ValueError(f"incomplete V15 candidate: {candidate}")
        anchor_pair = anchor_pair.iloc[0]

        adjustment = pd.to_numeric(
            rows["lower_causal_guard_adjustment_mean_s"], errors="coerce")
        enabled = pd.to_numeric(
            rows["lower_regularity_policy_enabled"], errors="coerce")
        policy_mode = rows["lower_regularity_policy_mode"].astype(str)
        scale_mode = rows[
            "lower_regularity_policy_constraint_scale_mode"].astype(str)
        initial = pd.to_numeric(
            rows["lower_regularity_policy_initial_lambda"], errors="coerce")
        cost_limit = pd.to_numeric(
            rows["lower_regularity_policy_cost_limit"], errors="coerce")
        scaled_limit = pd.to_numeric(
            rows["lower_regularity_policy_scaled_limit"], errors="coerce")
        action_regret = pd.to_numeric(
            rows["lower_regularity_policy_action_regret_mean"],
            errors="coerce")
        evidence = pd.to_numeric(
            rows["lower_regularity_policy_evidence_valid_mean"],
            errors="coerce")
        gain_enabled = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_enabled"],
            errors="coerce")
        gain_mode = rows[
            "lower_regularity_policy_capacity_gain_mode"].astype(str)
        gain_weight = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_weight"],
            errors="coerce")
        gain_scale = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_scale"],
            errors="coerce")
        gain_exponent = pd.to_numeric(
            rows["lower_regularity_policy_capacity_exponent"],
            errors="coerce")
        efficiency_penalty = pd.to_numeric(
            rows["lower_regularity_policy_action_efficiency_penalty"],
            errors="coerce")
        realized_gain = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_mean"],
            errors="coerce")
        realized_scaled_gain = pd.to_numeric(
            rows["lower_regularity_policy_scaled_capacity_gain_mean"],
            errors="coerce")
        realized_bonus = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gain_bonus"],
            errors="coerce")
        realized_capacity_gate = pd.to_numeric(
            rows["lower_regularity_policy_capacity_gate_mean"],
            errors="coerce")
        realized_efficiency = pd.to_numeric(
            rows["lower_regularity_policy_action_efficiency_gate_mean"],
            errors="coerce")

        anchor_cv = float(anchor_pair["delta_headway_cv_mean"])
        anchor_cv_ci_high = float(anchor_pair["delta_headway_cv_ci_high"])
        anchor_journey = float(anchor_pair[
            "delta_restricted_total_journey_horizon_min_mean"])
        noguard_cv, n1 = _paired_delta(
            per_eval, candidate, REFERENCE, "headway_cv")
        noguard_journey, n2 = _paired_delta(
            per_eval, candidate, REFERENCE,
            "restricted_total_journey_horizon_min")
        current_cv, n3 = _paired_delta(
            per_eval, candidate, CURRENT_MAIN, "headway_cv")
        current_journey, n4 = _paired_delta(
            per_eval, candidate, CURRENT_MAIN,
            "restricted_total_journey_horizon_min")
        v11_cv, n5 = _paired_delta(
            per_eval, candidate, SAME_ENTROPY, "headway_cv")
        v11_journey, n6 = _paired_delta(
            per_eval, candidate, SAME_ENTROPY,
            "restricted_total_journey_horizon_min")
        v14_cv, n7 = _paired_delta(
            per_eval, candidate, V14_ANCHOR, "headway_cv")
        v14_journey, n8 = _paired_delta(
            per_eval, candidate, V14_ANCHOR,
            "restricted_total_journey_horizon_min")
        pairs_complete = all(count == expected_pairs for count in (
            int(anchor_pair["n_pairs"]), n1, n2, n3, n4, n5, n6, n7, n8))
        candidate_holding = float(pd.to_numeric(
            rows["holding_vehicle_seconds"], errors="coerce").mean())
        candidate_denied = float(pd.to_numeric(
            rows["fleet_denied_dispatch_events"], errors="coerce").mean())
        candidate_action = float(pd.to_numeric(
            rows["lower_action_mean"], errors="coerce").mean())
        v14_holding = _mean(
            per_eval, V14_ANCHOR, "holding_vehicle_seconds")
        v14_denied = _mean(
            per_eval, V14_ANCHOR, "fleet_denied_dispatch_events")
        v14_action = _mean(per_eval, V14_ANCHOR, "lower_action_mean")
        minimum_efficiency = 1.0 / (1.0 + expected_penalty)

        mechanism_checks = {
            "paired_rollouts_complete": pairs_complete,
            "zero_execution_adjustment": bool(
                _finite(adjustment) and (adjustment.abs() <= 1e-12).all()),
            "efficiency_gain_mode_locked": bool(
                (enabled == 1.0).all()
                and (policy_mode
                     == "analytic_two_sided_efficiency_gain_regret_dual_v4"
                    ).all()),
            "dimensionless_regret_constraint": bool(
                (scale_mode == "cost_limit_ratio_v1").all()
                and np.allclose(initial, 0.01)
                and np.allclose(cost_limit, 0.00025)
                and np.allclose(scaled_limit, 1.0)),
            "efficiency_gain_contract_locked": bool(
                (gain_enabled == 1.0).all()
                and (gain_mode
                     == "positive_zero_hold_efficiency_gain_v2").all()
                and np.allclose(gain_weight, expected_weight)
                and np.allclose(gain_scale, 0.002)
                and np.allclose(gain_exponent, 1.0)
                and np.allclose(efficiency_penalty, expected_penalty)),
            "realized_gain_active_in_every_rollout": bool(
                _finite(realized_gain) and (realized_gain > 0.0).all()),
            "capacity_gate_active_in_every_rollout": bool(
                _finite(realized_capacity_gate)
                and (realized_capacity_gate > 0.0).all()
                and (realized_capacity_gate <= 1.0).all()),
            "efficiency_gate_active_in_every_rollout": bool(
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
                    realized_bonus, expected_weight * realized_scaled_gain,
                    atol=1e-7, rtol=1e-6)),
            "regret_limit_satisfied_every_rollout": bool(
                _finite(action_regret)
                and (action_regret <= 0.00025 + 1e-12).all()),
            "causal_evidence_coverage": bool(
                _finite(evidence) and (evidence >= 0.50).all()),
        }
        outcome_checks = {
            "cv_improves_v13_with_ci": (
                anchor_cv <= -0.010 and anchor_cv_ci_high < 0.0),
            "journey_preserved_vs_v13": anchor_journey <= 0.20,
            "journey_beats_noguard": noguard_journey <= -0.25,
            "cv_beats_noguard": noguard_cv <= -0.030,
            "journey_beats_current_main": current_journey <= -0.50,
            "cv_noninferior_to_current_main": current_cv <= 0.005,
            "journey_beats_v11": v11_journey <= -0.25,
            "cv_noninferior_to_v11": v11_cv <= 0.005,
            "journey_improves_v14": v14_journey <= -0.05,
            "cv_improves_v14": v14_cv <= -0.001,
            "holding_does_not_increase_vs_v14": (
                candidate_holding <= v14_holding),
            "denied_does_not_increase_vs_v14": (
                candidate_denied <= v14_denied),
            "action_does_not_increase_vs_v14": (
                candidate_action < v14_action),
            "historical_holding_limit": candidate_holding <= holding_limit,
            "historical_denied_limit": candidate_denied <= denied_limit,
        }
        candidate_results.append({
            "config": candidate,
            "expected_gain_weight": expected_weight,
            "expected_action_efficiency_penalty": expected_penalty,
            "mechanism_checks": mechanism_checks,
            "outcome_checks": outcome_checks,
            "passes": (
                all(mechanism_checks.values())
                and all(outcome_checks.values())),
            "anchor_delta_headway_cv_mean": anchor_cv,
            "anchor_delta_headway_cv_ci_high": anchor_cv_ci_high,
            "anchor_delta_journey_min_mean": anchor_journey,
            "v14_delta_headway_cv_mean": v14_cv,
            "v14_delta_journey_min_mean": v14_journey,
            "holding_vehicle_seconds_mean": candidate_holding,
            "fleet_denied_dispatch_events_mean": candidate_denied,
            "lower_action_mean_s": candidate_action,
            "action_regret_mean": float(action_regret.mean()),
            "causal_evidence_min": float(evidence.min()),
            "realized_efficiency_gate_mean": float(
                realized_efficiency.mean()),
            "realized_efficiency_gain_mean": float(realized_gain.mean()),
        })

    passing = {
        result["config"] for result in candidate_results if result["passes"]}
    selected = next((name for name in PRIORITY if name in passing), None)
    return {
        "gate_version": "freqduet-v15-efficiency-gain-screen-v1",
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
    result = evaluate_efficiency_gain_screen(args.aggregate_dir)
    out = args.out or args.aggregate_dir / "efficiency_gain_screen.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and result["status"] != "exploratory_candidate_selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
