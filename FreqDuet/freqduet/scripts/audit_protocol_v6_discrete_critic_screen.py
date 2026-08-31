#!/usr/bin/env python3
"""Audit the preregistered V19 lower discrete-critic screen."""

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
V13_INDEXED = (
    "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_qidx_hiro"
)
V13_ZERO_HOLD_ADVANTAGE = (
    "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_qadv0_hiro"
)
V14_INDEXED = (
    "F_freqduet_protocol_v6_w2adcapgain_l001_e25_r00025_"
    "w0020_x1_qidx_hiro"
)
V14_ZERO_HOLD_ADVANTAGE = (
    "F_freqduet_protocol_v6_w2adcapgain_l001_e25_r00025_"
    "w0020_x1_qadv0_hiro"
)
CANDIDATE_SPECS = [
    (V13_INDEXED, V13_ANCHOR, "indexed", False),
    (V13_ZERO_HOLD_ADVANTAGE, V13_ANCHOR, "zero_hold_advantage", False),
    (V14_INDEXED, V14_ANCHOR, "indexed", True),
    (
        V14_ZERO_HOLD_ADVANTAGE,
        V14_ANCHOR,
        "zero_hold_advantage",
        True,
    ),
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
    *CANDIDATES,
]
TRAIN_SEEDS = [24013, 24031, 24053, 24077]
EVAL_SEEDS = [57017, 57041, 57059, 57083]
PRIORITY = [
    V14_ZERO_HOLD_ADVANTAGE,
    V13_ZERO_HOLD_ADVANTAGE,
    V14_INDEXED,
    V13_INDEXED,
]


def _paired_metrics(
    per_eval: pd.DataFrame,
    candidate: str,
    reference: str,
) -> tuple[dict[str, float], list[int]]:
    metrics = {
        "headway_cv": "headway_cv",
        "journey_min": "restricted_total_journey_horizon_min",
    }
    deltas: dict[str, float] = {}
    counts: list[int] = []
    for label, metric in metrics.items():
        deltas[label], count = _paired_delta(
            per_eval, candidate, reference, metric)
        counts.append(count)
    return deltas, counts


def evaluate_discrete_critic_screen(
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
        raise FileNotFoundError(f"missing V19 aggregate artifacts: {missing}")

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
        raise ValueError(f"V19 strict checks failed: {strict_checks}")

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
        "lower_regularity_policy_capacity_gain_enabled",
        "lower_regularity_policy_capacity_gain_mode",
        "lower_regularity_policy_capacity_gain_weight",
        "lower_regularity_policy_capacity_gain_scale",
        "lower_regularity_policy_capacity_exponent",
        "lower_regularity_policy_capacity_gain_mean",
        "lower_regularity_policy_scaled_capacity_gain_mean",
        "lower_regularity_policy_capacity_gain_bonus",
        "lower_regularity_policy_capacity_gate_mean",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(f"V19 metrics are missing: {missing_columns}")

    scalar_controls = [
        HARD_MAIN,
        REFERENCE,
        MATCHED_CONTEXT,
        CURRENT_MAIN,
        SAME_ENTROPY,
        V13_ANCHOR,
        V14_ANCHOR,
    ]
    scalar_critic_locked = all(
        (_rows(per_eval, config)["lower_discrete_critic"].astype(str)
         == "continuous_action").all()
        for config in scalar_controls
    )
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
    for candidate, anchor, expected_critic, capacity_gain in CANDIDATE_SPECS:
        rows = _rows(per_eval, candidate)
        if len(rows) != expected_pairs:
            raise ValueError(f"incomplete V19 candidate: {candidate}")
        anchor_pair = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == V13_ANCHOR)]
        if len(anchor_pair) != 1 or int(
                anchor_pair.iloc[0]["n_pairs"]) != expected_pairs:
            raise ValueError(f"incomplete V19 aggregate pair: {candidate}")

        numeric = lambda column: pd.to_numeric(rows[column], errors="coerce")
        anchor_delta, anchor_counts = _paired_metrics(
            per_eval, candidate, anchor)
        comparisons = {}
        pair_counts = list(anchor_counts)
        for label, control in controls:
            comparisons[label], counts = _paired_metrics(
                per_eval, candidate, control)
            pair_counts.extend(counts)

        adjustment = numeric("lower_causal_guard_adjustment_mean_s")
        policy_frozen = numeric("lower_policy_frozen")
        critic_frozen = numeric("lower_critic_frozen")
        upper_frozen = numeric("upper_policy_frozen")
        enabled = numeric("lower_regularity_policy_enabled")
        policy_mode = rows["lower_regularity_policy_mode"].astype(str)
        constraint_cost_mode = rows[
            "lower_regularity_policy_constraint_cost_mode"].astype(str)
        scale_mode = rows[
            "lower_regularity_policy_constraint_scale_mode"].astype(str)
        initial = numeric("lower_regularity_policy_initial_lambda")
        cost_limit = numeric("lower_regularity_policy_cost_limit")
        scaled_limit = numeric("lower_regularity_policy_scaled_limit")
        action_regret = numeric("lower_regularity_policy_action_regret_mean")
        evidence = numeric("lower_regularity_policy_evidence_valid_mean")
        gain_enabled = numeric(
            "lower_regularity_policy_capacity_gain_enabled")
        gain_mode = rows[
            "lower_regularity_policy_capacity_gain_mode"].astype(str)
        gain_weight = numeric(
            "lower_regularity_policy_capacity_gain_weight")
        gain_scale = numeric(
            "lower_regularity_policy_capacity_gain_scale")
        gain_exponent = numeric(
            "lower_regularity_policy_capacity_exponent")
        realized_gain = numeric(
            "lower_regularity_policy_capacity_gain_mean")
        realized_scaled_gain = numeric(
            "lower_regularity_policy_scaled_capacity_gain_mean")
        realized_bonus = numeric(
            "lower_regularity_policy_capacity_gain_bonus")
        realized_gate = numeric(
            "lower_regularity_policy_capacity_gate_mean")

        if capacity_gain:
            objective_contract = bool(
                (policy_mode
                 == "analytic_two_sided_capacity_gain_regret_dual_v3").all()
                and (gain_enabled == 1.0).all()
                and (gain_mode == "positive_zero_hold_gain_v1").all()
                and np.allclose(gain_weight, 0.02)
                and np.allclose(gain_scale, 0.002)
                and np.allclose(gain_exponent, 1.0)
                and _finite(realized_gain)
                and (realized_gain > 0.0).all()
                and _finite(realized_gate)
                and (realized_gate > 0.0).all()
                and (realized_gate <= 1.0).all()
                and _finite(realized_scaled_gain)
                and _finite(realized_bonus)
                and np.allclose(
                    realized_scaled_gain, realized_gain / 0.002,
                    atol=3e-6, rtol=1e-6)
                and np.allclose(
                    realized_bonus, 0.02 * realized_scaled_gain,
                    atol=1e-7, rtol=1e-6))
        else:
            objective_contract = bool(
                (policy_mode
                 == "analytic_two_sided_zero_hold_regret_dual_v2").all()
                and (gain_enabled == 0.0).all()
                and (gain_mode == "disabled").all()
                and np.allclose(gain_weight, 0.0)
                and np.allclose(realized_gain, 0.0)
                and np.allclose(realized_bonus, 0.0))

        anchor_holding = _mean(
            per_eval, anchor, "holding_vehicle_seconds")
        anchor_denied = _mean(
            per_eval, anchor, "fleet_denied_dispatch_events")
        anchor_action = _mean(per_eval, anchor, "lower_action_mean")
        candidate_holding = float(numeric(
            "holding_vehicle_seconds").mean())
        candidate_denied = float(numeric(
            "fleet_denied_dispatch_events").mean())
        candidate_action = float(numeric("lower_action_mean").mean())
        mechanism_checks = {
            "paired_rollouts_complete": all(
                count == expected_pairs for count in pair_counts),
            "scalar_control_critics_locked": scalar_critic_locked,
            "candidate_critic_locked": bool(
                (rows["lower_discrete_critic"].astype(str)
                 == expected_critic).all()),
            "frozen_evaluation_locked": bool(
                _finite(policy_frozen) and (policy_frozen == 1.0).all()
                and _finite(critic_frozen) and (critic_frozen == 1.0).all()
                and _finite(upper_frozen) and (upper_frozen == 1.0).all()),
            "zero_execution_adjustment": bool(
                _finite(adjustment) and (adjustment.abs() <= 1e-12).all()),
            "zero_hold_regret_semantics_locked": bool(
                (enabled == 1.0).all()
                and (constraint_cost_mode == "zero_hold_regret_v2").all()),
            "dimensionless_regret_constraint": bool(
                (scale_mode == "cost_limit_ratio_v1").all()
                and np.allclose(initial, 0.01)
                and np.allclose(cost_limit, 0.00025)
                and np.allclose(scaled_limit, 1.0)),
            "inherited_objective_contract_locked": objective_contract,
            "regret_limit_satisfied_every_rollout": bool(
                _finite(action_regret)
                and (action_regret <= 0.00025 + 1e-12).all()),
            "causal_evidence_coverage": bool(
                _finite(evidence) and (evidence >= 0.50).all()),
        }
        outcome_checks = {
            "journey_improves_matched_scalar_anchor": (
                anchor_delta["journey_min"] <= -0.05),
            "cv_improves_matched_scalar_anchor": (
                anchor_delta["headway_cv"] <= -0.001),
            "holding_does_not_increase_vs_anchor": (
                candidate_holding <= anchor_holding),
            "denied_does_not_increase_vs_anchor": (
                candidate_denied <= anchor_denied),
            "action_does_not_increase_vs_anchor": (
                candidate_action <= anchor_action),
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
            "matched_scalar_anchor": anchor,
            "expected_discrete_critic": expected_critic,
            "mechanism_checks": mechanism_checks,
            "outcome_checks": outcome_checks,
            "passes": (
                all(mechanism_checks.values())
                and all(outcome_checks.values())),
            "anchor_delta_headway_cv_mean": anchor_delta["headway_cv"],
            "anchor_delta_journey_min_mean": anchor_delta["journey_min"],
            "control_deltas": comparisons,
            "holding_vehicle_seconds_mean": candidate_holding,
            "fleet_denied_dispatch_events_mean": candidate_denied,
            "lower_action_mean_s": candidate_action,
            "action_regret_mean": float(action_regret.mean()),
            "causal_evidence_min": float(evidence.min()),
        })

    passing = {
        result["config"] for result in candidate_results if result["passes"]}
    selected = next((name for name in PRIORITY if name in passing), None)
    return {
        "gate_version": "freqduet-v19-lower-discrete-critic-screen-v1",
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
    result = evaluate_discrete_critic_screen(args.aggregate_dir)
    out = args.out or args.aggregate_dir / "discrete_critic_screen.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and result[
            "status"] != "exploratory_candidate_selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
