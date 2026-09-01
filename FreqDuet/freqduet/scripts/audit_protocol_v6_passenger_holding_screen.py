#!/usr/bin/env python3
"""Audit the preregistered V20 causal APC passenger-holding screen."""

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
PASSENGER_SCALAR_SPECS = [
    (
        "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_"
        f"paxdual_b{label}_hiro",
        budget,
    )
    for label, budget in (("040", 0.04), ("060", 0.06), ("080", 0.08))
]
CANDIDATE_SPECS = [
    (
        "F_freqduet_protocol_v6_w2adregret_l001_e25_r00025_"
        f"paxdual_b{label}_qadv0_hiro",
        scalar,
        budget,
    )
    for (label, budget), (scalar, _) in zip(
        (("040", 0.04), ("060", 0.06), ("080", 0.08)),
        PASSENGER_SCALAR_SPECS,
    )
]
PASSENGER_CONFIGS = [
    name for name, _ in PASSENGER_SCALAR_SPECS
] + [name for name, _, _ in CANDIDATE_SPECS]
CANDIDATES = [name for name, _, _ in CANDIDATE_SPECS]
CONFIGS = [
    HARD_MAIN,
    REFERENCE,
    MATCHED_CONTEXT,
    CURRENT_MAIN,
    SAME_ENTROPY,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    *[name for name, _ in PASSENGER_SCALAR_SPECS],
    *CANDIDATES,
]
TRAIN_SEEDS = [25013, 25031, 25053, 25077]
EVAL_SEEDS = [58017, 58041, 58059, 58083]
PRIORITY = [name for name, _, _ in reversed(CANDIDATE_SPECS)]


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


def _passenger_mechanism_checks(
    rows: pd.DataFrame,
    *,
    budget: float,
    expected_critic: str,
) -> dict[str, bool]:
    numeric = lambda column: pd.to_numeric(rows[column], errors="coerce")
    policy_frozen = numeric("lower_policy_frozen")
    critic_frozen = numeric("lower_critic_frozen")
    upper_frozen = numeric("upper_policy_frozen")
    adjustment = numeric("lower_causal_guard_adjustment_mean_s")
    regret = numeric("lower_regularity_policy_action_regret_mean")
    evidence = numeric("lower_regularity_policy_evidence_valid_mean")
    passenger_enabled = numeric(
        "lower_regularity_passenger_holding_enabled")
    initial_lambda = numeric("lower_regularity_passenger_initial_lambda")
    cost_limit = numeric("lower_regularity_passenger_cost_limit")
    scaled_limit = numeric("lower_regularity_passenger_scaled_limit")
    expected_cost = numeric(
        "lower_regularity_passenger_expected_cost_mean")
    selected_cost = numeric(
        "lower_regularity_passenger_selected_cost_mean")
    load = numeric("lower_regularity_passenger_load_mean")
    passenger_lambda = numeric("lower_regularity_passenger_lambda")
    return {
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
            (numeric("lower_regularity_policy_enabled") == 1.0).all()
            and (rows["lower_regularity_policy_mode"].astype(str)
                 == "analytic_two_sided_zero_hold_regret_dual_v2").all()
            and (rows[
                "lower_regularity_policy_constraint_cost_mode"].astype(str)
                 == "zero_hold_regret_v2").all()
            and (rows[
                "lower_regularity_policy_constraint_scale_mode"].astype(str)
                 == "cost_limit_ratio_v1").all()
            and np.allclose(
                numeric("lower_regularity_policy_initial_lambda"), 0.01)
            and np.allclose(
                numeric("lower_regularity_policy_cost_limit"), 0.00025)
            and np.allclose(
                numeric("lower_regularity_policy_scaled_limit"), 1.0)),
        "regret_limit_satisfied_every_rollout": bool(
            _finite(regret) and (regret <= 0.00025 + 1e-12).all()),
        "causal_evidence_coverage": bool(
            _finite(evidence) and (evidence >= 0.50).all()),
        "passenger_contract_locked": bool(
            (passenger_enabled == 1.0).all()
            and (rows[
                "lower_regularity_passenger_holding_mode"].astype(str)
                 == "causal_apc_person_delay_dual_v1").all()
            and (rows[
                "lower_regularity_passenger_constraint_scale_mode"].astype(str)
                 == "cost_limit_ratio_v1").all()
            and np.allclose(initial_lambda, 0.01)
            and np.allclose(cost_limit, budget)
            and np.allclose(scaled_limit, 1.0)),
        "expected_passenger_budget_satisfied_every_rollout": bool(
            _finite(expected_cost)
            and (expected_cost >= 0.0).all()
            and (expected_cost <= budget + 1e-12).all()),
        "passive_passenger_telemetry_active": bool(
            _finite(selected_cost) and (selected_cost >= 0.0).all()
            and _finite(load) and (load > 0.0).all()
            and (load <= 1.0 + 1e-12).all()),
        "passenger_dual_finite_and_bounded": bool(
            _finite(passenger_lambda)
            and (passenger_lambda >= 0.0001 - 1e-12).all()
            and (passenger_lambda <= 2.0 + 1e-12).all()),
    }


def evaluate_passenger_holding_screen(
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
        raise FileNotFoundError(f"missing V20 aggregate artifacts: {missing}")

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
        raise ValueError(f"V20 strict checks failed: {strict_checks}")

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
        raise ValueError(f"V20 metrics are missing: {missing_columns}")

    expected_critics = {
        config: (
            "zero_hold_advantage"
            if config == V13_ZERO_HOLD_ADVANTAGE or config in CANDIDATES
            else "continuous_action")
        for config in CONFIGS
    }
    critic_contract_locked = all(
        (_rows(per_eval, config)["lower_discrete_critic"].astype(str)
         == expected_critic).all()
        for config, expected_critic in expected_critics.items()
    )
    nonpassenger_contract_locked = all(
        (pd.to_numeric(
            _rows(per_eval, config)[
                "lower_regularity_passenger_holding_enabled"],
            errors="coerce") == 0.0).all()
        for config in CONFIGS if config not in PASSENGER_CONFIGS
    )

    mechanism_controls = []
    scalar_checks: dict[str, dict[str, bool]] = {}
    for config, budget in PASSENGER_SCALAR_SPECS:
        rows = _rows(per_eval, config)
        checks = _passenger_mechanism_checks(
            rows, budget=budget, expected_critic="continuous_action")
        scalar_checks[config] = checks
        mechanism_controls.append({
            "config": config,
            "budget": budget,
            "mechanism_checks": checks,
            "passes": all(checks.values()),
            "expected_passenger_cost_max": float(pd.to_numeric(
                rows["lower_regularity_passenger_expected_cost_mean"],
                errors="coerce").max()),
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
    for candidate, scalar_passenger, budget in CANDIDATE_SPECS:
        rows = _rows(per_eval, candidate)
        if len(rows) != expected_pairs:
            raise ValueError(f"incomplete V20 candidate: {candidate}")
        aggregate_pair = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == V13_ANCHOR)]
        if len(aggregate_pair) != 1 or int(
                aggregate_pair.iloc[0]["n_pairs"]) != expected_pairs:
            raise ValueError(f"incomplete V20 aggregate pair: {candidate}")

        base_delta, base_counts = _paired_metrics(
            per_eval, candidate, V13_ANCHOR)
        qadv_delta, qadv_counts = _paired_metrics(
            per_eval, candidate, V13_ZERO_HOLD_ADVANTAGE)
        scalar_delta, scalar_counts = _paired_metrics(
            per_eval, candidate, scalar_passenger)
        comparisons = {}
        pair_counts = base_counts + qadv_counts + scalar_counts
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
        mechanism_checks = _passenger_mechanism_checks(
            rows, budget=budget, expected_critic="zero_hold_advantage")
        mechanism_checks.update({
            "paired_rollouts_complete": all(
                count == expected_pairs for count in pair_counts),
            "matrix_critic_contract_locked": critic_contract_locked,
            "nonpassenger_contract_locked": nonpassenger_contract_locked,
            "matched_scalar_mechanism_control_valid": all(
                scalar_checks[scalar_passenger].values()),
        })
        outcome_checks = {
            "journey_improves_v13_scalar_anchor": (
                base_delta["journey_min"] <= -0.05),
            "cv_improves_v13_scalar_anchor": (
                base_delta["headway_cv"] <= -0.001),
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
            "cv_noninferior_to_unbounded_qadv": (
                qadv_delta["headway_cv"] <= 0.002),
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
            "matched_scalar_passenger_control": scalar_passenger,
            "passenger_budget": budget,
            "mechanism_checks": mechanism_checks,
            "outcome_checks": outcome_checks,
            "passes": (
                all(mechanism_checks.values())
                and all(outcome_checks.values())),
            "v13_delta": base_delta,
            "unbounded_qadv_delta": qadv_delta,
            "matched_scalar_passenger_delta": scalar_delta,
            "control_deltas": comparisons,
            "holding_vehicle_seconds_mean": candidate_holding,
            "fleet_denied_dispatch_events_mean": candidate_denied,
            "lower_action_mean_s": candidate_action,
            "expected_passenger_cost_max": float(numeric(
                "lower_regularity_passenger_expected_cost_mean").max()),
        })

    passing = {
        result["config"] for result in candidate_results if result["passes"]}
    selected = next((name for name in PRIORITY if name in passing), None)
    return {
        "gate_version": "freqduet-v20-apc-passenger-holding-screen-v1",
        "status": (
            "exploratory_candidate_selected" if selected else "no_pass"),
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "strict_checks": strict_checks,
        "resource_limits": {
            "holding_vehicle_seconds_mean_max": holding_limit,
            "fleet_denied_dispatch_events_mean_max": denied_limit,
        },
        "mechanism_control_results": mechanism_controls,
        "candidate_results": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_passenger_holding_screen(args.aggregate_dir)
    out = args.out or args.aggregate_dir / "passenger_holding_screen.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_pass and result[
            "status"] != "exploratory_candidate_selected":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
