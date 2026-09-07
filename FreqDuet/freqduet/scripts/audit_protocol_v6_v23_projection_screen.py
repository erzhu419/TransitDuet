#!/usr/bin/env python3
"""Audit the preregistered V23 exact categorical projection screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_aggregate_gain_screen import (
    V13_ZERO_HOLD_ADVANTAGE,
    V20_QADV_B080,
)
from scripts.audit_protocol_v6_capacity_gain_screen import (
    CURRENT_MAIN,
    HARD_MAIN,
    REFERENCE,
    V13_ANCHOR,
    _finite,
    _mean,
    _paired_delta,
    _rows,
)


CANDIDATE = "F_freqduet_protocol_v6_v23_jointproj_r036_p075_hiro"
CONFIGS = [
    HARD_MAIN,
    CURRENT_MAIN,
    REFERENCE,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    V20_QADV_B080,
    CANDIDATE,
]
TRAIN_SEEDS = [28013, 28031, 28053, 28077]
EVAL_SEEDS = [61017, 61041, 61059, 61083]
REGULARITY_REPLAY_TARGET = 0.036
PASSENGER_REPLAY_TARGET = 0.075
REGULARITY_FROZEN_BUDGET = 0.05
PASSENGER_FROZEN_BUDGET = 0.08
PROJECTION_TOLERANCE = 1e-8


def _numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(rows[column], errors="coerce")


def _all_close(rows: pd.DataFrame, column: str, value: float) -> bool:
    values = _numeric(rows, column)
    return bool(_finite(values) and np.allclose(
        values.to_numpy(dtype=float), value, rtol=0.0, atol=1e-12))


def _all_zero(rows: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(
        _finite(_numeric(rows, column))
        and (_numeric(rows, column).abs() <= 1e-12).all()
        for column in columns
    )


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


def _paired_metrics(
    per_eval: pd.DataFrame,
    reference: str,
) -> tuple[dict[str, float], list[int]]:
    deltas: dict[str, float] = {}
    counts: list[int] = []
    for label, metric in {
        "headway_cv": "headway_cv",
        "journey_min": "restricted_total_journey_horizon_min",
    }.items():
        deltas[label], count = _paired_delta(
            per_eval, CANDIDATE, reference, metric)
        counts.append(count)
    return deltas, counts


def _load_training_rows(log_roots: Iterable[Path]) -> pd.DataFrame:
    roots = [Path(root).resolve() for root in log_roots]
    if not roots:
        raise ValueError("V23 audit requires at least one training log root")

    frames = []
    for train_seed in TRAIN_SEEDS:
        relative = Path(f"{CANDIDATE}_seed{train_seed}") / "diagnostics.csv"
        matches = [root / relative for root in roots if (root / relative).is_file()]
        if len(matches) != 1:
            raise ValueError(
                f"expected one V23 diagnostics file for seed {train_seed}, "
                f"found {matches}")
        frame = pd.read_csv(matches[0])
        if "ep" not in frame.columns:
            raise ValueError(f"{matches[0]}: missing ep column")
        episodes = _numeric(frame, "ep")
        if (len(frame) != 40 or not _finite(episodes)
                or set(episodes.astype(int)) != set(range(40))):
            raise ValueError(
                f"{matches[0]}: expected exactly episodes 0--39")
        frame = frame.copy()
        frame["train_seed"] = int(train_seed)
        frame["diagnostics_path"] = str(matches[0])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _candidate_eval_checks(rows: pd.DataFrame) -> dict[str, bool]:
    required_gain = _numeric(
        rows, "lower_regularity_gain_floor_required_gain_mean")
    absolute_shortfall = _numeric(
        rows,
        "lower_regularity_gain_floor_expected_absolute_shortfall_mean",
    )
    aggregate_ratio = _numeric(
        rows,
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
    )
    passenger_cost = _numeric(
        rows, "lower_regularity_passenger_expected_cost_mean")
    evidence = _numeric(
        rows, "lower_regularity_policy_evidence_valid_mean")

    return {
        "zero_hold_advantage_critic_locked": bool(
            (rows["lower_discrete_critic"].astype(str)
             == "zero_hold_advantage").all()),
        "exact_projection_contract_locked": bool(
            (_numeric(rows, "lower_regularity_policy_enabled") == 1.0).all()
            and (rows["lower_regularity_policy_mode"].astype(str)
                 == "analytic_two_sided_hf_aggregate_gain_projection_v10").all()
            and (rows[
                "lower_regularity_policy_constraint_cost_mode"].astype(str)
                 == "hf_aggregate_gain_shortfall_v4").all()
            and (rows[
                "lower_regularity_policy_constraint_scale_mode"].astype(str)
                 == "cost_limit_ratio_v1").all()
            and (rows[
                "lower_regularity_policy_dual_update_mode"].astype(str)
                 == "exact_projection_v1").all()
            and _all_close(
                rows, "lower_regularity_policy_augmented_lagrangian_rho", 0.0)
            and _all_close(
                rows, "lower_regularity_policy_cost_limit",
                REGULARITY_FROZEN_BUDGET)
            and (_numeric(rows, "lower_regularity_projection_enabled")
                 == 1.0).all()
            and (rows["lower_regularity_projection_mode"].astype(str)
                 == "joint_kl_soft_policy_target_v1").all()
            and _all_close(
                rows, "lower_regularity_projection_regularity_target",
                REGULARITY_REPLAY_TARGET)
            and _all_close(
                rows, "lower_regularity_projection_passenger_target",
                PASSENGER_REPLAY_TARGET)),
        "aggregate_gain_floor_contract_locked": bool(
            (_numeric(rows, "lower_regularity_gain_floor_enabled")
             == 1.0).all()
            and (rows["lower_regularity_gain_floor_mode"].astype(str)
                 == "causal_hf_aggregate_gain_floor_v2").all()
            and _all_close(
                rows, "lower_regularity_gain_floor_base_fraction", 0.30)
            and _all_close(
                rows, "lower_regularity_gain_floor_hf_increment", 0.30)
            and _all_close(
                rows, "lower_regularity_gain_floor_hf_energy_scale", 0.04)
            and _all_close(
                rows, "lower_regularity_gain_floor_hf_energy_exponent", 1.0)),
        "conditional_entropy_contract_locked": bool(
            (_numeric(rows, "lower_regularity_entropy_split_enabled")
             == 1.0).all()
            and _all_close(
                rows, "lower_regularity_entropy_target_fraction", 0.25)),
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
                 == "exact_projection_v1").all()
            and _all_close(
                rows,
                "lower_regularity_passenger_augmented_lagrangian_rho", 0.0)
            and _all_close(
                rows, "lower_regularity_passenger_cost_limit",
                PASSENGER_FROZEN_BUDGET)),
        "soft_duals_and_penalties_absent": bool(
            _all_zero(rows, (
                "lower_regularity_lambda",
                "lower_regularity_passenger_lambda",
                "lower_regularity_policy_penalty",
                "lower_regularity_policy_augmented_penalty",
                "lower_regularity_passenger_actor_penalty",
                "lower_regularity_passenger_actor_augmented_penalty",
            ))),
        "frozen_evaluation_locked": bool(
            (_numeric(rows, "lower_policy_frozen") == 1.0).all()
            and (_numeric(rows, "lower_critic_frozen") == 1.0).all()
            and (_numeric(rows, "upper_policy_frozen") == 1.0).all()
            and (_numeric(rows, "lower_regularity_projection_applied")
                 == 0.0).all()),
        "zero_execution_adjustment": bool(
            _all_zero(rows, ("lower_causal_guard_adjustment_mean_s",))),
        "aggregate_numerator_denominator_identity": bool(
            _finite(required_gain)
            and (required_gain > 0.0).all()
            and _finite(absolute_shortfall)
            and (absolute_shortfall >= 0.0).all()
            and (absolute_shortfall <= required_gain + 1e-8).all()
            and _ratio_identity(
                absolute_shortfall, required_gain, aggregate_ratio)),
        "causal_evidence_coverage": bool(
            _finite(evidence) and (evidence >= 0.50).all()),
        "frozen_regularity_budget": bool(
            _finite(aggregate_ratio)
            and (aggregate_ratio >= 0.0).all()
            and (aggregate_ratio
                 <= REGULARITY_FROZEN_BUDGET + 1e-12).all()),
        "frozen_passenger_budget": bool(
            _finite(passenger_cost)
            and (passenger_cost >= 0.0).all()
            and (passenger_cost
                 <= PASSENGER_FROZEN_BUDGET + 1e-12).all()),
    }


def _training_checks(rows: pd.DataFrame) -> tuple[dict[str, bool], dict[str, float]]:
    applied = rows.loc[
        _numeric(rows, "lower_regularity_projection_applied") == 1.0
    ].copy()
    applied_by_seed = applied.groupby("train_seed").size().to_dict()
    active_aggregate = rows.loc[
        _numeric(rows, "lower_regularity_gain_floor_required_gain_mean")
        > 1e-12
    ].copy()

    projection_finite_columns = (
        "lower_regularity_projection_iterations",
        "lower_regularity_projection_valid_count",
        "lower_regularity_projection_base_regularity_cost",
        "lower_regularity_projection_base_passenger_cost",
        "lower_regularity_projection_target_regularity_cost",
        "lower_regularity_projection_target_passenger_cost",
        "lower_regularity_projection_regularity_multiplier",
        "lower_regularity_projection_passenger_multiplier",
        "lower_regularity_projection_target_kl_from_soft",
        "lower_regularity_projection_target_entropy",
        "lower_regularity_projection_actor_reverse_kl",
        "lower_regularity_projection_base_action_mean_s",
        "lower_regularity_projection_target_action_mean_s",
        "lower_regularity_projection_target_action_change_mean_s",
        "lower_regularity_projection_max_constraint_violation",
    )
    required_gain = _numeric(
        active_aggregate, "lower_regularity_gain_floor_required_gain_mean")
    absolute_shortfall = _numeric(
        active_aggregate,
        "lower_regularity_gain_floor_expected_absolute_shortfall_mean",
    )
    aggregate_ratio = _numeric(
        active_aggregate,
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
    )

    checks = {
        "complete_training_diagnostics": bool(
            len(rows) == len(TRAIN_SEEDS) * 40
            and rows.groupby("train_seed")["ep"].nunique().eq(40).all()),
        "projection_applied_for_every_seed": bool(
            len(applied) > 0
            and all(applied_by_seed.get(seed, 0) > 0 for seed in TRAIN_SEEDS)),
        "every_recorded_projection_converged": bool(
            len(applied) > 0
            and (_numeric(
                applied, "lower_regularity_projection_converged") == 1.0).all()),
        "every_recorded_projection_meets_replay_targets": bool(
            len(applied) > 0
            and (_numeric(
                applied,
                "lower_regularity_projection_target_regularity_cost")
                 <= REGULARITY_REPLAY_TARGET + PROJECTION_TOLERANCE).all()
            and (_numeric(
                applied,
                "lower_regularity_projection_target_passenger_cost")
                 <= PASSENGER_REPLAY_TARGET + PROJECTION_TOLERANCE).all()
            and (_numeric(
                applied,
                "lower_regularity_projection_max_constraint_violation")
                 <= PROJECTION_TOLERANCE).all()),
        "projection_diagnostics_finite_and_bounded": bool(
            len(applied) > 0
            and all(_finite(_numeric(applied, column))
                    for column in projection_finite_columns)
            and (_numeric(
                applied, "lower_regularity_projection_iterations")
                 >= 0.0).all()
            and (_numeric(
                applied, "lower_regularity_projection_iterations")
                 <= 200.0).all()
            and (_numeric(
                applied, "lower_regularity_projection_valid_count")
                 > 0.0).all()
            and (_numeric(
                applied, "lower_regularity_projection_regularity_multiplier")
                 >= 0.0).all()
            and (_numeric(
                applied, "lower_regularity_projection_passenger_multiplier")
                 >= 0.0).all()
            and (_numeric(
                applied, "lower_regularity_projection_target_kl_from_soft")
                 >= -1e-12).all()
            and (_numeric(
                applied, "lower_regularity_projection_target_entropy")
                 >= -1e-12).all()),
        "training_contract_locked": bool(
            (rows["lower_regularity_policy_mode"].astype(str)
             == "analytic_two_sided_hf_aggregate_gain_projection_v10").all()
            and (rows[
                "lower_regularity_policy_dual_update_mode"].astype(str)
                 == "exact_projection_v1").all()
            and (rows[
                "lower_regularity_passenger_dual_update_mode"].astype(str)
                 == "exact_projection_v1").all()
            and (rows["lower_regularity_projection_mode"].astype(str)
                 == "joint_kl_soft_policy_target_v1").all()
            and _all_close(
                rows, "lower_regularity_projection_regularity_target",
                REGULARITY_REPLAY_TARGET)
            and _all_close(
                rows, "lower_regularity_projection_passenger_target",
                PASSENGER_REPLAY_TARGET)),
        "training_soft_duals_and_penalties_absent": bool(
            _all_zero(rows, (
                "lower_regularity_lambda",
                "lower_regularity_passenger_lambda",
                "lower_regularity_policy_penalty",
                "lower_regularity_policy_augmented_penalty",
                "lower_regularity_passenger_actor_penalty",
                "lower_regularity_passenger_actor_augmented_penalty",
            ))),
        "training_zero_execution_adjustment": bool(
            _all_zero(rows, ("lower_causal_guard_adjustment_mean_s",))),
        "training_aggregate_telemetry_identity": bool(
            len(active_aggregate) > 0
            and _finite(required_gain)
            and (required_gain > 0.0).all()
            and _finite(absolute_shortfall)
            and (absolute_shortfall >= 0.0).all()
            and (absolute_shortfall <= required_gain + 1e-8).all()
            and _ratio_identity(
                absolute_shortfall, required_gain, aggregate_ratio)),
    }
    diagnostics = {
        "training_rows": int(len(rows)),
        "projection_rows": int(len(applied)),
        "projection_rows_min_per_seed": int(min(
            applied_by_seed.get(seed, 0) for seed in TRAIN_SEEDS)),
        "projection_target_regularity_cost_max": float(_numeric(
            applied,
            "lower_regularity_projection_target_regularity_cost").max()),
        "projection_target_passenger_cost_max": float(_numeric(
            applied,
            "lower_regularity_projection_target_passenger_cost").max()),
        "projection_constraint_violation_max": float(_numeric(
            applied,
            "lower_regularity_projection_max_constraint_violation").max()),
        "projection_iterations_max": float(_numeric(
            applied, "lower_regularity_projection_iterations").max()),
        "projection_kl_max": float(_numeric(
            applied,
            "lower_regularity_projection_target_kl_from_soft").max()),
    }
    return checks, diagnostics


def evaluate_v23_projection_screen(
    aggregate_dir: Path,
    log_roots: Iterable[Path],
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
        raise FileNotFoundError(f"missing V23 aggregate artifacts: {missing}")

    manifest = json.loads(paths["manifest"].read_text())
    per_eval = pd.read_csv(paths["per_eval"])
    paired = pd.read_csv(paths["paired"])
    training = _load_training_rows(log_roots)
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
        raise ValueError(f"V23 strict checks failed: {strict_checks}")

    required_eval_columns = {
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
        "lower_regularity_policy_cost_limit",
        "lower_regularity_policy_evidence_valid_mean",
        "lower_regularity_policy_penalty",
        "lower_regularity_policy_augmented_penalty",
        "lower_regularity_lambda",
        "lower_regularity_gain_floor_enabled",
        "lower_regularity_gain_floor_mode",
        "lower_regularity_gain_floor_base_fraction",
        "lower_regularity_gain_floor_hf_increment",
        "lower_regularity_gain_floor_hf_energy_scale",
        "lower_regularity_gain_floor_hf_energy_exponent",
        "lower_regularity_gain_floor_required_gain_mean",
        "lower_regularity_gain_floor_expected_absolute_shortfall_mean",
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
        "lower_regularity_entropy_split_enabled",
        "lower_regularity_entropy_target_fraction",
        "lower_regularity_passenger_holding_enabled",
        "lower_regularity_passenger_holding_mode",
        "lower_regularity_passenger_constraint_scale_mode",
        "lower_regularity_passenger_dual_update_mode",
        "lower_regularity_passenger_augmented_lagrangian_rho",
        "lower_regularity_passenger_cost_limit",
        "lower_regularity_passenger_expected_cost_mean",
        "lower_regularity_passenger_actor_penalty",
        "lower_regularity_passenger_actor_augmented_penalty",
        "lower_regularity_passenger_lambda",
        "lower_regularity_projection_enabled",
        "lower_regularity_projection_mode",
        "lower_regularity_projection_applied",
        "lower_regularity_projection_regularity_target",
        "lower_regularity_projection_passenger_target",
    }
    missing_columns = sorted(required_eval_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(f"V23 evaluation metrics are missing: {missing_columns}")

    required_training_columns = {
        "ep", "lower_causal_guard_adjustment_mean_s",
        "lower_regularity_policy_mode",
        "lower_regularity_policy_dual_update_mode",
        "lower_regularity_policy_penalty",
        "lower_regularity_policy_augmented_penalty",
        "lower_regularity_lambda",
        "lower_regularity_gain_floor_enabled",
        "lower_regularity_gain_floor_mode",
        "lower_regularity_gain_floor_base_fraction",
        "lower_regularity_gain_floor_hf_increment",
        "lower_regularity_gain_floor_hf_energy_scale",
        "lower_regularity_gain_floor_hf_energy_exponent",
        "lower_regularity_gain_floor_required_gain_mean",
        "lower_regularity_gain_floor_expected_absolute_shortfall_mean",
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
        "lower_regularity_entropy_split_enabled",
        "lower_regularity_entropy_target_fraction",
        "lower_regularity_passenger_dual_update_mode",
        "lower_regularity_passenger_actor_penalty",
        "lower_regularity_passenger_actor_augmented_penalty",
        "lower_regularity_passenger_lambda",
        "lower_regularity_projection_mode",
        "lower_regularity_projection_applied",
        "lower_regularity_projection_regularity_target",
        "lower_regularity_projection_passenger_target",
        "lower_regularity_projection_converged",
        "lower_regularity_projection_iterations",
        "lower_regularity_projection_valid_count",
        "lower_regularity_projection_base_regularity_cost",
        "lower_regularity_projection_base_passenger_cost",
        "lower_regularity_projection_target_regularity_cost",
        "lower_regularity_projection_target_passenger_cost",
        "lower_regularity_projection_regularity_multiplier",
        "lower_regularity_projection_passenger_multiplier",
        "lower_regularity_projection_target_kl_from_soft",
        "lower_regularity_projection_target_entropy",
        "lower_regularity_projection_actor_reverse_kl",
        "lower_regularity_projection_base_action_mean_s",
        "lower_regularity_projection_target_action_mean_s",
        "lower_regularity_projection_target_action_change_mean_s",
        "lower_regularity_projection_max_constraint_violation",
    }
    missing_training_columns = sorted(
        required_training_columns - set(training.columns))
    if missing_training_columns:
        raise ValueError(
            f"V23 training metrics are missing: {missing_training_columns}")

    candidate_rows = _rows(per_eval, CANDIDATE)
    if len(candidate_rows) != expected_pairs:
        raise ValueError("incomplete V23 candidate evaluation rows")
    aggregate_pair = paired.loc[
        (paired["candidate"] == CANDIDATE)
        & (paired["reference"] == V13_ANCHOR)]
    if (len(aggregate_pair) != 1
            or int(aggregate_pair.iloc[0]["n_pairs"]) != expected_pairs):
        raise ValueError("incomplete V23 candidate paired artifact")

    control_projection_disabled = all(
        (_numeric(
            _rows(per_eval, config),
            "lower_regularity_projection_enabled") == 0.0).all()
        for config in CONFIGS if config != CANDIDATE
    )
    mechanism_checks = _candidate_eval_checks(candidate_rows)
    training_checks, training_diagnostics = _training_checks(training)
    mechanism_checks.update(training_checks)
    mechanism_checks["control_projection_disabled"] = bool(
        control_projection_disabled)

    v13_delta, pair_counts = _paired_metrics(per_eval, V13_ANCHOR)
    v19_delta, counts = _paired_metrics(
        per_eval, V13_ZERO_HOLD_ADVANTAGE)
    pair_counts.extend(counts)
    v20_delta, counts = _paired_metrics(per_eval, V20_QADV_B080)
    pair_counts.extend(counts)
    current_delta, counts = _paired_metrics(per_eval, CURRENT_MAIN)
    pair_counts.extend(counts)
    noguard_delta, counts = _paired_metrics(per_eval, REFERENCE)
    pair_counts.extend(counts)
    mechanism_checks["paired_rollouts_complete"] = all(
        count == expected_pairs for count in pair_counts)

    candidate_action = float(_numeric(
        candidate_rows, "lower_action_mean").mean())
    candidate_holding = float(_numeric(
        candidate_rows, "holding_vehicle_seconds").mean())
    candidate_denied = float(_numeric(
        candidate_rows, "fleet_denied_dispatch_events").mean())
    outcome_checks = {
        "journey_improves_v13_scalar_anchor": (
            v13_delta["journey_min"] <= -0.05),
        "cv_improves_v13_scalar_anchor": (
            v13_delta["headway_cv"] <= -0.001),
        "action_does_not_increase_vs_v13": (
            candidate_action <= _mean(
                per_eval, V13_ANCHOR, "lower_action_mean")),
        "holding_does_not_increase_vs_v13": (
            candidate_holding <= _mean(
                per_eval, V13_ANCHOR, "holding_vehicle_seconds")),
        "denied_does_not_increase_vs_v13": (
            candidate_denied <= _mean(
                per_eval, V13_ANCHOR, "fleet_denied_dispatch_events")),
        "journey_improves_v19_zero_hold_advantage": (
            v19_delta["journey_min"] <= -0.05),
        "cv_improves_v19_zero_hold_advantage": (
            v19_delta["headway_cv"] <= -0.001),
        "action_does_not_increase_vs_v19": (
            candidate_action <= _mean(
                per_eval, V13_ZERO_HOLD_ADVANTAGE, "lower_action_mean")),
        "holding_does_not_increase_vs_v19": (
            candidate_holding <= _mean(
                per_eval, V13_ZERO_HOLD_ADVANTAGE,
                "holding_vehicle_seconds")),
        "denied_does_not_increase_vs_v19": (
            candidate_denied <= _mean(
                per_eval, V13_ZERO_HOLD_ADVANTAGE,
                "fleet_denied_dispatch_events")),
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
    passes = bool(
        all(mechanism_checks.values()) and all(outcome_checks.values()))
    return {
        "gate_version": "freqduet-v23-exact-projection-screen-v1",
        "status": "exploratory_candidate_selected" if passes else "no_pass",
        "claim_eligible": False,
        "selected_for_confirmation": CANDIDATE if passes else None,
        "strict_checks": strict_checks,
        "mechanism_checks": mechanism_checks,
        "outcome_checks": outcome_checks,
        "training_diagnostics": training_diagnostics,
        "paired_deltas": {
            "v13_scalar_anchor": v13_delta,
            "v19_zero_hold_advantage": v19_delta,
            "v20_passenger": v20_delta,
            "confirmed_main": current_delta,
            "noguard": noguard_delta,
        },
        "candidate_means": {
            "lower_action_mean_s": candidate_action,
            "holding_vehicle_seconds": candidate_holding,
            "fleet_denied_dispatch_events": candidate_denied,
            "frozen_regularity_cost_max": float(_numeric(
                candidate_rows,
                "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio"
            ).max()),
            "frozen_passenger_cost_max": float(_numeric(
                candidate_rows,
                "lower_regularity_passenger_expected_cost_mean").max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument(
        "--logs-root", action="append", required=True, type=Path,
        help="training log root; repeat when formal shards span directories")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_v23_projection_screen(
        args.aggregate_dir, args.logs_root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    if args.require_pass and result["selected_for_confirmation"] is None:
        raise SystemExit("V23 gate found no promotion candidate")


if __name__ == "__main__":
    main()
