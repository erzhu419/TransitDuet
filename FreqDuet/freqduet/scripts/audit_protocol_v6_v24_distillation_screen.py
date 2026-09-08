#!/usr/bin/env python3
"""Audit the preregistered V24 projected-actor distillation screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

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
    _mean,
    _paired_delta,
    _rows,
)
from scripts.audit_protocol_v6_v23_projection_screen import (
    CANDIDATE as V23_CANDIDATE,
    PASSENGER_FROZEN_BUDGET,
    PASSENGER_REPLAY_TARGET,
    PROJECTION_TOLERANCE,
    REGULARITY_FROZEN_BUDGET,
    REGULARITY_REPLAY_TARGET,
    _all_close,
    _all_zero,
    _finite,
    _numeric,
    _ratio_identity,
)
from scripts.audit_protocol_v6_v24_distillation_smoke import CANDIDATE_SPECS


CANDIDATES = list(CANDIDATE_SPECS)
CONTROLS = [
    HARD_MAIN,
    CURRENT_MAIN,
    REFERENCE,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
    V20_QADV_B080,
    V23_CANDIDATE,
]
CONFIGS = [*CONTROLS, *CANDIDATES]
TRAIN_SEEDS = [29013, 29031, 29053, 29077]
EVAL_SEEDS = [62017, 62041, 62059, 62083]
TRAIN_EPISODES = 40
LATE_EPISODES = set(range(30, 40))
REVERSE_KL_V23_RATIO_LIMIT = 0.75
ACTION_GAP_V23_RATIO_LIMIT = 0.75


def _load_training_rows(log_roots: Iterable[Path]) -> pd.DataFrame:
    roots = [Path(root).resolve() for root in log_roots]
    if not roots:
        raise ValueError("V24 screen requires at least one training log root")
    frames = []
    for config in [V23_CANDIDATE, *CANDIDATES]:
        for train_seed in TRAIN_SEEDS:
            relative = Path(f"{config}_seed{train_seed}") / "diagnostics.csv"
            matches = [root / relative for root in roots
                       if (root / relative).is_file()]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one diagnostics file for {config} seed "
                    f"{train_seed}, found {matches}")
            frame = pd.read_csv(matches[0])
            episodes = _numeric(frame, "ep")
            if (len(frame) != TRAIN_EPISODES or not _finite(episodes)
                    or set(episodes.astype(int)) != set(range(TRAIN_EPISODES))):
                raise ValueError(
                    f"{matches[0]}: expected exactly episodes 0--39")
            frame = frame.copy()
            frame["config"] = config
            frame["train_seed"] = train_seed
            frame["diagnostics_path"] = str(matches[0])
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _paired_metrics(
    per_eval: pd.DataFrame,
    candidate: str,
    reference: str,
) -> tuple[dict[str, float], list[int]]:
    deltas = {}
    counts = []
    for label, metric in {
        "headway_cv": "headway_cv",
        "journey_min": "restricted_total_journey_horizon_min",
    }.items():
        deltas[label], count = _paired_delta(
            per_eval, candidate, reference, metric)
        counts.append(count)
    return deltas, counts


def _evaluation_checks(rows: pd.DataFrame) -> dict[str, bool]:
    required_gain = _numeric(
        rows, "lower_regularity_gain_floor_required_gain_mean")
    absolute_shortfall = _numeric(
        rows,
        "lower_regularity_gain_floor_expected_absolute_shortfall_mean")
    aggregate_ratio = _numeric(
        rows,
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio")
    passenger_cost = _numeric(
        rows, "lower_regularity_passenger_expected_cost_mean")
    evidence = _numeric(
        rows, "lower_regularity_policy_evidence_valid_mean")
    return {
        "exact_v24_contract": bool(
            (rows["lower_discrete_critic"].astype(str)
             == "zero_hold_advantage").all()
            and (rows["lower_regularity_policy_mode"].astype(str)
                 == "analytic_two_sided_hf_aggregate_gain_projection_v11").all()
            and (rows["lower_regularity_projection_mode"].astype(str)
                 == "joint_kl_soft_policy_distillation_v2").all()
            and (_numeric(rows, "lower_regularity_projection_enabled")
                 == 1.0).all()
            and _all_close(
                rows, "lower_regularity_projection_regularity_target",
                REGULARITY_REPLAY_TARGET)
            and _all_close(
                rows, "lower_regularity_projection_passenger_target",
                PASSENGER_REPLAY_TARGET)),
        "aggregate_gain_floor_contract": bool(
            (_numeric(rows, "lower_regularity_gain_floor_enabled")
             == 1.0).all()
            and (rows["lower_regularity_gain_floor_mode"].astype(str)
                 == "causal_hf_aggregate_gain_floor_v2").all()
            and _all_close(
                rows, "lower_regularity_gain_floor_base_fraction", 0.30)
            and _all_close(
                rows, "lower_regularity_gain_floor_hf_increment", 0.30)),
        "passenger_contract": bool(
            (_numeric(rows, "lower_regularity_passenger_holding_enabled")
             == 1.0).all()
            and (rows["lower_regularity_passenger_holding_mode"].astype(str)
                 == "causal_apc_person_delay_dual_v1").all()
            and _all_close(
                rows, "lower_regularity_passenger_cost_limit",
                PASSENGER_FROZEN_BUDGET)),
        "soft_duals_and_penalties_absent": bool(_all_zero(rows, (
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
        "zero_execution_adjustment": bool(_all_zero(
            rows, ("lower_causal_guard_adjustment_mean_s",))),
        "aggregate_numerator_denominator_identity": bool(
            _finite(required_gain) and (required_gain > 0.0).all()
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


def _v23_training_control_checks(rows: pd.DataFrame) -> dict[str, bool]:
    late = rows.loc[_numeric(rows, "ep").astype(int).isin(LATE_EPISODES)]
    return {
        "complete_v23_training_control": bool(
            len(rows) == len(TRAIN_SEEDS) * TRAIN_EPISODES
            and rows.groupby("train_seed")["ep"].nunique().eq(
                TRAIN_EPISODES).all()),
        "v23_contract_locked": bool(
            (rows["lower_regularity_policy_mode"].astype(str)
             == "analytic_two_sided_hf_aggregate_gain_projection_v10").all()
            and (rows["lower_regularity_projection_mode"].astype(str)
                 == "joint_kl_soft_policy_target_v1").all()
            and (rows["lower_regularity_projection_distillation"].astype(str)
                 == "reverse_kl_v1").all()
            and (_numeric(
                rows, "lower_regularity_projection_distillation_steps")
                 == 1.0).all()),
        "v23_late_action_gap_finite": bool(
            len(late) == len(TRAIN_SEEDS) * len(LATE_EPISODES)
            and _finite(_numeric(
                late,
                "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean"))),
    }


def _training_checks(
    rows: pd.DataFrame,
    v23_late: pd.DataFrame,
    distillation: str,
    steps: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    applied = rows.loc[
        _numeric(rows, "lower_regularity_projection_applied") == 1.0
    ].copy()
    late = rows.loc[_numeric(rows, "ep").astype(int).isin(LATE_EPISODES)].copy()
    pre_column = (
        "lower_regularity_projection_actor_forward_kl_episode_mean"
        if distillation == "forward_kl_v2"
        else "lower_regularity_projection_actor_reverse_kl_episode_mean")
    post_column = (
        "lower_regularity_projection_actor_post_forward_kl_episode_mean"
        if distillation == "forward_kl_v2"
        else "lower_regularity_projection_actor_post_reverse_kl_episode_mean")
    pre = _numeric(late, pre_column)
    post = _numeric(late, post_column)
    post_reverse_kl = _numeric(
        late,
        "lower_regularity_projection_actor_post_reverse_kl_episode_mean")
    v23_post_reverse_kl = _numeric(
        v23_late,
        "lower_regularity_projection_actor_post_reverse_kl_episode_mean")
    post_regularity = _numeric(
        late,
        "lower_regularity_projection_actor_post_regularity_cost_episode_mean")
    post_passenger = _numeric(
        late,
        "lower_regularity_projection_actor_post_passenger_cost_episode_mean")
    action_gap = _numeric(
        late,
        "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean")
    v23_action_gap = _numeric(
        v23_late,
        "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean")
    kl_ratio = (
        float(post.mean() / pre.mean())
        if len(late) and float(pre.mean()) > 1e-12 else 0.0)
    reverse_kl_v23_ratio = (
        float(post_reverse_kl.mean() / v23_post_reverse_kl.mean())
        if len(late) and float(v23_post_reverse_kl.mean()) > 1e-12
        else None)
    action_gap_mean = float(action_gap.mean())
    v23_action_gap_mean = float(v23_action_gap.mean())
    if len(late) and v23_action_gap_mean > 1e-12:
        action_gap_ratio = action_gap_mean / v23_action_gap_mean
    elif len(late) and action_gap_mean <= 1e-12:
        action_gap_ratio = 0.0
    else:
        action_gap_ratio = None
    per_seed_regularity = post_regularity.groupby(late["train_seed"]).mean()
    per_seed_passenger = post_passenger.groupby(late["train_seed"]).mean()
    finite_columns = (
        "lower_regularity_projection_iterations",
        "lower_regularity_projection_valid_count",
        "lower_regularity_projection_target_regularity_cost",
        "lower_regularity_projection_target_passenger_cost",
        "lower_regularity_projection_max_constraint_violation",
        "lower_regularity_projection_actor_reverse_kl_episode_mean",
        "lower_regularity_projection_actor_forward_kl_episode_mean",
        "lower_regularity_projection_actor_post_reverse_kl_episode_mean",
        "lower_regularity_projection_actor_post_forward_kl_episode_mean",
        "lower_regularity_projection_actor_post_regularity_cost_episode_mean",
        "lower_regularity_projection_actor_post_passenger_cost_episode_mean",
        "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean",
    )
    checks = {
        "complete_training_diagnostics": bool(
            len(rows) == len(TRAIN_SEEDS) * TRAIN_EPISODES
            and rows.groupby("train_seed")["ep"].nunique().eq(
                TRAIN_EPISODES).all()),
        "projection_applied_for_every_seed": bool(
            len(applied) > 0
            and set(applied["train_seed"].astype(int)) == set(TRAIN_SEEDS)),
        "exact_distillation_contract": bool(
            (rows["lower_regularity_policy_mode"].astype(str)
             == "analytic_two_sided_hf_aggregate_gain_projection_v11").all()
            and (rows["lower_regularity_projection_mode"].astype(str)
                 == "joint_kl_soft_policy_distillation_v2").all()
            and (rows["lower_regularity_projection_distillation"].astype(str)
                 == distillation).all()
            and (_numeric(
                rows, "lower_regularity_projection_distillation_steps")
                 == float(steps)).all()
            and _all_close(
                rows, "lower_regularity_projection_regularity_target",
                REGULARITY_REPLAY_TARGET)
            and _all_close(
                rows, "lower_regularity_projection_passenger_target",
                PASSENGER_REPLAY_TARGET)),
        "every_teacher_projection_converged": bool(
            len(applied) > 0
            and (_numeric(
                applied, "lower_regularity_projection_converged") == 1.0).all()),
        "every_teacher_projection_meets_targets": bool(
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
                applied, "lower_regularity_projection_max_constraint_violation")
                 <= PROJECTION_TOLERANCE).all()),
        "post_actor_diagnostics_finite": bool(
            len(applied) > 0
            and all(_finite(_numeric(applied, column))
                    for column in finite_columns)),
        "late_actor_regularity_within_budget_each_seed": bool(
            len(per_seed_regularity) == len(TRAIN_SEEDS)
            and (per_seed_regularity
                 <= REGULARITY_FROZEN_BUDGET + 1e-12).all()),
        "late_actor_passenger_within_budget_each_seed": bool(
            len(per_seed_passenger) == len(TRAIN_SEEDS)
            and (per_seed_passenger
                 <= PASSENGER_FROZEN_BUDGET + 1e-12).all()),
        "late_configured_kl_nonincreasing": bool(
            _finite(pre) and _finite(post)
            and (post <= pre + 1e-12).all()),
        "late_reverse_kl_improves_v23": bool(
            _finite(post_reverse_kl) and _finite(v23_post_reverse_kl)
            and reverse_kl_v23_ratio is not None
            and reverse_kl_v23_ratio
            <= REVERSE_KL_V23_RATIO_LIMIT + 1e-12),
        "late_action_gap_improves_v23": bool(
            _finite(action_gap) and _finite(v23_action_gap)
            and action_gap_ratio is not None
            and action_gap_ratio <= ACTION_GAP_V23_RATIO_LIMIT + 1e-12),
        "registered_update_count_observed": bool(
            len(applied) > 0
            and (_numeric(
                applied,
                "lower_regularity_projection_actor_distillation_steps")
                 == float(steps)).all()),
        "soft_duals_and_penalties_absent": bool(_all_zero(rows, (
            "lower_regularity_lambda",
            "lower_regularity_passenger_lambda",
            "lower_regularity_policy_penalty",
            "lower_regularity_policy_augmented_penalty",
            "lower_regularity_passenger_actor_penalty",
            "lower_regularity_passenger_actor_augmented_penalty",
        ))),
        "zero_execution_adjustment": bool(_all_zero(
            rows, ("lower_causal_guard_adjustment_mean_s",))),
    }
    diagnostics = {
        "projection_rows": int(len(applied)),
        "late_configured_kl_pre_mean": float(pre.mean()),
        "late_configured_kl_post_mean": float(post.mean()),
        "late_configured_kl_post_pre_ratio": kl_ratio,
        "late_post_reverse_kl_mean": float(post_reverse_kl.mean()),
        "v23_late_post_reverse_kl_mean": float(
            v23_post_reverse_kl.mean()),
        "late_post_reverse_kl_v23_ratio": reverse_kl_v23_ratio,
        "late_action_gap_abs_mean_s": action_gap_mean,
        "v23_late_action_gap_abs_mean_s": v23_action_gap_mean,
        "late_action_gap_v23_ratio": action_gap_ratio,
        "late_post_regularity_cost_max_seed_mean": float(
            per_seed_regularity.max()),
        "late_post_passenger_cost_max_seed_mean": float(
            per_seed_passenger.max()),
    }
    return checks, diagnostics


def _outcome_checks(
    per_eval: pd.DataFrame,
    candidate: str,
    expected_pairs: int,
) -> tuple[dict[str, bool], dict[str, object]]:
    deltas = {}
    counts = []
    references = {
        "v13_scalar_anchor": V13_ANCHOR,
        "v19_zero_hold_advantage": V13_ZERO_HOLD_ADVANTAGE,
        "v20_passenger": V20_QADV_B080,
        "confirmed_main": CURRENT_MAIN,
        "noguard": REFERENCE,
        "v23_projection": V23_CANDIDATE,
    }
    for label, reference in references.items():
        deltas[label], pair_counts = _paired_metrics(
            per_eval, candidate, reference)
        counts.extend(pair_counts)
    rows = _rows(per_eval, candidate)
    action = float(_numeric(rows, "lower_action_mean").mean())
    holding = float(_numeric(rows, "holding_vehicle_seconds").mean())
    denied = float(_numeric(rows, "fleet_denied_dispatch_events").mean())
    checks = {
        "paired_rollouts_complete": all(
            count == expected_pairs for count in counts),
        "journey_improves_v13_scalar_anchor": (
            deltas["v13_scalar_anchor"]["journey_min"] <= -0.05),
        "cv_improves_v13_scalar_anchor": (
            deltas["v13_scalar_anchor"]["headway_cv"] <= -0.001),
        "action_does_not_increase_vs_v13": (
            action <= _mean(per_eval, V13_ANCHOR, "lower_action_mean")),
        "holding_does_not_increase_vs_v13": (
            holding <= _mean(
                per_eval, V13_ANCHOR, "holding_vehicle_seconds")),
        "denied_does_not_increase_vs_v13": (
            denied <= _mean(
                per_eval, V13_ANCHOR, "fleet_denied_dispatch_events")),
        "journey_improves_v19_zero_hold_advantage": (
            deltas["v19_zero_hold_advantage"]["journey_min"] <= -0.05),
        "cv_improves_v19_zero_hold_advantage": (
            deltas["v19_zero_hold_advantage"]["headway_cv"] <= -0.001),
        "action_does_not_increase_vs_v19": (
            action <= _mean(
                per_eval, V13_ZERO_HOLD_ADVANTAGE, "lower_action_mean")),
        "holding_does_not_increase_vs_v19": (
            holding <= _mean(
                per_eval, V13_ZERO_HOLD_ADVANTAGE,
                "holding_vehicle_seconds")),
        "denied_does_not_increase_vs_v19": (
            denied <= _mean(
                per_eval, V13_ZERO_HOLD_ADVANTAGE,
                "fleet_denied_dispatch_events")),
        "cv_recovers_v20_passenger_collapse": (
            deltas["v20_passenger"]["headway_cv"] <= -0.020),
        "journey_noninferior_to_v20_passenger": (
            deltas["v20_passenger"]["journey_min"] <= 0.20),
        "journey_beats_noguard": (
            deltas["noguard"]["journey_min"] <= -0.25),
        "cv_beats_noguard": (
            deltas["noguard"]["headway_cv"] <= -0.030),
        "journey_beats_confirmed_main": (
            deltas["confirmed_main"]["journey_min"] <= -0.50),
        "cv_noninferior_to_confirmed_main": (
            deltas["confirmed_main"]["headway_cv"] <= 0.005),
        "cv_improves_v23_projection": (
            deltas["v23_projection"]["headway_cv"] <= -0.001),
        "journey_noninferior_to_v23_projection": (
            deltas["v23_projection"]["journey_min"] <= 0.20),
    }
    diagnostics = {
        "paired_deltas": deltas,
        "means": {
            "lower_action_mean_s": action,
            "holding_vehicle_seconds": holding,
            "fleet_denied_dispatch_events": denied,
        },
    }
    return checks, diagnostics


def evaluate_v24_distillation_screen(
    aggregate_dir: Path,
    log_roots: Iterable[Path],
) -> dict[str, object]:
    aggregate_dir = Path(aggregate_dir).resolve()
    paths = {
        key: aggregate_dir / filename for key, filename in {
            "manifest": "matrix_manifest.json",
            "per_eval": "frozen_per_eval.csv",
            "summary": "frozen_summary.csv",
            "paired": "frozen_paired_deltas.csv",
        }.items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V24 aggregate artifacts: {missing}")
    manifest = json.loads(paths["manifest"].read_text())
    per_eval = pd.read_csv(paths["per_eval"])
    paired = pd.read_csv(paths["paired"])
    training = _load_training_rows(log_roots)
    expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
    expected_rollouts = len(CONFIGS) * expected_pairs
    run_git = manifest.get("run_git_provenance", {}) or {}
    aggregate_git = manifest.get("git", {}) or {}
    source_commit = str(run_git.get("commit", ""))
    strict_checks = {
        "strict_complete": manifest.get("strict_complete") is True,
        "run_manifests_verified": manifest.get(
            "run_manifests_verified") is True,
        "common_random_numbers_verified": manifest.get(
            "common_random_numbers_verified") is True,
        "exploratory_stage": bool(
            manifest.get("stage") == "exploratory"
            and manifest.get("independent_confirmation") is False),
        "exact_configs": manifest.get("configs") == CONFIGS,
        "exact_train_seeds": manifest.get("train_seeds") == TRAIN_SEEDS,
        "exact_eval_seeds": manifest.get("eval_seeds") == EVAL_SEEDS,
        "forty_training_episodes": bool(
            manifest.get("train_episodes") == TRAIN_EPISODES
            and manifest.get("checkpoint_ep") == TRAIN_EPISODES - 1),
        "reference_is_v13_anchor": manifest.get("reference") == V13_ANCHOR,
        "source_is_clean_and_identified": bool(
            run_git.get("tracked_dirty") is False
            and len(source_commit) == 40
            and all(character in "0123456789abcdef"
                    for character in source_commit.lower())),
        "aggregate_git_matches_runs": bool(
            aggregate_git.get("commit") == source_commit
            and aggregate_git.get("tracked_dirty") is False),
        "expected_rollouts": bool(
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts),
        "unique_rollouts": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
    }
    if not all(strict_checks.values()):
        raise ValueError(f"V24 strict checks failed: {strict_checks}")

    required_eval_columns = {
        "config", "train_seed", "eval_seed", "headway_cv",
        "restricted_total_journey_horizon_min", "holding_vehicle_seconds",
        "fleet_denied_dispatch_events", "lower_action_mean",
        "lower_discrete_critic", "lower_policy_frozen",
        "lower_critic_frozen", "upper_policy_frozen",
        "lower_causal_guard_adjustment_mean_s",
        "lower_regularity_policy_evidence_valid_mean",
        "lower_regularity_policy_mode", "lower_regularity_projection_enabled",
        "lower_regularity_projection_mode",
        "lower_regularity_projection_applied",
        "lower_regularity_projection_regularity_target",
        "lower_regularity_projection_passenger_target",
        "lower_regularity_gain_floor_enabled",
        "lower_regularity_gain_floor_mode",
        "lower_regularity_gain_floor_base_fraction",
        "lower_regularity_gain_floor_hf_increment",
        "lower_regularity_gain_floor_required_gain_mean",
        "lower_regularity_gain_floor_expected_absolute_shortfall_mean",
        "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio",
        "lower_regularity_passenger_holding_enabled",
        "lower_regularity_passenger_holding_mode",
        "lower_regularity_passenger_cost_limit",
        "lower_regularity_passenger_expected_cost_mean",
        "lower_regularity_lambda", "lower_regularity_passenger_lambda",
        "lower_regularity_policy_penalty",
        "lower_regularity_policy_augmented_penalty",
        "lower_regularity_passenger_actor_penalty",
        "lower_regularity_passenger_actor_augmented_penalty",
    }
    missing = sorted(required_eval_columns - set(per_eval.columns))
    if missing:
        raise ValueError(f"V24 evaluation metrics are missing: {missing}")
    required_training_columns = {
        "ep", "config", "train_seed", "lower_regularity_policy_mode",
        "lower_regularity_projection_mode",
        "lower_regularity_projection_distillation",
        "lower_regularity_projection_distillation_steps",
        "lower_regularity_projection_applied",
        "lower_regularity_projection_converged",
        "lower_regularity_projection_iterations",
        "lower_regularity_projection_valid_count",
        "lower_regularity_projection_regularity_target",
        "lower_regularity_projection_passenger_target",
        "lower_regularity_projection_target_regularity_cost",
        "lower_regularity_projection_target_passenger_cost",
        "lower_regularity_projection_max_constraint_violation",
        "lower_regularity_projection_actor_distillation_steps",
        "lower_regularity_projection_actor_reverse_kl_episode_mean",
        "lower_regularity_projection_actor_forward_kl_episode_mean",
        "lower_regularity_projection_actor_post_reverse_kl_episode_mean",
        "lower_regularity_projection_actor_post_forward_kl_episode_mean",
        "lower_regularity_projection_actor_post_regularity_cost_episode_mean",
        "lower_regularity_projection_actor_post_passenger_cost_episode_mean",
        "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean",
        "lower_regularity_lambda", "lower_regularity_passenger_lambda",
        "lower_regularity_policy_penalty",
        "lower_regularity_policy_augmented_penalty",
        "lower_regularity_passenger_actor_penalty",
        "lower_regularity_passenger_actor_augmented_penalty",
        "lower_causal_guard_adjustment_mean_s",
    }
    missing = sorted(required_training_columns - set(training.columns))
    if missing:
        raise ValueError(f"V24 training metrics are missing: {missing}")

    v23_training = training.loc[
        training["config"] == V23_CANDIDATE].copy()
    v23_control_checks = _v23_training_control_checks(v23_training)
    v23_late = v23_training.loc[
        _numeric(v23_training, "ep").astype(int).isin(LATE_EPISODES)]
    candidate_results = {}
    passing = []
    for candidate, (distillation, steps) in CANDIDATE_SPECS.items():
        candidate_eval = _rows(per_eval, candidate)
        if len(candidate_eval) != expected_pairs:
            raise ValueError(f"incomplete V24 evaluation rows for {candidate}")
        paired_row = paired.loc[
            (paired["candidate"] == candidate)
            & (paired["reference"] == V13_ANCHOR)]
        if (len(paired_row) != 1
                or int(paired_row.iloc[0]["n_pairs"]) != expected_pairs):
            raise ValueError(f"incomplete paired artifact for {candidate}")
        candidate_training = training.loc[
            training["config"] == candidate].copy()
        evaluation_checks = _evaluation_checks(candidate_eval)
        training_checks, training_diagnostics = _training_checks(
            candidate_training, v23_late, distillation, steps)
        outcome_checks, outcome_diagnostics = _outcome_checks(
            per_eval, candidate, expected_pairs)
        passes = bool(
            all(v23_control_checks.values())
            and all(evaluation_checks.values())
            and all(training_checks.values())
            and all(outcome_checks.values()))
        if passes:
            passing.append(candidate)
        candidate_results[candidate] = {
            "passes": passes,
            "evaluation_checks": evaluation_checks,
            "training_checks": training_checks,
            "outcome_checks": outcome_checks,
            "training_diagnostics": training_diagnostics,
            **outcome_diagnostics,
        }
    selected = next(
        (candidate for candidate in CANDIDATES if candidate in passing), None)
    return {
        "gate_version": "freqduet-v24-distillation-screen-v1",
        "status": "exploratory_candidate_selected" if selected else "no_pass",
        "claim_eligible": False,
        "selected_for_confirmation": selected,
        "strict_checks": strict_checks,
        "v23_control_checks": v23_control_checks,
        "candidate_results": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate_dir", type=Path)
    parser.add_argument(
        "--logs-root", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = evaluate_v24_distillation_screen(
        args.aggregate_dir, args.logs_root)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    if args.require_pass and result["selected_for_confirmation"] is None:
        raise SystemExit("V24 screen found no confirmation candidate")


if __name__ == "__main__":
    main()
