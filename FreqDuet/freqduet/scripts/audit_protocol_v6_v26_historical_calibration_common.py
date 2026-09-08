"""Shared fail-closed checks for the V26 historical calibration gates."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_capacity_gain_screen import (
    CURRENT_MAIN,
    HARD_MAIN,
    REFERENCE,
    V13_ANCHOR,
)
from scripts.validate_freqduet_protocol_v6_configs import (
    V26_FOLLOWER_CALIBRATION_EXPECTED,
)


CANDIDATE_SPECS = {
    name: {
        "mode": mode,
        "history_alpha": alpha,
        "ridge": ridge,
        "adjustment_cap_s": cap,
    }
    for name, (mode, alpha, ridge, cap)
    in V26_FOLLOWER_CALIBRATION_EXPECTED.items()
}
CANDIDATES = list(CANDIDATE_SPECS)
PRIORITY = [
    "F_freqduet_protocol_v6_v26_histridge_a20_r005_c10_hiro",
    "F_freqduet_protocol_v6_v26_histridge_a10_r005_c10_hiro",
    "F_freqduet_protocol_v6_v26_histridge_a20_r005_c20_hiro",
    "F_freqduet_protocol_v6_v26_histbias_a20_c10_hiro",
]
CONTROLS = [HARD_MAIN, CURRENT_MAIN, REFERENCE, V13_ANCHOR]
MIN_HISTORY_EPISODES = 5
MIN_SAMPLES_PER_EPISODE = 128
UPDATE_SOURCE = "completed_learned_training_days_v1"


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def finite(values: pd.Series) -> bool:
    array = values.to_numpy(dtype=np.float64)
    return bool(array.size and np.isfinite(array).all())


def all_close(
    left: pd.Series, right: pd.Series | float, *, atol: float = 1e-8
) -> bool:
    return bool(np.allclose(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        rtol=0.0,
        atol=atol,
    ))


def load_aggregate(
    aggregate_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    root = Path(aggregate_dir).resolve()
    required = (
        "matrix_manifest.json",
        "frozen_per_eval.csv",
        "frozen_summary.csv",
        "frozen_paired_deltas.csv",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing V26 aggregate artifacts: {missing}")
    manifest = json.loads((root / "matrix_manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("V26 matrix manifest must be a JSON object")
    return manifest, pd.read_csv(root / "frozen_per_eval.csv")


def strict_matrix_checks(
    manifest: dict[str, object],
    per_eval: pd.DataFrame,
    *,
    configs: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    train_episodes: int,
    reference: str,
) -> dict[str, bool]:
    expected_rollouts = len(configs) * len(train_seeds) * len(eval_seeds)
    run_git = manifest.get("run_git_provenance", {}) or {}
    aggregate_git = manifest.get("git", {}) or {}
    source_commit = str(run_git.get("commit", ""))
    required_columns = {
        "config", "train_seed", "eval_seed", "checkpoint_ep",
        "lower_policy_frozen", "lower_critic_frozen",
        "upper_policy_frozen",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(
            f"V26 frozen evaluation columns are missing: {missing_columns}")
    return {
        "strict_complete": manifest.get("strict_complete") is True,
        "run_manifests_verified": manifest.get(
            "run_manifests_verified") is True,
        "common_random_numbers_verified": manifest.get(
            "common_random_numbers_verified") is True,
        "exploratory_nonconfirmation": bool(
            manifest.get("stage") == "exploratory"
            and manifest.get("independent_confirmation") is False),
        "exact_configs": manifest.get("configs") == configs,
        "exact_train_seeds": manifest.get("train_seeds") == train_seeds,
        "exact_eval_seeds": manifest.get("eval_seeds") == eval_seeds,
        "exact_training_horizon": bool(
            manifest.get("train_episodes") == train_episodes
            and manifest.get("checkpoint_ep") == train_episodes - 1),
        "registered_reference": manifest.get("reference") == reference,
        "source_is_clean_and_identified": bool(
            run_git.get("tracked_dirty") is False
            and re.fullmatch(r"[0-9a-f]{40}", source_commit)),
        "aggregate_git_matches_runs": bool(
            aggregate_git.get("commit") == source_commit
            and aggregate_git.get("tracked_dirty") is False),
        "expected_rollouts": bool(
            manifest.get("expected_rollouts") == expected_rollouts
            and len(per_eval) == expected_rollouts),
        "unique_rollouts": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
        "exact_evaluation_rows": bool(
            set(per_eval["config"].astype(str)) == set(configs)
            and set(numeric(per_eval, "train_seed").astype(int))
            == set(train_seeds)
            and set(numeric(per_eval, "eval_seed").astype(int))
            == set(eval_seeds)
            and (numeric(per_eval, "checkpoint_ep")
                 == train_episodes - 1).all()),
        "frozen_policies": bool(
            (numeric(per_eval, "lower_policy_frozen") == 1.0).all()
            and (numeric(per_eval, "lower_critic_frozen") == 1.0).all()
            and (numeric(per_eval, "upper_policy_frozen") == 1.0).all()),
    }


def load_training_rows(
    log_roots: Iterable[Path],
    *,
    train_seeds: list[int],
    train_episodes: int,
) -> pd.DataFrame:
    roots = [Path(root).resolve() for root in log_roots]
    if not roots:
        raise ValueError("V26 audit requires at least one training log root")
    frames = []
    for candidate in CANDIDATES:
        for train_seed in train_seeds:
            relative = Path(
                f"{candidate}_seed{train_seed}") / "diagnostics.csv"
            matches = [root / relative for root in roots
                       if (root / relative).is_file()]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one V26 diagnostics file for {candidate} "
                    f"seed {train_seed}, found {matches}")
            frame = pd.read_csv(matches[0])
            episodes = numeric(frame, "ep")
            if (len(frame) != train_episodes or not finite(episodes)
                    or set(episodes.astype(int))
                    != set(range(train_episodes))):
                raise ValueError(
                    f"{matches[0]}: expected exactly episodes "
                    f"0--{train_episodes - 1}")
            frame = frame.copy()
            frame["config"] = candidate
            frame["train_seed"] = int(train_seed)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


TRAINING_COLUMNS = {
    "ep", "follower_target_calibration_enabled",
    "follower_target_calibration_mode",
    "follower_target_calibration_post_update_active",
    "follower_target_calibration_post_update_history_episodes",
    "follower_target_calibration_post_update_history_samples",
    "follower_target_calibration_post_update_coefficient_norm",
    "follower_target_calibration_episode_samples",
    "follower_target_calibration_episode_updated",
    "follower_target_calibration_update_source",
    "follower_forecast_registered_count",
    "follower_forecast_resolved_count",
    "follower_forecast_resolution_rate",
    "follower_forecast_calibration_active_mean",
    "follower_forecast_calibration_history_episodes_mean",
    "follower_forecast_calibration_requested_adjustment_abs_mean_s",
    "follower_forecast_calibration_requested_adjustment_abs_max_s",
    "follower_forecast_target_action_prediction_mae_s",
    "follower_forecast_base_target_action_prediction_mae_s",
}


EVALUATION_COLUMNS = TRAINING_COLUMNS.difference({"ep"}).union({
    "config", "train_seed", "eval_seed", "lower_policy_frozen",
    "lower_critic_frozen", "upper_policy_frozen",
    "follower_target_calibration_post_update_intercept_s",
    "follower_forecast_hold_need_false_positive_mean",
    "follower_forecast_hold_need_false_negative_mean",
    "follower_forecast_base_hold_need_false_positive_mean",
    "follower_forecast_base_hold_need_false_negative_mean",
    "follower_forecast_predicted_follower_gap_s_mean",
    "follower_forecast_base_predicted_follower_gap_s_mean",
    "follower_forecast_raw_gap_prediction_error_s_mean",
    "follower_forecast_base_gap_prediction_error_s_mean",
    "follower_forecast_calibration_target_adjustment_abs_mean_s",
})


def candidate_training_checks(
    rows: pd.DataFrame,
    *,
    candidate: str,
    train_seeds: list[int],
    train_episodes: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    missing = sorted(TRAINING_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"V26 training metrics are missing: {missing}")
    spec = CANDIDATE_SPECS[candidate]
    ep = numeric(rows, "ep").astype(int)
    pre_activation = ep < MIN_HISTORY_EPISODES
    active = ~pre_activation
    expected_post_active = ep >= MIN_HISTORY_EPISODES - 1
    adjustment_mean = numeric(
        rows,
        "follower_forecast_calibration_requested_adjustment_abs_mean_s")
    adjustment_max = numeric(
        rows,
        "follower_forecast_calibration_requested_adjustment_abs_max_s")
    used_active = numeric(rows, "follower_forecast_calibration_active_mean")
    used_history = numeric(
        rows, "follower_forecast_calibration_history_episodes_mean")
    resolved = numeric(rows, "follower_forecast_resolved_count")
    coefficient_norm = numeric(
        rows, "follower_target_calibration_post_update_coefficient_norm")
    active_adjustment_by_seed = rows.loc[active].assign(
        _adjustment=adjustment_mean.loc[active].to_numpy()).groupby(
            "train_seed")["_adjustment"].max()
    checks = {
        "complete_training_diagnostics": bool(
            len(rows) == len(train_seeds) * train_episodes
            and rows.groupby("train_seed")["ep"].nunique().eq(
                train_episodes).all()),
        "exact_calibration_contract": bool(
            (numeric(rows, "follower_target_calibration_enabled")
             == 1.0).all()
            and (rows["follower_target_calibration_mode"].astype(str)
                 == spec["mode"]).all()
            and (rows["follower_target_calibration_update_source"].astype(str)
                 == UPDATE_SOURCE).all()),
        "one_completed_day_update_per_episode": bool(
            (numeric(rows, "follower_target_calibration_episode_updated")
             == 1.0).all()
            and (numeric(
                rows,
                "follower_target_calibration_post_update_history_episodes")
                 == ep + 1).all()
            and (numeric(
                rows, "follower_target_calibration_episode_samples")
                 >= MIN_SAMPLES_PER_EPISODE).all()),
        "post_update_activation_boundary": bool(
            (numeric(
                rows, "follower_target_calibration_post_update_active")
             == expected_post_active.astype(float)).all()),
        "used_generation_has_no_same_day_activation": bool(
            all_close(used_active.loc[pre_activation], 0.0)
            and all_close(
                used_history.loc[pre_activation],
                ep.loc[pre_activation].astype(float))),
        "used_generation_active_after_five_days": bool(
            active.any()
            and (used_active.loc[active] >= 1.0 - 1e-8).all()
            and all_close(
                used_history.loc[active], ep.loc[active].astype(float))),
        "resolved_forecast_coverage": bool(
            finite(resolved) and (resolved > 0.0).all()
            and all_close(
                numeric(rows, "follower_forecast_resolution_rate"), 1.0)),
        "requested_adjustment_within_registered_cap": bool(
            finite(adjustment_max)
            and (adjustment_max >= -1e-12).all()
            and (adjustment_max
                 <= float(spec["adjustment_cap_s"]) + 1e-8).all()),
        "pre_activation_is_behavior_neutral": bool(
            all_close(adjustment_mean.loc[pre_activation], 0.0)
            and all_close(adjustment_max.loc[pre_activation], 0.0)),
        "active_adjustment_observed_for_every_seed": bool(
            len(active_adjustment_by_seed) == len(train_seeds)
            and (active_adjustment_by_seed > 1e-6).all()),
        "fitted_coefficients_are_finite_and_nonzero": bool(
            finite(coefficient_norm)
            and (coefficient_norm >= 0.0).all()
            and (coefficient_norm.loc[expected_post_active] > 1e-9).all()),
        "paired_forecast_metrics_are_finite": bool(
            finite(numeric(
                rows, "follower_forecast_target_action_prediction_mae_s"))
            and finite(numeric(
                rows,
                "follower_forecast_base_target_action_prediction_mae_s"))),
    }
    diagnostics = {
        "active_adjustment_abs_mean_s": float(
            adjustment_mean.loc[active].mean()),
        "active_adjustment_abs_max_s": float(
            adjustment_max.loc[active].max()),
        "post_update_coefficient_norm_max": float(coefficient_norm.max()),
        "resolved_forecast_count": float(resolved.sum()),
    }
    return checks, diagnostics


def candidate_evaluation_checks(
    rows: pd.DataFrame,
    *,
    candidate: str,
    train_seeds: list[int],
    eval_seeds: list[int],
    train_episodes: int,
) -> tuple[dict[str, bool], dict[str, float]]:
    missing = sorted(EVALUATION_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"V26 evaluation metrics are missing: {missing}")
    spec = CANDIDATE_SPECS[candidate]
    adjustment_mean = numeric(
        rows,
        "follower_forecast_calibration_requested_adjustment_abs_mean_s")
    adjustment_max = numeric(
        rows,
        "follower_forecast_calibration_requested_adjustment_abs_max_s")
    resolved = numeric(rows, "follower_forecast_resolved_count")
    adjustment_by_seed = rows.assign(
        _adjustment=adjustment_mean.to_numpy()).groupby(
            "train_seed")["_adjustment"].max()
    checks = {
        "exact_candidate_rows": bool(
            len(rows) == len(train_seeds) * len(eval_seeds)
            and set(numeric(rows, "train_seed").astype(int))
            == set(train_seeds)
            and set(numeric(rows, "eval_seed").astype(int))
            == set(eval_seeds)),
        "frozen_policies": bool(
            (numeric(rows, "lower_policy_frozen") == 1.0).all()
            and (numeric(rows, "lower_critic_frozen") == 1.0).all()
            and (numeric(rows, "upper_policy_frozen") == 1.0).all()),
        "frozen_calibrator_contract": bool(
            (numeric(rows, "follower_target_calibration_enabled")
             == 1.0).all()
            and (rows["follower_target_calibration_mode"].astype(str)
                 == spec["mode"]).all()
            and (numeric(
                rows, "follower_target_calibration_post_update_active")
                 == 1.0).all()
            and (numeric(
                rows,
                "follower_target_calibration_post_update_history_episodes")
                 == float(train_episodes)).all()
            and (numeric(
                rows, "follower_target_calibration_episode_updated")
                 == 0.0).all()),
        "calibrator_used_for_every_registered_forecast": bool(
            (numeric(rows, "follower_forecast_calibration_active_mean")
             >= 1.0 - 1e-8).all()
            and all_close(
                numeric(
                    rows,
                    "follower_forecast_calibration_history_episodes_mean"),
                float(train_episodes))),
        "resolved_forecast_coverage": bool(
            finite(resolved) and (resolved > 0.0).all()
            and all_close(
                numeric(rows, "follower_forecast_resolution_rate"), 1.0)),
        "requested_adjustment_within_registered_cap": bool(
            finite(adjustment_max)
            and (adjustment_max >= -1e-12).all()
            and (adjustment_max
                 <= float(spec["adjustment_cap_s"]) + 1e-8).all()),
        "active_adjustment_observed_for_every_seed": bool(
            len(adjustment_by_seed) == len(train_seeds)
            and (adjustment_by_seed > 1e-6).all()),
        "paired_forecast_metrics_are_finite": bool(
            finite(numeric(
                rows, "follower_forecast_target_action_prediction_mae_s"))
            and finite(numeric(
                rows,
                "follower_forecast_base_target_action_prediction_mae_s"))
            and finite(numeric(
                rows, "follower_forecast_hold_need_false_positive_mean"))
            and finite(numeric(
                rows, "follower_forecast_hold_need_false_negative_mean"))
            and finite(numeric(
                rows,
                "follower_forecast_base_hold_need_false_positive_mean"))
            and finite(numeric(
                rows,
                "follower_forecast_base_hold_need_false_negative_mean"))),
    }
    diagnostics = {
        "adjustment_abs_mean_s": float(adjustment_mean.mean()),
        "adjustment_abs_max_s": float(adjustment_max.max()),
        "resolved_forecast_count": float(resolved.sum()),
    }
    return checks, diagnostics


def control_evaluation_checks(
    rows: pd.DataFrame,
) -> dict[str, bool]:
    missing = sorted(EVALUATION_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"V26 control metrics are missing: {missing}")
    zero_columns = (
        "follower_forecast_calibration_active_mean",
        "follower_forecast_calibration_requested_adjustment_abs_mean_s",
        "follower_forecast_calibration_requested_adjustment_abs_max_s",
        "follower_forecast_calibration_target_adjustment_abs_mean_s",
        "follower_target_calibration_episode_updated",
    )
    identities = (
        ("follower_forecast_predicted_follower_gap_s_mean",
         "follower_forecast_base_predicted_follower_gap_s_mean"),
        ("follower_forecast_raw_gap_prediction_error_s_mean",
         "follower_forecast_base_gap_prediction_error_s_mean"),
        ("follower_forecast_target_action_prediction_mae_s",
         "follower_forecast_base_target_action_prediction_mae_s"),
        ("follower_forecast_hold_need_false_positive_mean",
         "follower_forecast_base_hold_need_false_positive_mean"),
        ("follower_forecast_hold_need_false_negative_mean",
         "follower_forecast_base_hold_need_false_negative_mean"),
    )
    return {
        "calibration_disabled": bool(
            (numeric(rows, "follower_target_calibration_enabled")
             == 0.0).all()
            and (rows["follower_target_calibration_mode"].astype(str)
                 == "disabled").all()),
        "zero_calibration_adjustment": all(
            all_close(numeric(rows, column), 0.0)
            for column in zero_columns),
        "base_effective_forecast_identity": all(
            all_close(numeric(rows, left), numeric(rows, right))
            for left, right in identities),
    }


def weighted_mean(rows: pd.DataFrame, column: str) -> float:
    values = numeric(rows, column).to_numpy(dtype=np.float64)
    weights = numeric(
        rows, "follower_forecast_resolved_count").to_numpy(dtype=np.float64)
    if (not values.size or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or (weights <= 0.0).any()):
        raise ValueError(f"cannot pool V26 follower metric {column}")
    return float(np.average(values, weights=weights))
