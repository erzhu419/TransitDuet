"""Shared fail-closed checks for the V27 causal multi-step value gates."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.audit_protocol_v6_aggregate_gain_screen import (
    V13_ZERO_HOLD_ADVANTAGE,
)
from scripts.audit_protocol_v6_capacity_gain_screen import (
    CURRENT_MAIN,
    REFERENCE,
    V13_ANCHOR,
)
from scripts.run_freqduet_protocol_v2_matrix import resolved_config
from scripts.validate_freqduet_protocol_v6_configs import (
    V27_MULTISTEP_VALUE_EXPECTED,
)


CANDIDATE_SPECS = {
    name: {"horizon_steps": horizon, "ucb_beta": ucb_beta}
    for name, (horizon, ucb_beta) in V27_MULTISTEP_VALUE_EXPECTED.items()
}
CANDIDATES = list(CANDIDATE_SPECS)
PRIORITY = [
    "F_freqduet_protocol_v6_v27_msvalue_h4_u050_r0010_hiro",
    "F_freqduet_protocol_v6_v27_msvalue_h4_u000_r0010_hiro",
    "F_freqduet_protocol_v6_v27_msvalue_h2_u000_r0010_hiro",
    "F_freqduet_protocol_v6_v27_msvalue_h6_u050_r0010_hiro",
]
CONTROLS = [
    CURRENT_MAIN,
    REFERENCE,
    V13_ANCHOR,
    V13_ZERO_HOLD_ADVANTAGE,
]
VALUE_MODE = "discounted_future_arrival_cost_change_v1"
POLICY_MODE = "causal_multistep_arrival_delta_regret_dual_v12"
CONSTRAINT_COST_MODE = "downstream_arrival_value_regret_v5"
CONSTRAINT_SCALE_MODE = "cost_limit_ratio_v1"
VALUE_COST_LIMIT = 0.001
MIN_REPLAY_SIZE = 512
MIN_CRITIC_UPDATES = 30
REGISTERED_ACTIONS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]


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
        raise FileNotFoundError(f"missing V27 aggregate artifacts: {missing}")
    manifest = json.loads((root / "matrix_manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("V27 matrix manifest must be a JSON object")
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
    expected_grid = set(product(configs, train_seeds, eval_seeds))
    run_git = manifest.get("run_git_provenance", {}) or {}
    aggregate_git = manifest.get("git", {}) or {}
    source_commit = str(run_git.get("commit", ""))
    required_columns = {
        "config", "train_seed", "eval_seed", "checkpoint_ep",
        "scenario_tape_id", "lower_policy_frozen", "lower_critic_frozen",
        "upper_policy_frozen",
    }
    missing_columns = sorted(required_columns - set(per_eval.columns))
    if missing_columns:
        raise ValueError(
            f"V27 frozen evaluation columns are missing: {missing_columns}")
    actual_grid = set(zip(
        per_eval["config"].astype(str),
        numeric(per_eval, "train_seed").astype(int),
        numeric(per_eval, "eval_seed").astype(int),
    ))
    scenario_tapes = per_eval.groupby("eval_seed")[
        "scenario_tape_id"].nunique(dropna=False)
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
        "exact_cartesian_rollouts": bool(
            actual_grid == expected_grid
            and len(per_eval) == len(expected_grid)
            and manifest.get("expected_rollouts") == len(expected_grid)),
        "unique_rollouts": not per_eval.duplicated(
            ["config", "train_seed", "eval_seed"]).any(),
        "exact_checkpoint_rows": bool(
            (numeric(per_eval, "checkpoint_ep")
             == train_episodes - 1).all()),
        "one_scenario_tape_per_eval_seed": bool(
            per_eval["scenario_tape_id"].notna().all()
            and scenario_tapes.eq(1).all()),
        "frozen_policies": bool(
            (numeric(per_eval, "lower_policy_frozen") == 1.0).all()
            and (numeric(per_eval, "lower_critic_frozen") == 1.0).all()
            and (numeric(per_eval, "upper_policy_frozen") == 1.0).all()),
    }


def registered_config_checks(candidate: str) -> dict[str, bool]:
    config = resolved_config(candidate)
    lower = config["lower"]
    frequency = config["frequency"]
    policy = lower["causal_regularity_policy"]
    value = policy["multi_step_value"]
    spec = CANDIDATE_SPECS[candidate]
    return {
        "exact_registered_value_architecture": bool(
            value.get("enable") is True
            and value.get("mode") == VALUE_MODE
            and value.get("horizon_steps") == spec["horizon_steps"]
            and value.get("discount") == 1.0
            and value.get("ucb_beta") == spec["ucb_beta"]
            and value.get("min_replay_size") == MIN_REPLAY_SIZE
            and value.get("min_critic_updates") == MIN_CRITIC_UPDATES
            and value.get("replay_capacity") == 500000
            and value.get("hidden_dim") == 64
            and value.get("ensemble_size") == 5
            and value.get("n_layers") == 2),
        "exact_registered_actor_objective": bool(
            policy.get("enable") is True
            and policy.get("mode") == POLICY_MODE
            and policy.get("evidence_mode") == "compact_causal_target_v7"
            and policy.get("constraint_scale_mode") == CONSTRAINT_SCALE_MODE
            and policy.get("cost_limit") == VALUE_COST_LIMIT),
        "historical_harmonic_frequency_prior_retained": bool(
            frequency.get("enable") is True
            and frequency.get("method") == "harmonic"
            and frequency.get("forecast_mode") == "causal"
            and frequency.get("use_historical_prior") is True
            and frequency.get("upper_mode") == "low"
            and frequency.get("lower_mode") == "high"
            and frequency.get("observation_source") == "apc_boardings"),
        "deployable_causal_state_retained": bool(
            lower.get("observation_contract") == "deployable_apc_avl_v4"
            and lower.get("headway_reward_mode") == "forward_event_only"
            and lower.get("unobserved_action_mode") == "zero"
            and lower.get("state_encoder", {}).get("input_schema")
            == "causal_forward_v4"),
        "seven_executable_actions_and_v13_reward_critic_retained": bool(
            lower.get("action_bins") == REGISTERED_ACTIONS
            and lower.get("discrete_critic", "continuous_action")
            == "continuous_action"),
        "sampled_no_guard_no_calibration_contract": bool(
            lower.get("causal_holding_guard", {}).get("enable") is False
            and not bool((lower.get(
                "follower_forecast_calibration", {}) or {}).get(
                    "enable", False))
            and policy.get("conditional_entropy", {}).get("enable") is True),
    }


BASE_RUNTIME_COLUMNS = {
    "lower_observation_contract", "headway_reward_mode",
    "frequency_observation_source", "lower_discrete_critic",
    "lower_causal_guard_enabled", "lower_causal_guard_adjustment_mean_s",
    "follower_target_calibration_enabled",
    "follower_target_calibration_mode",
    "lower_regularity_policy_enabled", "lower_regularity_policy_mode",
    "lower_regularity_policy_constraint_cost_mode",
    "lower_regularity_policy_constraint_scale_mode",
    "lower_regularity_policy_cost_limit",
    "lower_regularity_policy_valid_fraction",
    "lower_multistep_value_enabled", "lower_multistep_value_mode",
    "lower_multistep_value_horizon_steps",
    "lower_multistep_value_discount", "lower_multistep_value_ucb_beta",
    "lower_multistep_value_ready", "lower_multistep_value_replay_size",
    "lower_multistep_value_targets_emitted",
    "lower_multistep_value_terminal_tails_discarded",
    "lower_multistep_value_episode_tails_discarded",
    "lower_multistep_value_critic_updates",
    "lower_multistep_value_critic_loss", "lower_multistep_value_target_mean",
    "lower_multistep_value_target_std",
    "lower_multistep_value_prediction_mean",
    "lower_multistep_value_prediction_std",
    "lower_multistep_value_grad_norm",
    "lower_multistep_value_action_span_mean",
    "lower_multistep_value_advantage_mean",
    "lower_multistep_value_advantage_std_mean",
    "lower_multistep_value_positive_regret_mean",
    "lower_multistep_value_positive_regret_max",
    "lower_multistep_value_frozen",
}
TRAINING_COLUMNS = BASE_RUNTIME_COLUMNS.union({"ep"})
EVALUATION_COLUMNS = BASE_RUNTIME_COLUMNS.union({
    "config", "train_seed", "eval_seed", "lower_policy_frozen",
    "lower_critic_frozen", "upper_policy_frozen",
})


def load_training_rows(
    log_roots: Iterable[Path],
    *,
    train_seeds: list[int],
    train_episodes: int,
) -> pd.DataFrame:
    roots = [Path(root).resolve() for root in log_roots]
    if not roots:
        raise ValueError("V27 audit requires at least one training log root")
    frames = []
    for candidate in CANDIDATES:
        for train_seed in train_seeds:
            relative = Path(
                f"{candidate}_seed{train_seed}") / "diagnostics.csv"
            matches = [root / relative for root in roots
                       if (root / relative).is_file()]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one V27 diagnostics file for {candidate} "
                    f"seed {train_seed}, found {matches}")
            frame = pd.read_csv(matches[0])
            if "ep" not in frame:
                raise ValueError(f"{matches[0]}: missing episode column")
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


def _runtime_contract_checks(
    rows: pd.DataFrame, candidate: str
) -> dict[str, bool]:
    spec = CANDIDATE_SPECS[candidate]
    return {
        "exact_value_runtime_contract": bool(
            (numeric(rows, "lower_multistep_value_enabled") == 1.0).all()
            and (rows["lower_multistep_value_mode"].astype(str)
                 == VALUE_MODE).all()
            and all_close(
                numeric(rows, "lower_multistep_value_horizon_steps"),
                float(spec["horizon_steps"]))
            and all_close(
                numeric(rows, "lower_multistep_value_discount"), 1.0)
            and all_close(
                numeric(rows, "lower_multistep_value_ucb_beta"),
                float(spec["ucb_beta"]))),
        "exact_actor_constraint_runtime_contract": bool(
            (numeric(rows, "lower_regularity_policy_enabled") == 1.0).all()
            and (rows["lower_regularity_policy_mode"].astype(str)
                 == POLICY_MODE).all()
            and (rows[
                "lower_regularity_policy_constraint_cost_mode"].astype(str)
                 == CONSTRAINT_COST_MODE).all()
            and (rows[
                "lower_regularity_policy_constraint_scale_mode"].astype(str)
                 == CONSTRAINT_SCALE_MODE).all()
            and all_close(
                numeric(rows, "lower_regularity_policy_cost_limit"),
                VALUE_COST_LIMIT)),
        "deployable_causal_runtime_contract": bool(
            (rows["lower_observation_contract"].astype(str)
             == "deployable_apc_avl_v4").all()
            and (rows["headway_reward_mode"].astype(str)
                 == "forward_event_only").all()
            and (rows["frequency_observation_source"].astype(str)
                 == "apc_boardings").all()),
        "continuous_v13_reward_critic": bool(
            (rows["lower_discrete_critic"].astype(str)
             == "continuous_action").all()),
        "rejected_calibration_and_guard_absent": bool(
            (numeric(rows, "follower_target_calibration_enabled")
             == 0.0).all()
            and (rows["follower_target_calibration_mode"].astype(str)
                 == "disabled").all()
            and (numeric(rows, "lower_causal_guard_enabled") == 0.0).all()
            and all_close(
                numeric(rows, "lower_causal_guard_adjustment_mean_s"), 0.0)),
    }


def _monotonic_by_seed(rows: pd.DataFrame, column: str) -> bool:
    for _, seed_rows in rows.groupby("train_seed"):
        values = numeric(seed_rows.sort_values("ep"), column)
        if not finite(values) or (values.diff().dropna() < 0.0).any():
            return False
    return True


def candidate_training_checks(
    rows: pd.DataFrame,
    *,
    candidate: str,
    train_seeds: list[int],
    train_episodes: int,
) -> tuple[dict[str, bool], dict[str, object]]:
    missing = sorted(TRAINING_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"V27 training metrics are missing: {missing}")
    replay = numeric(rows, "lower_multistep_value_replay_size")
    targets = numeric(rows, "lower_multistep_value_targets_emitted")
    updates = numeric(rows, "lower_multistep_value_critic_updates")
    ready = numeric(rows, "lower_multistep_value_ready")
    expected_ready = (
        (replay >= MIN_REPLAY_SIZE) & (updates >= MIN_CRITIC_UPDATES))
    final = rows.sort_values("ep").groupby("train_seed").tail(1)
    active = rows.loc[ready == 1.0]
    objective_metrics = (
        "lower_multistep_value_critic_loss",
        "lower_multistep_value_target_mean",
        "lower_multistep_value_target_std",
        "lower_multistep_value_prediction_mean",
        "lower_multistep_value_prediction_std",
        "lower_multistep_value_grad_norm",
        "lower_multistep_value_action_span_mean",
        "lower_multistep_value_advantage_mean",
        "lower_multistep_value_advantage_std_mean",
        "lower_multistep_value_positive_regret_mean",
        "lower_multistep_value_positive_regret_max",
    )
    positive_by_seed = {}
    for column in (
            "lower_multistep_value_critic_loss",
            "lower_multistep_value_target_std",
            "lower_multistep_value_grad_norm",
            "lower_multistep_value_action_span_mean"):
        positive_by_seed[column] = bool(
            len(active) > 0
            and active.assign(_value=numeric(active, column)).groupby(
                "train_seed")["_value"].max().gt(0.0).reindex(
                    train_seeds, fill_value=False).all())
    checks = {
        "complete_training_diagnostics": bool(
            len(rows) == len(train_seeds) * train_episodes
            and rows.groupby("train_seed")["ep"].nunique().eq(
                train_episodes).all()),
        **registered_config_checks(candidate),
        **_runtime_contract_checks(rows, candidate),
        "cumulative_training_evidence_is_monotonic": all(
            _monotonic_by_seed(rows, column)
            for column in (
                "lower_multistep_value_replay_size",
                "lower_multistep_value_targets_emitted",
                "lower_multistep_value_terminal_tails_discarded",
                "lower_multistep_value_episode_tails_discarded",
                "lower_multistep_value_critic_updates",
            )),
        "readiness_matches_registered_warmup": bool(
            finite(replay) and finite(updates) and finite(ready)
            and (ready == expected_ready.astype(float)).all()),
        "final_training_evidence_is_ready": bool(
            (numeric(final, "lower_multistep_value_ready") == 1.0).all()
            and (numeric(final, "lower_multistep_value_replay_size")
                 >= MIN_REPLAY_SIZE).all()
            and (numeric(final, "lower_multistep_value_targets_emitted")
                 >= MIN_REPLAY_SIZE).all()
            and (numeric(final, "lower_multistep_value_critic_updates")
                 >= MIN_CRITIC_UPDATES).all()),
        "complete_targets_do_not_exceed_emitted_labels": bool(
            finite(replay) and finite(targets) and (targets >= replay).all()),
        "trip_or_day_tail_discard_observed_per_seed": bool(
            (numeric(final, "lower_multistep_value_terminal_tails_discarded")
             + numeric(final, "lower_multistep_value_episode_tails_discarded")
             > 0.0).all()),
        "active_objective_metrics_are_finite": bool(
            len(active) > 0 and all(
                finite(numeric(active, column)) for column in objective_metrics)),
        "nonzero_critic_loss_per_seed": positive_by_seed[
            "lower_multistep_value_critic_loss"],
        "nonzero_target_variance_per_seed": positive_by_seed[
            "lower_multistep_value_target_std"],
        "nonzero_critic_gradient_per_seed": positive_by_seed[
            "lower_multistep_value_grad_norm"],
        "nonzero_categorical_value_span_per_seed": positive_by_seed[
            "lower_multistep_value_action_span_mean"],
        "causal_actor_states_observed": bool(
            len(active) > 0
            and finite(numeric(
                active, "lower_regularity_policy_valid_fraction"))
            and (numeric(active, "lower_regularity_policy_valid_fraction")
                 > 0.0).all()),
        "training_objective_not_frozen": bool(
            (numeric(rows, "lower_multistep_value_frozen") == 0.0).all()),
    }
    diagnostics = {
        "final_by_train_seed": {
            str(int(row.train_seed)): {
                "replay_size": int(row.lower_multistep_value_replay_size),
                "targets_emitted": int(
                    row.lower_multistep_value_targets_emitted),
                "critic_updates": int(row.lower_multistep_value_critic_updates),
                "tails_discarded": int(
                    row.lower_multistep_value_terminal_tails_discarded
                    + row.lower_multistep_value_episode_tails_discarded),
            }
            for row in final.itertuples()
        },
        "ready_training_rows": int(len(active)),
    }
    return checks, diagnostics


def candidate_evaluation_checks(
    rows: pd.DataFrame,
    *,
    candidate: str,
    train_seeds: list[int],
    eval_seeds: list[int],
) -> tuple[dict[str, bool], dict[str, float]]:
    missing = sorted(EVALUATION_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"V27 evaluation metrics are missing: {missing}")
    replay = numeric(rows, "lower_multistep_value_replay_size")
    targets = numeric(rows, "lower_multistep_value_targets_emitted")
    updates = numeric(rows, "lower_multistep_value_critic_updates")
    checks = {
        "exact_candidate_rows": bool(
            len(rows) == len(train_seeds) * len(eval_seeds)
            and set(numeric(rows, "train_seed").astype(int))
            == set(train_seeds)
            and set(numeric(rows, "eval_seed").astype(int))
            == set(eval_seeds)),
        **registered_config_checks(candidate),
        **_runtime_contract_checks(rows, candidate),
        "frozen_value_objective_is_ready": bool(
            (numeric(rows, "lower_multistep_value_ready") == 1.0).all()
            and finite(replay) and (replay >= MIN_REPLAY_SIZE).all()
            and finite(targets) and (targets >= MIN_REPLAY_SIZE).all()
            and finite(updates) and (updates >= MIN_CRITIC_UPDATES).all()),
        "frozen_training_evidence_is_consistent": bool(
            (targets >= replay).all()
            and (numeric(
                rows, "lower_multistep_value_terminal_tails_discarded")
                + numeric(
                    rows, "lower_multistep_value_episode_tails_discarded")
                > 0.0).all()),
        "frozen_policies_and_value_critic": bool(
            (numeric(rows, "lower_policy_frozen") == 1.0).all()
            and (numeric(rows, "lower_critic_frozen") == 1.0).all()
            and (numeric(rows, "upper_policy_frozen") == 1.0).all()
            and (numeric(rows, "lower_multistep_value_frozen") == 1.0).all()),
    }
    diagnostics = {
        "replay_size_min": float(replay.min()),
        "targets_emitted_min": float(targets.min()),
        "critic_updates_min": float(updates.min()),
    }
    return checks, diagnostics


def control_evaluation_checks(rows: pd.DataFrame) -> dict[str, bool]:
    missing = sorted(EVALUATION_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"V27 control metrics are missing: {missing}")
    zero_columns = (
        "lower_multistep_value_horizon_steps",
        "lower_multistep_value_discount",
        "lower_multistep_value_ucb_beta",
        "lower_multistep_value_ready",
        "lower_multistep_value_replay_size",
        "lower_multistep_value_targets_emitted",
        "lower_multistep_value_terminal_tails_discarded",
        "lower_multistep_value_episode_tails_discarded",
        "lower_multistep_value_critic_updates",
        "lower_multistep_value_critic_loss",
        "lower_multistep_value_target_mean",
        "lower_multistep_value_target_std",
        "lower_multistep_value_prediction_mean",
        "lower_multistep_value_prediction_std",
        "lower_multistep_value_grad_norm",
        "lower_multistep_value_action_span_mean",
        "lower_multistep_value_advantage_mean",
        "lower_multistep_value_advantage_std_mean",
        "lower_multistep_value_positive_regret_mean",
        "lower_multistep_value_positive_regret_max",
        "lower_causal_guard_adjustment_mean_s",
    )
    return {
        "multistep_value_disabled": bool(
            (numeric(rows, "lower_multistep_value_enabled") == 0.0).all()
            and (rows["lower_multistep_value_mode"].astype(str)
                 == "disabled").all()),
        "no_multistep_state_or_execution_adjustment": all(
            all_close(numeric(rows, column), 0.0) for column in zero_columns),
        "rejected_calibration_disabled": bool(
            (numeric(rows, "follower_target_calibration_enabled")
             == 0.0).all()
            and (rows["follower_target_calibration_mode"].astype(str)
                 == "disabled").all()),
        "frozen_control_policies": bool(
            (numeric(rows, "lower_policy_frozen") == 1.0).all()
            and (numeric(rows, "lower_critic_frozen") == 1.0).all()
            and (numeric(rows, "upper_policy_frozen") == 1.0).all()),
    }
