"""Freeze paired option/episode-credit first updates on the same archives."""

from scripts import pointmaze_backtracking_stage60_spec as source
from scripts import pointmaze_credit_diagnostics_stage62_spec as diagnosis

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_episode_credit_stage63_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_episode_credit_stage63.py"
POLICY, METHODS = "episode_credit", source.METHODS
PERIODS, TRAIN_POLICIES, roots, options = source.PERIODS, source.TRAIN_POLICIES, source.roots, source.options
arguments = diagnosis.arguments
TREATMENTS = ("option_credit", "episode_credit")
KL_BUDGET = source.KL_BUDGET
SOURCE_PREFLIGHT_RUN = "pointmaze_backtracking_stage60_preflight_20261001_r1"
SOURCE_FULL_RUN = "pointmaze_backtracking_stage60_full_20261001_r1"


def source_result(root, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    roles = source.source.previous.seed_roles(root, preflight=preflight)
    return {"calibration": roles["warmup"],
            "first_training_probe": roles["training"][:options(preflight=preflight)["rollouts_per_iteration"]]}


def budget(*, preflight):
    old, opt = source.budget(preflight=preflight), options(preflight=preflight)
    cases = len(PERIODS) * len(TRAIN_POLICIES)
    updates = len(PERIODS) * (sum(len(source.source.levels(a, "train")) for a in TRAIN_POLICIES) + len(TRAIN_POLICIES))
    warmup = 3 * cases * opt["critic_warmup_iterations"]
    return {"native_steps": 0, "new_evaluation_steps": 0, "new_forecaster_fits": 0, "supervised_steps": 0,
        "archive_episodes": old["archive_episodes"],
        "reconstructed_lower_calls": old["reconstructed_lower_calls"], "reconstructed_upper_calls": old["reconstructed_upper_calls"],
        "episode_critic_scalar_calls": old["reconstructed_lower_calls"], "archive_network_checks": 2 * old["archive_episodes"],
        "warmup_critic_updates": warmup, "diagnostic_updates": updates,
        "ppo_gae_calls": warmup + updates, "diagnostic_gae_calls": updates + 3 * cases,
        "diagnostic_distribution_passes": 2 * updates, "diagnostic_value_passes": 2 * updates,
        "probe_mc_calls": 2 * cases, "source_clone_loads": len(PERIODS), "forecaster_loads": 1,
        "warmup_actor_state_checks": 8 * cases, "upper_state_transfers": 4 * cases,
        "upper_pair_state_checks": 4 * cases, "candidate_checkpoint_writes": len(TREATMENTS) * cases}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL, "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN],
        "periods": list(PERIODS), "arms": list(TRAIN_POLICIES), "treatments": list(TREATMENTS),
        "initialization": "same_Stage55_clone_networks_and_Adam_no_reset",
        "pairing": "same_Stage57_calibration_and_first_training_archives_actions_states_rewards_logp_noise_and_shuffle",
        "option_credit": "exact_Stage60_backtracking_first_update_and_original_critic_calibration",
        "episode_credit": "only_lower_done_at_true_episode_end_lower_critic_recalibrated_with_same_PPO_GAE_and_nominal_value_steps",
        "upper": "control_upper_calibration_and_first_update_executed_once_shared_final_four_upper_network_Adam_states",
        "guard": source.contract()["intervention"], "kl_budget": KL_BUDGET,
        "probe": "first_training_batch_unseen_by_calibration_pre_actor_MC_fit_and_actual_pre_update_advantage_alignment",
        "cost": "shared_archive_reconstruction_plus_every_extra_episode_critic_scalar_call_lower_calibration_update_guard_retry_and_checkpoint_write",
        "decision": "full_native_prerequisite_all_active_actors_nonzero_and_all_episode_probe_EV_positive_MSE_below_option_critic_on_episode_MC",
        "selection": "all_roots_both_periods_both_arms_same_hyperparameters_and_calibration_budget_no_seed_KL_or_fit_sweep",
        "limits": "archive_first_update_credit_plus_consistent_critic_package_not_new_native_reward_or_full_training_confirmation"}
