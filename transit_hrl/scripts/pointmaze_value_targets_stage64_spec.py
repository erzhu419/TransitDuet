"""Isolate continuing-critic target horizon and output units."""

from scripts import pointmaze_episode_credit_stage63_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_value_targets_stage64_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_value_targets_stage64.py"
POLICY, METHODS = "value_targets", source.METHODS
PERIODS, TRAIN_POLICIES, roots, arguments, seed_roles = source.PERIODS, source.TRAIN_POLICIES, source.roots, source.arguments, source.seed_roles
TREATMENTS = ("gae_raw", "mc_raw", "gae_normalized", "mc_normalized")
CANDIDATE, MIN_PROBE_EV, VALUE_BATCH_SIZE = "mc_normalized", .10, 512
SOURCE_PREFLIGHT_RUN = "pointmaze_episode_credit_stage63_preflight_20261001_r1"
SOURCE_FULL_RUN = "pointmaze_episode_credit_stage63_full_20261001_r1"


def source_result(root, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def training_result(root, *, preflight):
    return source.source.source.source_result(root, preflight=preflight)


def options(*, preflight):
    opt = source.options(preflight=preflight)
    return {k: opt[k] for k in ("critic_warmup_iterations", "rollouts_per_iteration", "workers")} | {
        "actor_updates": 0, "value_batch_size": VALUE_BATCH_SIZE}


def budget(*, preflight):
    old, opt = source.budget(preflight=preflight), options(preflight=preflight)
    cases, warm = len(PERIODS) * len(TRAIN_POLICIES), opt["critic_warmup_iterations"]
    n = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon * opt["rollouts_per_iteration"]
    return {"native_steps": 0, "new_evaluation_steps": 0, "new_forecaster_fits": 0, "actor_optimizer_steps": 0,
        "archive_episodes": old["archive_episodes"], "reconstructed_lower_calls": old["reconstructed_lower_calls"],
        "reconstructed_upper_calls": old["reconstructed_upper_calls"],
        "extra_critic_scalar_calls": 3 * old["reconstructed_lower_calls"], "archive_network_checks": 4 * old["archive_episodes"],
        "calibration_updates": 4 * cases * warm, "mc_supervised_updates": 2 * cases * warm,
        "calibration_gae_calls": 2 * cases * warm, "mc_target_calls": cases * (warm + 1), "probe_gae_calls": 4 * cases,
        "normalization_frame_fits": cases, "initialization_checks": 2 * cases, "initialization_value_rows": 2 * cases * n,
        "frozen_state_checks": 6 * 4 * cases, "representation_forward_batches": 4 * cases * ((n + VALUE_BATCH_SIZE - 1) // VALUE_BATCH_SIZE),
        "source_clone_loads": len(PERIODS), "forecaster_loads": 1, "critic_checkpoint_writes": 4 * cases}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL, "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN],
        "periods": list(PERIODS), "arms": list(TRAIN_POLICIES), "treatments": list(TREATMENTS),
        "initialization": "same_Stage55_clone_values_empty_value_Adam_no_reset_same_public_initial_prediction",
        "pairing": "same_Stage57_calibration_and_first_training_probe_archives_causal_states_actions_rewards_logp_and_shuffle",
        "control": "gae_raw_pre_actor_episode_MC_probe_exactly_reproduces_Stage63_episode_critic",
        "targets": "continuing_existing_PPO_GAE_vs_discounted_complete_episode_MC_same_gamma_true_episode_done",
        "normalization": "fixed_mean_std_from_first_calibration_episode_MC_only_rebase_last_linear_head_train_in_normalized_units_export_public_values",
        "architecture": "unchanged_two_Tanh_ValueNet_actor_upper_networks_and_their_Adam_all_frozen",
        "training": "same_nominal_value_steps_LR_epochs_minibatches_coef_gradient_clip_all_extra_MC_supervision_and_forward_calls_counted",
        "probe": "first_training_paths_disjoint_from_all_calibration_no_probe_target_or_prediction_used_by_training_or_normalization",
        "decision": {"candidate": CANDIDATE, "minimum_EV_each_case": MIN_PROBE_EV,
                     "MSE_each_case": "below_gae_raw", "next": "critic_only_pass_still_requires_guarded_actor_and_native_trial"},
        "selection": "all_eight_roots_both_periods_both_arms_no_fit_LR_seed_sweep_no_best_arm_selection",
        "checkpoint": "critic_only_training_units_Adam_and_normalization_plus_public_value_weights_not_a_native_ready_full_policy",
        "limits": "archive_critic_fit_factorial_not_reward_frequency_OOD_or_full_training_confirmation"}
