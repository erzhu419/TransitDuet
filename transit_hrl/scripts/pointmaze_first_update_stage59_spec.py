"""Freeze an archive-only comparison at Stage57's first actor update."""

from scripts import pointmaze_update_diagnostics_stage58_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_first_update_stage59_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_first_update_stage59.py"
POLICY, METHODS = "first_update", ("task_clock",)
PERIODS, TRAIN_POLICIES = source.PERIODS, source.TRAIN_POLICIES
roots, options = source.roots, source.options
TREATMENTS = ("plain", "conditional_kl")
KL_BUDGET = 0.02
SOURCE_PREFLIGHT_RUN = "pointmaze_update_diagnostics_stage58_preflight_20260930_r1"
SOURCE_FULL_RUN = "pointmaze_update_diagnostics_stage58_full_20260930_r1"


def diagnostic_result(root, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = source.previous.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    episodes_per_arm = (opt["critic_warmup_iterations"] + 1) * opt["rollouts_per_iteration"]
    updates = len(TREATMENTS) * sum(len(source.levels(arm, "train")) for arm in TRAIN_POLICIES) * len(PERIODS)
    return {"native_steps": 0, "new_evaluation_steps": 0, "new_fits": 0,
        "archive_episodes": len(PERIODS) * len(TRAIN_POLICIES) * episodes_per_arm,
        "reconstructed_lower_calls": len(PERIODS) * len(TRAIN_POLICIES) * episodes_per_arm * horizon,
        "reconstructed_upper_calls": len(TRAIN_POLICIES) * episodes_per_arm * sum(horizon // p for p in PERIODS),
        "warmup_critic_updates": len(PERIODS) * len(TRAIN_POLICIES) * 2 * opt["critic_warmup_iterations"],
        "diagnostic_updates": updates, "diagnostic_distribution_passes": 2 * updates,
        "diagnostic_value_passes": 2 * updates, "diagnostic_gae_calls": updates,
        "source_clone_loads": len(PERIODS), "forecaster_loads": 1}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL,
        "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN],
        "periods": list(PERIODS), "arms": list(TRAIN_POLICIES), "treatments": list(TREATMENTS),
        "kl_budget": KL_BUDGET,
        "reference": "fixed_sampling_policy_full_batch_mean_exact_conditional_Gaussian_KL_old_to_new_per_decision",
        "intervention": "after_each_actor_Adam_step_reject_if_KL_exceeds_budget_restore_actor_and_Adam_continue_minibatches_no_backtracking",
        "pairing": "same_reconstructed_first_training_batch_same_four_networks_and_Adam_states_after_exact_archived_critic_warmup",
        "unchanged": "original_credit_targets_critic_updates_LR_entropy_shuffle_and_nominal_actor_step_attempts",
        "identity": "plain_first_update_diagnostics_exact_Stage58_and_paired_final_critic_networks_and_Adam_exact",
        "selection": "all_roots_both_periods_both_arms_no_KL_LR_credit_seed_or_reward_tuning",
        "decision": "mechanical_validity_plus_nonzero_actor_step_prerequisite_no_native_launch_if_any_active_actor_frozen",
        "limits": "archive_only_mechanics_not_reward_gain_causal_attribution_trajectory_KL_or_heldout_confirmation"}
