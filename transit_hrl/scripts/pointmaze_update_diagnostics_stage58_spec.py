"""Replay archived Stage57 updates without collecting new environment data."""

from scripts import pointmaze_matched_upper_stage57_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_update_diagnostics_stage58_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_update_diagnostics_stage58.py"
POLICY, METHODS = "update_diagnostics", ("task_clock",)
PERIODS, TRAIN_POLICIES = previous.PERIODS, previous.TRAIN_POLICIES
roots, options = previous.roots, previous.options
SOURCE_PREFLIGHT_RUN = "pointmaze_matched_upper_stage57_preflight_20260930_r1"
SOURCE_FULL_RUN = "pointmaze_matched_upper_stage57_full_20260930_r1"


def source_result(root, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def levels(arm, phase):
    return ("upper", "lower") if phase == "warmup" or arm == "joint_ppo" else ("lower",)


def budget(*, preflight):
    opt, old = options(preflight=preflight), previous.budget(preflight=preflight)
    updates = len(PERIODS) * sum(len(levels(arm, phase)) * iterations for arm in TRAIN_POLICIES
        for phase, iterations in (("warmup", opt["critic_warmup_iterations"]), ("train", opt["learning_iterations"])))
    return {"native_steps": 0, "new_fits": 0, "new_evaluation_steps": 0,
        "archive_episodes": len(PERIODS) * len(TRAIN_POLICIES) * opt["rollouts_per_iteration"]
            * (opt["critic_warmup_iterations"] + opt["learning_iterations"]),
        "reconstructed_lower_calls": sum(old["primitive_steps"][p] for p in ("warmup", "train")),
        "reconstructed_upper_calls": sum(old["upper_calls"][p] for p in ("warmup", "train")),
        "diagnostic_updates": updates, "diagnostic_distribution_passes": 2 * updates,
        "diagnostic_value_passes": 2 * updates, "diagnostic_gae_calls": updates,
        "source_clone_loads": len(PERIODS), "forecaster_loads": 1,
        "final_checkpoint_loads": len(PERIODS) * len(TRAIN_POLICIES)}


def contract():
    return {"source_protocol": previous.EXPERIMENT_PROTOCOL,
        "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN], "periods": list(PERIODS),
        "arms": list(TRAIN_POLICIES), "sampling": "archived_observations_rewards_contexts_and_original_policy_noise_seeds",
        "execution": "reconstruct_actual_proposed_upper_and_pre_squash_lower_actions_no_environment_calls",
        "updates": "unchanged_Stage57_PPO_credit_shuffle_optimizers_and_iteration_roster",
        "identity": "exact_archived_upper_and_executed_lower_actions_and_final_four_networks_four_optimizers",
        "diagnostics": "Gaussian_conditional_and_episode_KL_ratio_clipping_std_GAE_target_value_MSE_explained_variance",
        "decision": "diagnosis_only_no_performance_gate_no_stability_intervention_selected_before_diagnostics",
        "selection": "all_eight_roots_both_periods_both_learned_arms_all_warmup_and_training_iterations",
        "limits": "posthoc_development_diagnostics_not_causal_failure_attribution_or_new_performance_validation"}
