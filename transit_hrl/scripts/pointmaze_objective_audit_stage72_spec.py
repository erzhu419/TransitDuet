"""Frozen native-versus-discounted episode objective audit."""

from scripts import pointmaze_calibrated_cv_stage71_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_objective_audit_stage72_v1"
POLICY = "objective_audit"
RUNNER_SCRIPT = "scripts/run_pointmaze_objective_audit_stage72.py"
PERIODS, TRAIN_POLICIES = source.PERIODS, source.TRAIN_POLICIES
roots, arguments = source.roots, source.arguments
legacy = source.source
ESTIMATORS = (*legacy.ESTIMATORS, "mc_native", "mc_native_zero", "mc_discounted_objective")


def options(*, preflight):
    return {**legacy.options(preflight=preflight),
        "calibration_episodes": legacy.values_source.options(preflight=preflight)["rollouts_per_iteration"]}


def seed_roles(root, *, preflight):
    roles = source.seed_roles(root, preflight=preflight)
    n = options(preflight=preflight)["calibration_episodes"]
    return {"first_calibration": roles["calibration"][:n], "archive_batches": roles["archive_batches"]}


def prerequisite_result(root, *, preflight):
    run = "pointmaze_calibrated_cv_stage71_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    old, opt = legacy.budget(preflight=preflight), options(preflight=preflight)
    cases = len(PERIODS) * len(TRAIN_POLICIES)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    cal = opt["calibration_episodes"]
    return {"calibration_archive_episodes": cases * cal, "probe_archive_episodes": old["archive_episodes"],
        "reconstructed_lower_calls": old["reconstructed_lower_calls"] + cases * cal * h,
        "reconstructed_upper_calls": old["reconstructed_upper_calls"] + len(TRAIN_POLICIES) * cal * sum(h // p for p in PERIODS),
        "archive_network_checks": old["archive_network_checks"] + cases * cal,
        "source_clone_loads": len(PERIODS), "forecaster_loads": 1, "critic_checkpoint_loads": 2 * cases,
        "probe_value_rows": old["probe_value_rows"], "mc_calls": 2 * old["mc_calls"],
        "gae_calls": old["gae_calls"], "historical_reward_rate_fits": cases,
        "native_return_identity_checks": old["mc_calls"], "source_gradient_checks": cases,
        "legacy_variance_identity_checks": old["control_variate_identity_checks"],
        "actor_score_forward_batches": old["actor_score_forward_batches"],
        "actor_score_backward_batches": 10 * old["actor_score_forward_batches"],
        "frozen_model_checks": old["frozen_model_checks"]}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "estimators": list(ESTIMATORS),
        "native_objective": "loss_gradient_of_negative_expected_undiscounted_episode_task_reward_divided_by_fixed_H",
        "native_signal": "undiscounted_reward_to_go_minus_remaining_steps_times_first_historical_batch_task_reward_mean",
        "discounted_objective": "gamma_power_episode_time_times_discounted_reward_to_go_minus_unchanged_common_baseline",
        "legacy_surrogate": "unchanged_uniform_time_discounted_MC_and_native_lambda_GAE_not_exact_native_reward_gradients",
        "pairing": "all_Stage69_archives_exact_roots_cases_execution_and_episode_order_no_resampling",
        "baseline": "only_first_Stage57_warmup_batch_reward_rate_mean_no_probe_frame_critic_or_coefficient_fit",
        "checks": "native_return_recursion_identity_Stage70_controls_reproduced_and_models_Adam_bitexact",
        "primary": "raw_episode_covariance_signed_SNR_repeatability_and_cross_independent_batch_objective_cosines",
        "secondary": "separate_per_batch_PPO_normalization_no_same_batch_reference_as_truth",
        "decision": "diagnosis_only_no_actor_or_critic_update_no_gamma_lambda_or_coefficient_sweep_Stage67_HOLD_unchanged",
        "limits": "reused_teacher_initialized_development_roots_and_probes_no_native_performance_or_frequency_claim_noisy_MC_reference_is_not_true_gradient"}
