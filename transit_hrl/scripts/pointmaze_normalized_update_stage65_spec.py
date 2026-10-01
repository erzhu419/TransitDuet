"""Freeze the native first-update test of Stage64's selected critic."""

from scripts import pointmaze_value_targets_stage64_spec as source
from scripts import pointmaze_native_update_stage61_spec as native_source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_normalized_update_stage65_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_normalized_update_stage65.py"
POLICY, METHODS = "normalized_first_update", source.METHODS
PERIODS, TRAIN_POLICIES, roots, arguments = source.PERIODS, source.TRAIN_POLICIES, source.roots, source.arguments
TREATMENTS, CANDIDATE = ("gae_raw", "mc_normalized"), source.CANDIDATE
MODE, KL_BUDGET = native_source.MODE, source.source.KL_BUDGET
deployment = native_source.deployment
SOURCE_PREFLIGHT_RUN = "pointmaze_value_targets_stage64_preflight_20261001_r1"
SOURCE_FULL_RUN = "pointmaze_value_targets_stage64_full_20261001_r1"
COMPARATORS = ("gae_raw", "frozen_lower", "clone")
ENDPOINTS = tuple(f"period{p}:{a}:mc_normalized_minus_{b}"
    for p in PERIODS for a in TRAIN_POLICIES for b in COMPARATORS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (65, 65065)


def options(*, preflight):
    return {"workers": source.options(preflight=preflight)["workers"],
        "rollouts_per_iteration": source.options(preflight=preflight)["rollouts_per_iteration"],
        "actor_iterations": 1, "evaluation_paths": 2 if preflight else 16, "deployment_mode": MODE}


def source_result(root, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def source_qualification(*, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "qualification_summary.json"


def upper_result(root, *, preflight):
    return source.source_result(root, preflight=preflight)


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 65_090_000 if preflight else 65_100_000 + 10000 * index
    return {"first_training": source.seed_roles(root, preflight=preflight)["first_training_probe"],
        "evaluation": list(range(base + 5001, base + 5001 + options(preflight=preflight)["evaluation_paths"]))}


def budget(*, preflight):
    opt = options(preflight=preflight)
    paths, evaluation = opt["rollouts_per_iteration"], opt["evaluation_paths"]
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    cases = len(PERIODS) * len(TRAIN_POLICIES)
    # zero_train frozen lower is the shared clone, not a second execution.
    per_period = (1 + len(TRAIN_POLICIES) * len(TREATMENTS) + 1) * evaluation
    episodes = per_period * len(PERIODS)
    upper = per_period * sum(horizon // p for p in PERIODS)
    fits, basis = upper - episodes, per_period * sum(p + 1 for p in PERIODS)
    return {"archive": {"archive_episodes": cases * paths,
        "reconstructed_lower_calls": cases * paths * horizon,
        "reconstructed_upper_calls": len(TRAIN_POLICIES) * paths * sum(horizon // p for p in PERIODS),
        "extra_critic_scalar_calls": cases * paths * horizon, "archive_network_checks": 2 * cases * paths,
        "source_clone_loads": len(PERIODS), "forecaster_loads": 1, "critic_checkpoint_loads": 2 * cases,
        "upper_checkpoint_loads": cases, "critic_resume_checks": 2 * cases, "probe_mc_calls": cases,
        "source_probe_checks": 2 * cases, "actor_updates": 2 * cases, "actor_gae_calls": 2 * cases,
        "critic_continuation_updates": 2 * cases, "MC_continuation_updates": cases,
        "post_update_public_value_passes": 2 * cases, "upper_frozen_state_checks": 8 * cases,
        "candidate_checkpoint_writes": 2 * cases},
        "native_evaluation": {"primitive_steps": episodes * horizon, "lower_inference_calls": episodes * horizon,
        "upper_inference_calls": upper, "gate_inference_calls": 0, "plan_ols_fits": fits, "audit_ols_fits": fits,
        "plan_ridge_predictions": fits, "audit_ridge_predictions": fits, "reference_evaluations": episodes * horizon,
        "actor_context_evaluations": episodes * horizon, "upper_plan_decodes": upper,
        "bernstein_basis_evaluations": basis, "audit_bernstein_basis_evaluations": basis},
        "native_trace_audits": episodes, "new_forecaster_fits": 0, "new_training_native_steps": 0}


def contrasts(means):
    return {f"period{p}:{a}:mc_normalized_minus_{b}":
        means[str(p)][a][CANDIDATE]["episode_return"] - means[str(p)][a][b]["episode_return"]
        for p in PERIODS for a in TRAIN_POLICIES for b in COMPARATORS}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL, "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN],
        "candidate": CANDIDATE, "treatments": list(TREATMENTS), "arms": list(TRAIN_POLICIES), "periods": list(PERIODS),
        "source_gate": "full_Stage64_candidate_fit_pass_required_preflight_mechanics_only",
        "lower": "same_Stage55_actor_Adam_same_Stage57_first_batch_episode_done_public_reward_unit_GAE",
        "upper": "Stage63_common_upper_actor_value_Adam_loaded_once_per_case_and_frozen_for_both_treatments_and_frozen_lower_control",
        "normalization": "restore_Stage64_training_weights_Adam_fixed_frame_exact_public_weights_no_refit_reset_or_unit_relabel",
        "update": "one_guarded_lower_PPO_update_then_same_budget_critic_continuation_GAE_raw_or_MC_normalized",
        "guard": "unchanged_Stage60_backtracking_conditional_mean_KL", "kl_budget": KL_BUDGET,
        "controls": "pure_clone_shared_per_period_zero_train_frozen_lower_reuses_clone_joint_frozen_lower_executes_common_upper",
        "evaluation": "fresh_shared_deterministic_native_paths_all_arms_treatments_periods_no_selection",
        "cost": "checkpoint_reuse_no_recalibration_all_archive_value_actor_guard_MC_and_native_work_counted",
        "endpoints": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
        "interval": "two_sided_percentile_Bonferroni12_equal_root_paired_path_means",
        "decision": "repair_gain_vs_gae_raw_and_training_gain_vs_frozen_lower_and_clone_require_positive_all_periods_arms",
        "selection": "all_eight_roots_all_cases_no_seed_KL_target_scale_checkpoint_or_budget_sweep",
        "limits": "one_update_teacher_initialized_development_roots_not_full_training_frequency_specific_OOD_or_equal_FLOPs"}
