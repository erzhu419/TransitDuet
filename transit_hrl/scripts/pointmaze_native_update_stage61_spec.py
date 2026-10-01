"""Freeze native evaluation of the exact Stage60 first-update policies."""

from scripts import pointmaze_backtracking_stage60_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_native_update_stage61_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_native_update_stage61.py"
POLICY, METHODS = "native_first_update", source.METHODS
PERIODS, TRAIN_POLICIES, TREATMENTS = source.PERIODS, source.TRAIN_POLICIES, source.TREATMENTS
roots = source.roots
deployment = source.source.previous
arguments, METRICS = deployment.arguments, deployment.METRICS
MODE = "deterministic"
SOURCE_PREFLIGHT_RUN = "pointmaze_backtracking_stage60_preflight_20261001_r1"
SOURCE_FULL_RUN = "pointmaze_backtracking_stage60_full_20261001_r1"
COMPARATORS = ("plain", "conditional_kl", "clone")
ENDPOINTS = tuple(f"period{p}:{a}:backtracking_kl_minus_{b}"
    for p in PERIODS for a in TRAIN_POLICIES for b in COMPARATORS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (61, 61061)


def options(*, preflight):
    return {**source.options(preflight=preflight), "evaluation_paths": 2 if preflight else 16,
            "deployment_mode": MODE}


def source_result(root, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 61_090_000 if preflight else 61_100_000 + index * 10000
    return {"evaluation": list(range(base + 5001, base + 5001 + options(preflight=preflight)["evaluation_paths"]))}


def budget(*, preflight):
    paths = options(preflight=preflight)["evaluation_paths"]
    per_period = (1 + len(TRAIN_POLICIES) * len(TREATMENTS)) * paths
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    calls = per_period * sum(horizon // p for p in PERIODS)
    episodes = per_period * len(PERIODS)
    steps, fits = episodes * horizon, calls - episodes
    basis = per_period * sum(p + 1 for p in PERIODS)
    return {"archive": source.budget(preflight=preflight),
        "native_evaluation": {"primitive_steps": steps, "upper_inference_calls": calls,
            "lower_inference_calls": steps, "gate_inference_calls": 0,
            "plan_ols_fits": fits, "audit_ols_fits": fits, "plan_ridge_predictions": fits,
            "audit_ridge_predictions": fits, "reference_evaluations": steps,
            "actor_context_evaluations": steps, "upper_plan_decodes": calls,
            "bernstein_basis_evaluations": basis, "audit_bernstein_basis_evaluations": basis},
        "native_trace_audits": episodes,
        "candidate_checkpoint_writes": len(PERIODS) * len(TRAIN_POLICIES) * len(TREATMENTS),
        "new_forecaster_fits": 0, "supervised_steps": 0}


def contrasts(means):
    return {f"period{p}:{a}:backtracking_kl_minus_{b}":
        means[str(p)][a]["backtracking_kl"]["episode_return"] - means[str(p)][a][b]["episode_return"]
        for p in PERIODS for a in TRAIN_POLICIES for b in COMPARATORS}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL,
        "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN],
        "archive": "exact_Stage60_first_update_comparisons_and_costs_same_Stage57_warmup_and_batch",
        "policies": ["clone", *TREATMENTS], "arms": list(TRAIN_POLICIES), "periods": list(PERIODS),
        "clone": "one_unmodified_Stage55_clone_evaluation_per_period_shared_across_arms",
        "deployment": "fixed_one_update_models_deterministic_native_execution_rule_of_each_training_arm",
        "unchanged": "credit_critics_forecaster_KL_budget_backtracking_roots_periods_and_PPO_initialization",
        "pairing": "fresh_shared_environment_paths_all_treatments_arms_periods_no_new_training_rollouts",
        "policy_seed": "unchanged_Stage57_policy_seed_with_new_environment_seed",
        "cost": "count_Stage60_reconstruction_and_all_new_native_calls_audits_and_candidate_checkpoint_writes_separately",
        "endpoints": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": list(BOOTSTRAP_SEED),
        "interval": "two_sided_percentile_Bonferroni12_equal_root_paired_path_means",
        "decision": "repair_gain_backtracking_vs_plain_and_training_gain_vs_clone_each_require_positive_all_arms_and_periods",
        "selection": "no_checkpoint_root_period_path_LR_credit_or_budget_selection_after_results",
        "limits": "single_update_reused_teacher_initialized_training_roots_new_eval_paths_not_full_training_frequency_specific_or_equal_FLOPs_confirmation"}
