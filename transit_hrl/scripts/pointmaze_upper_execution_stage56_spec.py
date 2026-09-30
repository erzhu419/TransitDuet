"""Same-policy ablation of the executed learned upper residual."""

import numpy as np
from scripts import pointmaze_learned_plan_stage55_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_upper_execution_stage56_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_upper_execution_stage56.py"
POLICY, METHODS = "upper_execution", ("task_clock",)
POLICIES = ("normal", "zero_residual")
PERIODS, MODES, METRICS = previous.PERIODS, previous.MODES, previous.METRICS
roots, arguments = previous.roots, previous.arguments
SOURCE_FULL_RUN = "pointmaze_learned_plan_stage55_full_20260930_r1"
SOURCE_PREFLIGHT_RUN = "pointmaze_learned_plan_stage55_preflight_20260930_r1"
ENDPOINTS = tuple(f"period{p}:normal_minus_zero_residual" for p in PERIODS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (56, 56056)


def source_result(root, *, preflight):
    run = SOURCE_PREFLIGHT_RUN if preflight else SOURCE_FULL_RUN
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def options(*, preflight):
    return {"evaluation_paths": 2 if preflight else 16, "workers": 1 if preflight else 8}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 15_090_000 if preflight else 15_100_000 + index * 10000
    return {"evaluation": list(range(base + 5001, base + 5001 + options(preflight=preflight)["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([56, root, seed, 56017]).generate_state(1)[0])


def rollout_arguments(root, seed, *, mode):
    sampled = mode == "lower_sampled"
    return {"sample": False, "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([56, root, seed, 56019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    n = len(POLICIES) * len(MODES) * options(preflight=preflight)["evaluation_paths"]
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    calls = n * sum(horizon // p for p in PERIODS)
    fits = calls - n * len(PERIODS)
    steps = n * len(PERIODS) * horizon
    basis = n * sum(p + 1 for p in PERIODS)
    return {"total_primitive_steps": steps, "native_trace_audits": n * len(PERIODS),
        "upper_inference_calls": calls, "lower_inference_calls": steps, "gate_inference_calls": 0,
        "plan_ols_fits": fits, "audit_ols_fits": fits, "plan_ridge_predictions": fits, "audit_ridge_predictions": fits,
        "reference_evaluations": steps, "actor_context_evaluations": steps, "upper_plan_decodes": calls,
        "bernstein_basis_evaluations": basis, "audit_bernstein_basis_evaluations": basis,
        "checkpoint_loads": len(PERIODS), "forecaster_loads": 1, "new_forecaster_fits": 0,
        "optimizer_steps": 0, "verification_primitive_steps": 0}


def contrasts(means):
    return {f"period{p}:normal_minus_zero_residual": means[str(p)]["deterministic"]["normal"]["episode_return"]
            - means[str(p)]["deterministic"]["zero_residual"]["episode_return"] for p in PERIODS}


def contract():
    return {"source_protocol": previous.EXPERIMENT_PROTOCOL, "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN],
        "checkpoint": "fixed_final_joint_ppo_each_root_period_no_selection", "forecaster": "same_saved_Stage55_forecaster_no_refitting",
        "policies": list(POLICIES), "periods": list(PERIODS), "normal": "unchanged_Stage55_executed_residual_plan",
        "zero_residual": "infer_same_upper_then_execute_zero_four_dim_action_on_same_causal_base_forecast",
        "lower": "same_saved_joint_MLP_weights_std_and_velocity_context_rule_no_analytic_feedback_or_retraining",
        "credit": "native_episode_return_no_shaping_or_new_objective", "pairing": "fresh_shared_environment_and_stepwise_lower_noise_seeds",
        "audit": "proposed_and_executed_actions_separate_independent_plan_velocity_reconstruction_exact_frozen_networks_and_native_metrics",
        "cost": "both_arms_charge_all_actual_upper_and_lower_calls_no_gate_or_preview",
        "primary_mode": "deterministic", "secondary_mode": "lower_sampled", "endpoints": list(ENDPOINTS),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
        "interval": "two_sided_percentile_Bonferroni2_equal_root_paired_means",
        "decision": "normal_minus_zero_residual_positive_at_both_periods",
        "selection": "no_checkpoint_period_mode_root_seed_or_training_budget_selection_after_outcomes",
        "limits": "execution_ablation_with_coadapted_lower_conditional_on_reused_training_roots_not_frequency_or_promotion_confirmation"}
