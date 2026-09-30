"""Frozen forecast x tracking factorial; native return is the adoption endpoint."""

import numpy as np
from scripts import pointmaze_plan_alignment_stage52_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_forecast_tracking_stage54_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_forecast_tracking_stage54.py"
POLICY, METHODS = "forecast_tracking", ("task_clock",)
POLICIES = ("frozen", "target_hold", "linear_position", "linear_velocity", "ridge_position", "ridge_velocity")
PERIODS, MODES, METRICS = previous.PERIODS, previous.MODES, previous.METRICS
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments, warmup_iterations = previous.source_result, previous.roots, previous.arguments, previous.warmup_iterations
LOOKBACK_STEPS, DT_SECONDS = previous.LOOKBACK_STEPS, previous.DT_SECONDS
FORECAST_STEPS, RIDGE_LAMBDA, MOTION_TOLERANCE = 100, 1., .001
EFFECTS = ("forecast_main", "velocity_main", "interaction", "combined_minus_hold", "combined_minus_frozen", "combined_minus_linear_position")
ENDPOINTS = tuple(f"period{p}:{k}" for p in PERIODS for k in EFFECTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (54, 54054)


def options(*, preflight):
    return {"fitting_paths": 2 if preflight else 32, "evaluation_paths": 2 if preflight else 16,
            "workers": 1 if preflight else 8}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 13_090_000 if preflight else 13_100_000 + index * 10000
    opt = options(preflight=preflight)
    return {"fitting": list(range(base + 1, base + 1 + opt["fitting_paths"])),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([54, root, seed, 54017]).generate_state(1)[0])


def rollout_arguments(root, seed, *, mode):
    sampled = mode == "lower_sampled"
    return {"sample": False, "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([54, root, seed, 54019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt, old = options(preflight=preflight), previous.budget(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    builds = 4 * len(MODES) * opt["evaluation_paths"] * (sum(horizon // p for p in PERIODS) - len(PERIODS))
    return {"total_primitive_steps": old["total_primitive_steps"], "native_trace_audits": old["native_trace_audits"],
            "upper_inference_calls": old["upper_inference_calls"], "reference_evaluations": old["reference_evaluations"],
            "plan_ols_fits": builds, "audit_ols_fits": builds,
            "plan_ridge_predictions": builds // 2, "audit_ridge_predictions": builds // 2,
            "actor_context_evaluations": old["total_primitive_steps"] // 3,
            "fitting_native_steps": 0, "fitting_driver_paths": opt["fitting_paths"],
            "fitting_observations": opt["fitting_paths"] * (horizon + 1),
            "fitting_rows": opt["fitting_paths"] * (horizon - FORECAST_STEPS), "ridge_solves": 1}


def contrasts(means):
    result = {}
    for p in PERIODS:
        r = {k: v["episode_return"] for k, v in means[str(p)]["deterministic"].items()}
        lp, lv, rp, rv = [r[k] for k in POLICIES[2:]]
        values = ((rp + rv - lp - lv) / 2, (lv + rv - lp - rp) / 2,
                  (rv - rp) - (lv - lp), rv - r["target_hold"], rv - r["frozen"], rv - lp)
        result.update({f"period{p}:{k}": v for k, v in zip(EFFECTS, values)})
    return result


def contract():
    return {"source_protocol": "pointmaze_critic_clock_stage42_v1", "source_method": "task_clock",
        "source_iteration": "warmup16_full_warmup2_preflight", "policies": list(POLICIES), "periods": list(PERIODS),
        "linear": "unchanged_Stage52_OLS64_anchor_plus_velocity_times_age_times_dt",
        "ridge": "standardized_17_observable_features_intercept_unpenalized_multioutput_future_displacements_lags1_to100",
        "features": "position_last_velocity_mean_velocity4_mean_velocity16_OLS64_observed_motion_run_age_age_times_velocity_position_outer_velocity",
        "motion_run": "consecutive_recent_velocity_difference_norm_leq_tolerance_capped_at63_not_latent_regime",
        "motion_tolerance": MOTION_TOLERANCE, "ridge_lambda": RIDGE_LAMBDA,
        "fit": "one_closed_form_numpy_solve_per_root_disjoint_fitting_driver_paths_no_MuJoCo_fit_steps_no_validation_selection",
        "labels": "future_targets_in_fitting_paths_only_t1_through_horizon_minus100",
        "reference": "predict_once_at_renewal_anchor_visible_target_clip_same_goal_bounds_initial_flat_no_padding",
        "velocity": "forward_difference_of_clipped_frozen_plan_lag_age_plus1_minus_age_over_dt",
        "tracking": "Stage51_same_LQR_gain_clip_source_Gaussian_std_position_only_or_position_plus_planned_velocity",
        "state": "planned_velocity_explicit_two_dim_actor_context_critic_and_cost_state_unchanged",
        "calls": "same_upper_at_all_fixed_renewals_lower_every_step_no_gate_or_actor_value_optimizer_updates",
        "primary_mode": "deterministic", "secondary_mode": "lower_sampled",
        "endpoints": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
        "interval": "two_sided_percentile_Bonferroni12_equal_root_paired_means",
        "decision": "combined_native_reward_positive_vs_hold_frozen_and_linear_position_at_both_periods",
        "selection": "no_gain_window_lambda_feature_root_period_mode_or_checkpoint_selection_after_outcomes",
        "limits": "learned_forecaster_fixed_feedback_conditional_development_not_learned_HRL_confirmation_same_calls_not_equal_FLOPs"}
