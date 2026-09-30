"""Fixed-budget causal reference plans under frozen native feedback."""

import numpy as np
from scripts import pointmaze_lower_learnability_stage51_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_plan_alignment_stage52_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_plan_alignment_stage52.py"
POLICY = "plan_alignment"
METHODS = ("task_clock",)
POLICIES = ("frozen", "waypoint", "target_hold", "target_curve", "reverse_curve", "current_target")
PERIODS = (50, 100)
MODES, METRICS = previous.MODES, (*previous.METRICS, "reference_target_squared_error_integral")
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments, warmup_iterations = previous.source_result, previous.roots, previous.arguments, previous.warmup_iterations
LOOKBACK_STEPS, DT_SECONDS = 64, .01
RETURN_PAIRS = (("target_hold", "waypoint"), ("target_curve", "target_hold"),
                ("target_curve", "reverse_curve"), ("target_curve", "frozen"))
ENDPOINTS = tuple(f"period{p}:{a}_minus_{b}" for p in PERIODS for a, b in RETURN_PAIRS)
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (52, 52052)


def options(*, preflight):
    return {"evaluation_paths": 2 if preflight else 16, "workers": 1 if preflight else 8}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 12_090_000 if preflight else 12_100_000 + index * 10000
    return {"evaluation": list(range(base + 3001, base + 3001 + options(preflight=preflight)["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([52, root, seed, 52017]).generate_state(1)[0])


def rollout_arguments(root, seed, *, mode):
    sampled = mode == "lower_sampled"
    return {"sample": False, "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([52, root, seed, 52019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    per_period = len(POLICIES) * len(MODES) * opt["evaluation_paths"]
    calls = sum(horizon // p for p in PERIODS)
    fits = 2 * len(MODES) * opt["evaluation_paths"] * (calls - len(PERIODS))
    return {"total_primitive_steps": len(PERIODS) * per_period * horizon,
            "native_trace_audits": len(PERIODS) * per_period,
            "upper_inference_calls": per_period * calls,
            "plan_regression_fits": fits, "audit_regression_fits": fits,
            "reference_evaluations": len(PERIODS) * per_period * horizon}


def contrasts(means):
    return {f"period{p}:{a}_minus_{b}": means[str(p)]["deterministic"][a]["episode_return"]
            - means[str(p)]["deterministic"][b]["episode_return"] for p in PERIODS for a, b in RETURN_PAIRS}


def contract():
    return {"source_protocol": "pointmaze_critic_clock_stage42_v1", "source_method": "task_clock",
        "source_iteration": "warmup16_full_warmup2_preflight_not_post_actor_update",
        "periods": list(PERIODS), "policies": list(POLICIES),
        "renewal": "fixed_exogenous_period_no_gate_no_future_reference_trajectory_schedule_replay",
        "upper": "same_source_upper_actor_called_once_per_fixed_renewal_all_arms",
        "waypoint": "original_decoded_upper_anchor_held_between_calls",
        "target_hold": "currently_observed_target_at_renewal_held_for_the_option",
        "target_curve": "same_observed_anchor_plus_OLS_target_velocity_times_option_age_seconds",
        "reverse_curve": "same_OLS_and_anchor_opposite_velocity_negative_control",
        "current_target": "per_primitive_observed_target_diagnostic_not_low_frequency_plan",
        "forecast": "numpy_lstsq_position_on_intercept_and_past_relative_time_at_renewals_only",
        "lookback_steps": LOOKBACK_STEPS, "dt_seconds": DT_SECONDS,
        "initialization": "valid_observations_only_initial_zero_velocity_no_padding_fit",
        "bounds": "clip_evaluated_reference_to_same_environment_goal_bounds_no_maze_route_or_future_access",
        "lower": "Stage51_fixed_LQR_gain_action_clip_source_Gaussian_std_no_velocity_feedforward",
        "frozen": "original_source_lower_MLP_reference_control_all_networks_critics_optimizers_unchanged",
        "state": "separate_upper_anchor_and_lower_phase_reference_unchanged_actor_dimensions",
        "sampling": "deterministic_upper_lower_deterministic_and_sampled_same_noise_seeds",
        "cost": "equal_upper_lower_calls_per_period_zero_gate_or_optimizer_steps_all_plan_and_audit_regressions_counted",
        "primary_mode": "deterministic", "secondary_mode": "lower_sampled",
        "primary_endpoints": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": list(BOOTSTRAP_SEED), "interval": "two_sided_percentile_Bonferroni8_equal_root_paired_means",
        "decision": "curve_vs_hold_reverse_and_frozen_positive_at_both_fixed_periods_for_plan_phase_utility",
        "selection": "no_gain_or_window_tuning_no_root_extension_no_period_or_deployment_mode_selection",
        "limits": "analytic_plan_feedback_development_not_learned_FreqHRL_confirmation_extra_OLS_compute_not_free_same_call_budget_not_equal_FLOPs"}
