"""Frozen offline decomposition of the complete Stage52 recovery trajectories."""

from scripts import pointmaze_plan_alignment_stage52_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_plan_error_stage53_v1"
SOURCE_RUN = "pointmaze_plan_alignment_stage52_full_20260930_r2"
ERROR_METRICS = ("forecast_ise", "controller_ise", "cross_ise", "total_vector_ise")
METRICS = (*ERROR_METRICS, "episode_return")
ENDPOINTS = tuple(f"period{p}:curve_minus_hold:{m}" for p in source.PERIODS for m in ERROR_METRICS)
PARTITIONS = {"event": ("stable", "regime_only", "geometry_only", "mixed"),
              "timing": ("clean", "history_only", "future_only", "both"),
              "phase": ("early", "middle", "late")}
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (53, 53053)


def budget():
    roots = source.roots(preflight=False)
    paths = len(roots) * source.options(preflight=False)["evaluation_paths"]
    traces = paths * len(source.PERIODS) * len(source.POLICIES) * len(source.MODES)
    return {"raw_trace_reads": traces, "recorded_steps_processed": sum(
        source.arguments(r, preflight=False).horizon for r in roots)
        * source.options(preflight=False)["evaluation_paths"]
        * len(source.PERIODS) * len(source.POLICIES) * len(source.MODES),
        "exogenous_driver_regenerations": paths, "new_native_steps": 0, "optimizer_steps": 0}


def contract():
    return {"source_run": SOURCE_RUN, "roots": list(source.roots(preflight=False)),
        "periods": list(source.PERIODS), "modes": list(source.MODES), "policies": list(source.POLICIES),
        "alignment": "achieved_after_minus_target_before_same_as_native_reward",
        "identity": "total=controller+forecast+2_dot_controller_forecast_signed_cross_term",
        "labels": "offline_only_regenerated_driver_matches_all_recorded_measurements",
        "regime_effect_step": "driver_change_step_plus_one_first_changed_target_increment",
        "geometry": "dominant_motion_axis_change_or_non_regime_direction_reversal",
        "event_window": "valid_64_target_fit_history_increments_plus_future_option_target_increments",
        "partitions": PARTITIONS, "phase": "equal_normalized_age_thirds",
        "comparison": "curve_minus_hold_paired_seed_and_root",
        "primary": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": list(BOOTSTRAP_SEED), "interval": "equal_root_paired_percentile_Bonferroni8",
        "strata": "descriptive_per_episode_contributions_and_pooled_conditional_squared_errors_no_subgroup_CIs",
        "selection": "no_new_paths_or_policy_gain_window_period_mode_tuning",
        "limits": "post_outcome_mechanism_diagnosis_not_causal_mediation_or_new_performance_confirmation"}
