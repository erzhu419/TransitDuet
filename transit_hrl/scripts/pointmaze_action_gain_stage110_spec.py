"""Frozen native directional gain above the current optional-advice lower."""
from scripts import pointmaze_control_response_stage109_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
roots, arguments, task_options = source.roots, source.arguments, source.task_options
EXPERIMENT_PROTOCOL = "pointmaze_action_gain_stage110_v1"
POLICY = "fixed_lower_deterministic_upper_directional_intervention"
RUNNER_SCRIPT = "scripts/run_pointmaze_action_gain_stage110.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_action_gain_stage110.py"
METHODS = {}
EPSILON, ALPHA = source.EPSILON, 1.
PANELS = ("A", "B")
DIRECTIONS = tuple(f"axis{i}_{s}" for i in range(4) for s in ("plus", "minus"))
VARIANTS = ("forecast", "blind", "zero", "mean", *DIRECTIONS)
METRICS = ("mean_minus_forecast", *(f"{v}_minus_forecast" for v in DIRECTIONS),
    *(f"axis{i}_slope" for i in range(4)), *(f"axis{i}_curvature" for i in range(4)), "forecast_minus_blind")
ENDPOINTS = tuple(f"{p}/{m}" for p in PERIODS for m in METRICS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (110, 110110)


def options(*, preflight):
    return {"workers": 2 if preflight else 4, "paths_per_panel": 2 if preflight else 8}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 110000000 if preflight else 110100000 + roots(preflight=False).index(root)*100000
    n = options(preflight=preflight)["paths_per_panel"]
    return {"panels": {name: list(range(base+95001+1000*i, base+95001+1000*i+n)) for i, name in enumerate(PANELS)}}


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    pairs = len(PANELS)*o["paths_per_panel"]
    n = len(PERIODS)*pairs*len(VARIANTS)
    renewals = pairs*sum(h//p for p in PERIODS)
    planned = len(VARIANTS)-1
    return {"native_episodes": n, "native_steps": n*h, "native_lower_calls": n*h,
        "native_upper_calls": (1+len(DIRECTIONS))*renewals,
        "plan_renewals": planned*renewals, "plan_ols_fits": planned*pairs*sum(h//p-1 for p in PERIODS),
        "plan_ridge_predictions": planned*pairs*sum(h//p-1 for p in PERIODS),
        "reference_evaluations": planned*len(PERIODS)*pairs*h,
        "actor_context_evaluations": planned*len(PERIODS)*pairs*h,
        "native_network_checks": n, "native_pair_groups": len(PERIODS)*pairs,
        "initial_feedback_pair_checks": len(PERIODS)*pairs*(len(VARIANTS)-1),
        "external_measurement_pair_checks": len(PERIODS)*pairs*(len(VARIANTS)-1),
        "lower_innovation_pair_checks": len(PERIODS)*pairs*(len(VARIANTS)-1),
        "zero_forecast_command_checks": len(PERIODS)*pairs, "source_freeze_checks": len(PERIODS)}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "donors": "all8_Stage107_final_learned_hint_lowers_fixed_Stage106_upper",
        "task": source.contract()["task"], "periods": list(PERIODS), "variants": list(VARIANTS),
        "feedback": "all392_flat_features_plus4_advice_same396_actor398_value",
        "upper": "deterministic_current_mean_with_fixed_plus_minus0.25_coordinate_offset_every_renewal_std_unchanged_unused",
        "decoder": "existing_Bernstein_tanh_clip_full_alpha1_no_scale_search_zero_action_exact_forecast",
        "lower": "same_frozen_learned_hint_lower_original_stochastic_Gaussian_box_action_paired_per_step_noise",
        "panels": "fresh_disjoint_A_B8_paths_each_per_root_period_pref2_no_candidate_selection",
        "statistics": "36_equal_root_pooled_panel_effects_bootstrap65536_Bonferroni36_seed110_110110",
        "directional_signal": "central_slope_Rplus_minus_Rminus_over2epsilon_curvature_Rplus_plus_Rminus_minus2Rmean_overepsilon_squared",
        "gain_gate": "per_period_any_direction_minus_forecast_corrected_CI_positive_with_matching_nonzero_slope_CI_and_same_sign_A_B_equal_root_slopes",
        "decision": "diagnostic_gain_detected_both_periods_partial_or_not_supported_no_production_action_selection",
        "admission": "frozen_pref_full_before_native_pref_mechanical_admission_only",
        "artifacts": "no_training_checkpoint_or_trace_writes_temporary_pair_audits_discarded_server_compact_JSON_pull",
        "limits": "whole_episode_upper_bias_direction_diagnostic_not_local_Q_learned_policy_superiority_or_frequency_superiority"}
