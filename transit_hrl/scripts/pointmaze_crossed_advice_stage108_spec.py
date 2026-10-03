"""Fixed-lower execution isolates learned upper content from forecast and noise."""

from scripts import pointmaze_optional_plan_stage107_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
roots, arguments, task_options = source.roots, source.arguments, source.task_options
SOURCE_RUN = "pointmaze_optional_plan_stage107_full_20261004_r1"
LOWERS = ("learned_hint", "forecast_hint")
PLAN_MODES = ("learned", "forecast", "noise", "blind")
VARIANTS = {f"{lower}_{plan}": (lower, plan) for lower in LOWERS for plan in PLAN_MODES}
CONTRAST_PAIRS = (("learned", "forecast"), ("learned", "noise"), ("forecast", "blind"),
    ("learned", "blind"), ("noise", "forecast"))
ENDPOINTS = tuple(f"{p}/{lower}/{a}_minus_{b}" for p in PERIODS for lower in LOWERS for a, b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/learned_hint/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS[:2])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (108, 108108)
EXPERIMENT_PROTOCOL = "pointmaze_crossed_advice_stage108_v1"
POLICY = "fixed_lower_crossed_learned_forecast_noise_blind"
METHODS = {}
RUNNER_SCRIPT = "scripts/run_pointmaze_crossed_advice_stage108.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_crossed_advice_stage108.py"


def options(*, preflight):
    return {"workers": 2 if preflight else 4, "evaluation_episodes": 4 if preflight else 32}


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def donor_checkpoint(root, period, method):
    return source_result(root).parent/"final_weights"/f"period_{period}_{method}.pt"


def source_record(root):
    return {"initialization": source.source_record(root), "donor_protocol": source.EXPERIMENT_PROTOCOL,
        "donor_result": str(source_result(root)), "donors": {str(p): {m: str(donor_checkpoint(root, p, m))
            for m in LOWERS} for p in PERIODS}, "selection": "all_eight_roots_both_periods_final_update8_no_donor_selection"}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 108000000 if preflight else 108100000 + roots(preflight=False).index(root)*100000
    return {"native_evaluation": list(range(base+95001, base+95001+options(preflight=preflight)["evaluation_episodes"]))}


def budget(*, preflight):
    e = options(preflight=preflight)["evaluation_episodes"]
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n = len(PERIODS)*len(VARIANTS)*e
    return {"source_clone_loads": 2, "forecaster_loads": 1, "decoder_loads": 2, "source_cell_loads": 2,
        "donor_checkpoint_loads": 8, "expanded_model_initializations": 2, "frozen_lower_models_initialized": 4, "donor_freeze_checks": 4,
        "zero_mean_upper_interventions": 4, "native_episodes": n, "native_steps": n*h,
        "native_lower_calls": n*h, "native_upper_calls": 2*len(LOWERS)*e*sum(h//p for p in PERIODS),
        "native_network_checks": n, "native_pair_checks": len(PERIODS)*e,
        "upper_noise_pair_checks": len(PERIODS)*len(LOWERS)*e, "frozen_model_checks": 4}


def planning_budget(*, preflight):
    e = options(preflight=preflight)["evaluation_episodes"]
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    planned = len(LOWERS)*3*e
    fits = planned*sum(h//p-1 for p in PERIODS)
    return {"plan_ols_fits": fits, "plan_ridge_predictions": fits,
        "reference_evaluations": len(PERIODS)*planned*h, "actor_context_evaluations": len(PERIODS)*planned*h}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "initialization": "all_Stage107_final_learned_and_forecast_lowers_no_training",
        "task": source.contract()["task"], "periods": list(PERIODS),
        "variants": {k: list(v) for k, v in VARIANTS.items()},
        "feedback": "all392_flat_feedback_features_unchanged_same396_actor398_critic",
        "interventions": "within_each_fixed_lower_swap_learned_forecast_zero_mean_same_std_noise_and_zero_advice",
        "noise_control": "zero_only_upper_final_mean_layer_same_std_decoder_alpha_and_paired_independent_innovations",
        "forecast": "causal_ridge_same_frozen_predictor_no_upper_calls",
        "blind": "four_advice_zeros_no_forecast_or_upper_calls",
        "freeze": "all_actor_weights_except_declared_zero_mean_execution_control_std_values_Adam_predictor_decoder_no_update",
        "evaluation": "all8_policies32_fresh_paired_native_paths_per_period_no_selection",
        "statistics": "all20_equal_root_bootstrap65536_Bonferroni20_seed108_108108_no_pooling",
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "decision": "all4_learned_lower_learned_plan_minus_forecast_and_noise_corrected_lower_CI_bounds_positive_pref_mechanical_only",
        "cost": "only_native_execution_extra_upper_and_forecast_costs_reported_inherited_training_separate",
        "artifacts": "compact_JSON_only_no_new_checkpoints_or_raw_traces",
        "limits": "fixed_donor_plan_content_diagnostic_not_new_HRL_training_frequency_superiority_or_from_scratch_generalization"}
