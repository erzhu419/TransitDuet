"""Local option credit for a new zero-initialized action residual."""
from scripts import pointmaze_optional_plan_stage107_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
roots, arguments, task_options = source.roots, source.arguments, source.task_options
EXPERIMENT_PROTOCOL = "pointmaze_option_residual_stage111_v1"
POLICY = "frozen_strong_flat_local_option_residual_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_option_residual_stage111.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_option_residual_stage111.py"
SOURCE_RUN = "pointmaze_optional_plan_stage107_full_20261004_r1"
METHODS = {}
ARMS, PANELS = ("blind", "forecast", "learned"), ("A", "B")
EPSILON = .05
VARIANTS = ("zero", "axis0_plus", "axis0_minus", "axis1_plus", "axis1_minus")
METRICS = ("local_credit_cosine", "local_credit_dot", "local_credit_rms")
ENDPOINTS = tuple(f"{p}/{m}" for p in PERIODS for m in METRICS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (111, 111111)


def options(*, preflight):
    return {"workers": 2 if preflight else 4, "queries": 2 if preflight else 12}


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def donor_checkpoint(root, period):
    return source_result(root).parent/"final_weights"/f"period_{period}_blind.pt"


def source_record(root):
    return {"donor_protocol": source.EXPERIMENT_PROTOCOL, "donor_result": str(source_result(root)),
        "donors": {str(p): str(donor_checkpoint(root, p)) for p in PERIODS},
        "selection": "all8_final_blind_donors_update8_no_selection"}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 111000000 if preflight else 111100000 + roots(preflight=False).index(root)*100000
    h = arguments(root, preflight=preflight).horizon
    starts = (100,) if preflight else (h//4, h//2, 3*h//4)
    return {"queries": [{"scenario_seed": base+95001+i, "prefix_noise_seed": base+1001+i,
        "suffix_noise_seeds": {name: base+10001+1000*j+i for j, name in enumerate(PANELS)},
        "start": starts[i%len(starts)]} for i in range(options(preflight=preflight)["queries"])]}


def budget(*, preflight):
    q = len(PERIODS)*options(preflight=preflight)["queries"]
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n = q*len(PANELS)*len(VARIANTS)
    return {"native_episodes": n, "native_steps": n*h, "native_lower_calls": n*h,
        "native_pair_groups": q, "network_freeze_checks": q, "prefix_pair_checks": q*(len(PANELS)*len(VARIANTS)-1),
        "exogenous_pair_checks": q*(len(PANELS)*len(VARIANTS)-1),
        "innovation_pair_checks": q*len(PANELS)*(len(VARIANTS)-1), "suffix_credit_identity_checks": q*len(PANELS)*(len(VARIANTS)-1),
        "branch_zero_checks": q*len(ARMS), "advice_upper_calls": q,
        "advice_ols_fits": 2*q, "advice_ridge_predictions": 2*q,
        "advice_reference_calls": 2*q, "advice_context_calls": 2*q, "source_freeze_checks": len(PERIODS)}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "donors": "all8_final_Stage107_blind_lowers_fixed_Stage106_upper",
        "task": source.contract()["task"], "periods": list(PERIODS), "arms": list(ARMS),
        "architecture": "frozen396_base_with_zero_advice_plus_trainable_zero_linear396_to2_readout_same_all_arms_std_fixed",
        "initial_policy": "exact_strong_flat_mean_and_std_for_arbitrary_advice_no_extra_exploration",
        "intervention": "readout_bias_axis_plus_minus0.05_raw_Gaussian_mean_for_one_option_only_then_zero_to_episode_end",
        "query_states": "fresh12_scenarios_per_root_period_balanced_start300_600_900_pref2_start100_replayed_prefix",
        "noise": "same_prefix_and_state_A_B_independent_suffix_noise_common_across5_interventions_within_each_panel",
        "credit": "undiscounted_suffix_return_central_difference_two_action_bias_axes_no_critic_or_option_truncation",
        "advice": "same_query392_feedback_plus4_blind_forecast_or_deterministic_upper_full_alpha1_advice_zero_branch_identity_only",
        "statistics": "six_equal_root_endpoints_bootstrap65536_Bonferroni6_seed111_111111",
        "gate": "both_periods_positive_corrected_CI_for_local_credit_cosine_and_dot_RMS_descriptive",
        "admission": "pref_mechanical_only_frozen_full_before_pref_no_tuning_on_pref",
        "artifacts": "server_scalar_queries_only_no_native_trace_checkpoint_or_training_writes_compact_pull",
        "limits": "conditional_local_bias_credit_replication_not_policy_gain_advice_or_hierarchy_superiority_Stage67_HOLD_unchanged"}
