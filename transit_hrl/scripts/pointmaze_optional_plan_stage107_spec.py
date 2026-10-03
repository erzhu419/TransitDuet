"""Optional planning above the strong flat interface, with fixed upper donors."""

import math
from scripts import pointmaze_plan_baselines_stage106_spec as source

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, options, arguments, task_options = source.roots, source.options, source.arguments, source.task_options
SOURCE_RUN = "pointmaze_plan_baselines_stage106_full_20261003_r2"
METHODS = {m: ("lower",) for m in ("blind", "forecast_hint", "learned_hint")}
VARIANTS = ("base", *METHODS, "forecast_blinded", "learned_blinded")
CONTRAST_PAIRS = (("learned_hint", "blind"), ("learned_hint", "forecast_hint"),
    ("learned_hint", "learned_blinded"), ("forecast_hint", "blind"),
    ("blind", "base"), ("forecast_hint", "base"), ("learned_hint", "base"),
    ("forecast_hint", "forecast_blinded"), ("forecast_blinded", "blind"), ("learned_blinded", "blind"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS[:3])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (107, 107107)
EXPERIMENT_PROTOCOL = "pointmaze_optional_plan_stage107_v1"
POLICY = "strong_flat_optional_fixed_plan_hint"
RUNNER_SCRIPT = "scripts/run_pointmaze_optional_plan_stage107.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_optional_plan_stage107.py"


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def donor_checkpoint(root, period, method):
    return source_result(root).parent/"final_weights"/f"period_{period}_{method}.pt"


def source_record(root):
    return {"teacher_decoder": source.source_record(root), "donor_protocol": source.EXPERIMENT_PROTOCOL,
        "donor_result": str(source_result(root)), "lower_method": "flat_lower", "upper_method": "joint_conditioned",
        "donors": {str(p): {m: str(donor_checkpoint(root, p, m)) for m in ("flat_lower", "joint_conditioned")} for p in PERIODS},
        "selection": "all_eight_roots_both_periods_fixed_final_update8_no_donor_selection"}


def allocation(method, period):
    return {"lower": 1.}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 107_000_000 if preflight else 107_100_000 + roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    rounds = [{name: [{"scenario_seed": base+10000*j+offset+i,
        "noise_seeds": [base+10000*j+offset+2001+2*i, base+10000*j+offset+2002+2*i]}
        for i in range(o["credit_scenarios_per_batch"])]
        for name, offset in (("credit_A", 1), ("credit_B", 1001), ("lower_credit_A", 4001), ("lower_credit_B", 5001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds, "native_evaluation": list(range(base+95001, base+95001+o["evaluation_episodes"]))}


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    k, n, e = o["updates"], 4*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"], o["evaluation_episodes"]
    credit, evaluation = len(PERIODS)*len(METHODS)*k*n, len(PERIODS)*len(VARIANTS)*e
    updates = len(PERIODS)*len(METHODS)*k
    forward, fisher = credit*math.ceil(h/CHUNK_SIZE), updates*math.ceil(n*h/CHUNK_SIZE)
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "decoder_loads": len(PERIODS),
        "source_cell_loads": 1, "donor_checkpoint_loads": 2*len(PERIODS), "expanded_model_initializations": len(PERIODS),
        "training_models_initialized": len(PERIODS)*len(METHODS), "credit_episodes": credit, "evaluation_episodes": evaluation,
        "native_episodes": credit+evaluation, "native_steps": (credit+evaluation)*h, "native_lower_calls": (credit+evaluation)*h,
        "native_upper_calls": (k*n+e)*sum(h//p for p in PERIODS), "native_network_checks": credit+evaluation,
        "native_pair_checks": len(PERIODS)*e, "scenario_pair_checks": credit//o["rollouts_per_scenario"],
        "objective_checks": credit, "mc_calls": credit, "actor_score_forward_batches": forward,
        "actor_score_backward_batches": 3*forward, "fisher_jvp_batches": fisher, "exact_kl_forward_batches": 2*fisher,
        "actor_parameter_perturbations": 2*updates, "parameter_part_checks": updates, "actor_mean_parameter_updates": updates,
        "policy_updates": updates, "training_freeze_checks": updates, "frozen_model_checks": len(PERIODS),
        "checkpoint_writes": 0 if preflight else len(PERIODS)*len(METHODS)}


def planning_budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    count = 2*(o["updates"]*4*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]+o["evaluation_episodes"])
    return {"plan_ols_fits": count*sum(h//p-1 for p in PERIODS), "plan_ridge_predictions": count*sum(h//p-1 for p in PERIODS),
        "reference_evaluations": len(PERIODS)*count*h, "actor_context_evaluations": len(PERIODS)*count*h}


def contract():
    return {"source": source_record(roots(preflight=False)[0])["donor_protocol"],
        "source_initialization": "all_Stage106_final_flat_lowers_and_conditioned_joint_uppers_original_forecaster_decoder_no_refit",
        "task": source.contract()["task_change"], "periods": list(PERIODS), "methods": list(METHODS), "variants": list(VARIANTS),
        "actor_inputs": "first392_exact_flat_feedback_plus4_advice_reference_minus_current_target_and_planned_minus_causal_velocity",
        "architecture": "lower396_value398_zero_padded_first_layers_same_all_arms_no_replacement_of_current_target_error",
        "initial_policy": "all_advice_columns_zero_so_initial_lower_ignores_arbitrary_hints_same_trained_flat_function",
        "training": "eight_lower_MC_mean_updates_preflight_two_128_native_paths_per_update_preflight16_shared_rosters",
        "upper": "fixed_Stage106_joint_conditioned_mean_std_no_upper_updates_original_independent_stochastic_sampling",
        "freeze": "std_values_Adam_forecaster_decoder_donors_and_upper_unchanged_no_critic_fit",
        "budget": "matched_lower_samples_mean_updates_nominal_KL_.001_per_update_plan_and_upper_inference_costs_extra_reported",
        "evaluation": "six_final_or_initial_policies32_fresh_independent_paired_paths_no_intermediate_selection",
        "execution_ablation": "forecast_blinded_learned_blinded_keep_trained_lower_but_zero_advice_and_no_plan_or_upper_calls",
        "statistics": "all20_equal_root_bootstrap65536_Bonferroni20_seed107_107107_no_pooling",
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "decision": "all_six_learned_hint_minus_blind_forecast_and_own_blinded_primary_lower_CI_bounds_positive_pref_mechanical_only",
        "artifacts": "server_final_weights_only_no_intermediate_checkpoint_raw_trace_or_local_native_training_compact_JSON_pull",
        "limits": "conditional_fixed_upper_hint_learning_above_reused_strong_flat_not_joint_HRL_unseen_tasks_or_frequency_superiority"}
