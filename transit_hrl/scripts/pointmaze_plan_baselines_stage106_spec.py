"""Matched-sample learned-upper, forecast-only and genuine flat controls."""

import math
from scripts import pointmaze_conditioning_shift_stage105_spec as source

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, options, arguments, source_record, task_options = source.roots, source.options, source.arguments, source.source_record, source.task_options
METHODS = {**source.METHODS, "forecast_lower": ("lower",), "flat_lower": ("lower",)}
VARIANTS = ("joint_base", "forecast_base", "flat_base", *METHODS)
CONTRAST_PAIRS = (("joint_conditioned", "forecast_lower"), ("joint_conditioned", "flat_lower"),
    ("joint_conditioned", "joint_independent"), ("joint_independent", "forecast_lower"),
    ("joint_independent", "flat_lower"), ("forecast_lower", "flat_lower"),
    ("joint_conditioned", "joint_base"), ("joint_independent", "joint_base"),
    ("forecast_lower", "forecast_base"), ("flat_lower", "flat_base"),
    ("joint_base", "forecast_base"), ("joint_base", "flat_base"), ("forecast_base", "flat_base"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS[:2])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (106, 106106)
EXPERIMENT_PROTOCOL = "pointmaze_plan_baselines_stage106_v1"
POLICY = "matched_native_samples_joint_forecast_flat"
RUNNER_SCRIPT = "scripts/run_pointmaze_plan_baselines_stage106.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_plan_baselines_stage106.py"


def allocation(method, period):
    return source.allocation(method, period) if method.startswith("joint_") else {"lower": 1.}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 106_000_000 if preflight else 106_100_000 + roots(preflight=False).index(root)*100000
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
    k, n, e = o["updates"], 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"], o["evaluation_episodes"]
    credit, evaluation = len(PERIODS)*4*k*2*n, len(PERIODS)*len(VARIANTS)*e
    actor_updates = len(PERIODS)*k*6
    forward = sum(k*(2*n*math.ceil((h//p)/CHUNK_SIZE)+6*n*math.ceil(h/CHUNK_SIZE)) for p in PERIODS)
    fisher = sum(k*(2*math.ceil(n*(h//p)/CHUNK_SIZE)+2*math.ceil(n*h/CHUNK_SIZE)
        +2*math.ceil(2*n*h/CHUNK_SIZE)) for p in PERIODS)
    upper_calls = (4*k*n+3*e)*sum(h//p for p in PERIODS)
    pair_checks = len(PERIODS)*k*2*o["credit_scenarios_per_batch"]
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "decoder_loads": len(PERIODS),
        "training_models_initialized": len(PERIODS)*4, "credit_episodes": credit, "evaluation_episodes": evaluation,
        "native_episodes": credit+evaluation, "native_steps": (credit+evaluation)*h,
        "native_lower_calls": (credit+evaluation)*h, "native_upper_calls": upper_calls,
        "pairing_upper_forward_calls": upper_calls, "native_network_checks": credit+evaluation,
        "native_pair_checks": len(PERIODS)*e, "scenario_pair_checks": credit//o["rollouts_per_scenario"],
        "objective_checks": credit, "mc_calls": credit*3//2,
        "actor_score_forward_batches": forward, "actor_score_backward_batches": 3*forward,
        "fisher_jvp_batches": fisher, "exact_kl_forward_batches": 2*fisher,
        "actor_parameter_perturbations": 2*actor_updates, "parameter_part_checks": actor_updates,
        "actor_mean_parameter_updates": actor_updates, "policy_updates": len(PERIODS)*k*4,
        "training_freeze_checks": len(PERIODS)*k*4, "frozen_model_checks": len(PERIODS),
        "checkpoint_writes": 0 if preflight else len(PERIODS)*4,
        "upper_independent_pair_checks": 2*pair_checks, "lower_independent_pair_checks": pair_checks,
        "lower_common_pair_checks": pair_checks, "primitive_pair_checks": 4*pair_checks,
        "upper_replay_forward_calls": k*2*o["credit_scenarios_per_batch"]*sum(h//p for p in PERIODS)}


def planning_budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
    planned = 6*o["updates"]*n+5*o["evaluation_episodes"]
    return {"plan_ols_fits": planned*sum(h//p-1 for p in PERIODS),
        "plan_ridge_predictions": planned*sum(h//p-1 for p in PERIODS),
        "reference_evaluations": len(PERIODS)*planned*h, "actor_context_evaluations": len(PERIODS)*planned*h}


def contract():
    inherited = source.contract()
    return {k: inherited[k] for k in ("source", "decoder", "credit", "artifacts", "task_change")} | {
        "periods": list(PERIODS), "methods": {str(p): {m: allocation(m, p) for m in METHODS} for p in PERIODS},
        "variants": list(VARIANTS), "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "source_cohort": "original_eight_Stage96_teachers_Stage97_decoder_no_trained_donor_or_refit",
        "training": "eight_mean_updates_two_preflight_joint64_upper_plus64_lower_baselines128_lower_paths_per_round",
        "baseline_sampling": "union_of_registered_upper_and_lower_credit_rosters_same_total_native_samples",
        "budget": "nominal_per_primitive_call_weighted_KL_.001_each_update_baselines_all_to_lower_joint_as_Stage105_actual_compute_reported",
        "forecast_lower": "causal_ridge_reference_and_velocity_only_no_upper_actor_inference_or_learned_residual",
        "flat_lower": "physical_current_target_error_full64step_history_causal_latest_target_velocity_no_plan_forecast_or_upper_inference",
        "initialization": "identical_original_lower392_weights_all_methods_three_own_input_semantic_base_controls_no_baseline_refit",
        "flat_period": "teacher_initialization_index_not_flat_action_or_plan_frequency",
        "conditioning": "only_joint_conditioned_lower_training_common_upper_innovations_independent_lower_noise",
        "freeze": "std_values_Adam_source_models_forecaster_decoder_fixed_baseline_upper_bit_exact_unused",
        "evaluation": "seven_fixed_final_or_initial_policies_fresh_paired_independent_noise_no_selection",
        "statistics": "all26_equal_root_bootstrap65536_Bonferroni26_seed106_106106_no_stage_pooling",
        "decision": "all_four_conditioned_minus_trained_forecast_and_flat_lower_CI_bounds_positive_pref_mechanical_only",
        "limits": "teacher_assisted_MC_mean_learning_not_from_scratch_flat_PPO_SAC_equal_parameter_count_full_actor_critic_or_frequency_superiority"}
