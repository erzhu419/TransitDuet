"""Matched branch-only learning above the qualified Stage111 donor."""
import math

from scripts import pointmaze_option_residual_stage111_spec as source

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, 1024, .001
roots, arguments, task_options = source.roots, source.arguments, source.task_options
EXPERIMENT_PROTOCOL = "pointmaze_option_residual_train_stage112_v1"
POLICY = "matched_branch_only_mc_mean_update_frozen_flat_base"
RUNNER_SCRIPT = "scripts/run_pointmaze_option_residual_train_stage112.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_option_residual_train_stage112.py"
SOURCE_RUN = "pointmaze_option_residual_stage111_full_20261004_r1"
ARMS = ("blind", "forecast", "learned")
METHODS = {arm: ("lower_residual",) for arm in ARMS}
VARIANTS = ("base", *ARMS, "forecast_blinded", "learned_blinded")
CONTRAST_PAIRS = (("learned", "blind"), ("learned", "forecast"), ("learned", "learned_blinded"),
    ("forecast", "blind"), ("forecast_blinded", "blind"), ("blind", "base"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS[:3])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (112, 112112)


def options(*, preflight):
    return {"workers": 2 if preflight else 4, "credit_scenarios_per_batch": 2 if preflight else 16,
        "rollouts_per_scenario": 2, "evaluation_episodes": 4 if preflight else 32, "updates": 2 if preflight else 8}


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def donor_checkpoint(root, period):
    return source_result(root).parent/"final_weights"/f"period_{period}_blind.pt"


def source_record(root):
    return {"protocol": source.EXPERIMENT_PROTOCOL, "result": str(source_result(root)),
        "donors": {str(p): str(donor_checkpoint(root, p)) for p in PERIODS},
        "selection": "all8_final_Stage111_blind_donors_no_selection"}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 112000000 if preflight else 112100000 + roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    rounds = [{name: [{"scenario_seed": base+10000*j+offset+i,
        "noise_seeds": [base+10000*j+offset+2001+2*i, base+10000*j+offset+2002+2*i]}
        for i in range(o["credit_scenarios_per_batch"])]
        for name, offset in (("credit_A", 1), ("credit_B", 1001), ("lower_credit_A", 4001), ("lower_credit_B", 5001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds, "native_evaluation": list(range(base+95001, base+95001+o["evaluation_episodes"]))}


def budget(*, preflight):
    o, h = options(preflight=preflight), arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n = 4*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
    credit = len(PERIODS)*len(ARMS)*o["updates"]*n
    evaluation = len(PERIODS)*len(VARIANTS)*o["evaluation_episodes"]
    episodes = credit+evaluation
    training_plan_episodes = o["updates"]*n
    evaluation_plan_episodes = o["evaluation_episodes"]
    score_batches = 2*math.ceil((n//2)*h/CHUNK_SIZE)
    planned_episodes = 2*(training_plan_episodes+evaluation_plan_episodes)
    plan_renewals = (training_plan_episodes+evaluation_plan_episodes)*sum(h//p for p in PERIODS)*2
    plan_fits = (training_plan_episodes+evaluation_plan_episodes)*sum(h//p-1 for p in PERIODS)*2
    plan_calls = (training_plan_episodes+evaluation_plan_episodes)*sum(h for _ in PERIODS)*2
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "decoder_loads": len(PERIODS),
        "source_cell_loads": 1, "donor_checkpoint_loads": len(PERIODS), "branch_initializations": len(PERIODS)*len(ARMS),
        "training_episodes": credit, "evaluation_episodes": evaluation, "native_episodes": episodes,
        "native_steps": episodes*h, "native_lower_calls": episodes*h,
        "native_upper_calls": training_plan_episodes*sum(h//p for p in PERIODS)+evaluation_plan_episodes*sum(h//p for p in PERIODS),
        "native_network_checks": episodes, "scenario_pair_checks": credit//o["rollouts_per_scenario"],
        "native_pair_checks": len(PERIODS)*o["evaluation_episodes"], "objective_checks": credit,
        "mc_calls": credit, "actor_score_forward_batches": score_batches*len(ARMS)*len(PERIODS)*o["updates"],
        "actor_score_backward_batches": score_batches*2*len(ARMS)*len(PERIODS)*o["updates"],
        "residual_fisher_batches": score_batches*len(ARMS)*len(PERIODS)*o["updates"],
        "residual_kl_checks": len(ARMS)*len(PERIODS)*o["updates"],
        "residual_parameter_updates": len(ARMS)*len(PERIODS)*o["updates"],
        "training_freeze_checks": len(ARMS)*len(PERIODS)*o["updates"],
        "frozen_model_checks": len(PERIODS), "checkpoint_writes": 0 if preflight else len(PERIODS)*len(ARMS),
        "planning_renewals": plan_renewals, "planning_fits": plan_fits, "planning_predictions": plan_fits,
        "planning_reference_calls": plan_calls, "planning_context_calls": plan_calls}


def contract():
    return {"source": source_record(roots(preflight=False)[0])["protocol"],
        "task": source.contract()["task"], "periods": list(PERIODS), "arms": list(ARMS), "variants": list(VARIANTS),
        "architecture": "OptionalActionResidual396_to2_zero_initialized_frozen_Stage107_blind_lower_donor",
        "training": "eight_mean_only_MC_updates_each_arm_128_native_paths_update_frozen_upper_value_std_and_base",
        "credit": "scenario_leave_other_out_exact_lower_returns_no_fitted_critic",
        "upper": "fixed_deterministic_Stage106_upper_for_learned_plan_no_upper_update",
        "evaluation": "six_final_or_initial_policies32_fresh_paired_paths_with_forecast_and_learned_advice_blinded",
        "statistics": "all12_equal_root_bootstrap65536_Bonferroni12_seed112_112112_no_selection",
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "decision": "learned_minus_blind_forecast_and_own_blinded_positive_corrected_CI_required_before_upper_learning",
        "freeze": "Stage111_blind_donor_base_upper_std_values_Adam_and_critic_fixed",
        "artifacts": "server_final_branch_weights_only_no_intermediate_checkpoint_or_native_trace_pull_compact_JSON",
        "limits": "branch_only_teacher_initialized_MC_mean_training_not_full_actor_critic_or_end_to_end_HRL"}
