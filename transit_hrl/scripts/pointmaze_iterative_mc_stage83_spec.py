"""Preregistered iterative MC mean learning, joint versus lower-only."""

import math
from scripts import pointmaze_joint_mean_stage82_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
roots, arguments, source_result = source.roots, source.arguments, source.source_result
FISHER_RADIUS = source.FISHER_RADIUS
EXPERIMENT_PROTOCOL = "pointmaze_iterative_mc_stage83_v1"
POLICY = "iterative_mc_mean"
RUNNER_SCRIPT = "scripts/run_pointmaze_iterative_mc_stage83.py"
METHODS = {"joint_trained": {"upper": .5, "lower": .5}, "lower_trained": {"lower": 1.}}
VARIANTS = ("base", "zero", *METHODS)
CONTRAST_PAIRS = (("joint_trained", "lower_trained"), ("joint_trained", "base"), ("lower_trained", "base"),
    ("joint_trained", "zero"), ("lower_trained", "zero"), ("base", "zero"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (83, 83083)


def options(*, preflight):
    return {**source.options(preflight=preflight), "updates": 2 if preflight else 8}


def seed_roles(root, *, preflight):
    base = 83_000_000 if preflight else 83_100_000 + roots(preflight=False).index(root) * 100000
    o = options(preflight=preflight)
    rounds = [{"credit_" + name: [{"scenario_seed": base + 10000*j + offset + i,
        "noise_seeds": [base + 10000*j + offset + 2001 + 2*i, base + 10000*j + offset + 2002 + 2*i]}
        for i in range(o["credit_scenarios_per_batch"])] for name,offset in (("A",1),("B",1001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds, "native_evaluation": list(range(base+95001,base+95001+o["evaluation_episodes"]))}


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    k, n, e = o["updates"], 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"], o["evaluation_episodes"]
    credit = len(PERIODS)*len(METHODS)*k*n
    evaluation = len(PERIODS)*len(VARIANTS)*e
    count = credit + evaluation
    forward = fisher = 0
    for p in PERIODS:
        for allocation in METHODS.values():
            for actor in allocation:
                t = h//p if actor == "upper" else h
                forward += k*n*math.ceil(t/CHUNK_SIZE)
                fisher += k*math.ceil(n*t/CHUNK_SIZE)
    actor_updates = len(PERIODS)*k*sum(map(len,METHODS.values()))
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "decoder_loads": len(PERIODS),
        "training_models_initialized": len(PERIODS)*len(METHODS), "credit_episodes": credit, "evaluation_episodes": evaluation,
        "native_episodes": count, "native_steps": count*h, "native_lower_calls": count*h,
        "native_upper_calls": (len(METHODS)*k*n+len(VARIANTS)*e)*sum(h//p for p in PERIODS),
        "pairing_upper_forward_calls": (len(METHODS)*k*n+len(VARIANTS)*e)*sum(h//p for p in PERIODS),
        "native_network_checks": count, "native_pair_checks": len(PERIODS)*e,
        "scenario_pair_checks": credit//o["rollouts_per_scenario"], "objective_checks": credit, "mc_calls": 2*credit,
        "actor_score_forward_batches": forward, "actor_score_backward_batches": 3*forward,
        "fisher_jvp_batches": fisher, "exact_kl_forward_batches": 2*fisher,
        "actor_parameter_perturbations": 2*actor_updates, "parameter_part_checks": actor_updates,
        "actor_mean_parameter_updates": actor_updates, "policy_updates": len(PERIODS)*k*len(METHODS),
        "training_freeze_checks": len(PERIODS)*k*len(METHODS), "frozen_model_checks": len(PERIODS),
        "checkpoint_writes": 0 if preflight else len(PERIODS)*len(METHODS)}


def contract():
    return {"source": source.contract()["source"], "decoder": source.contract()["decoder"],
        "periods": list(PERIODS), "variants": list(VARIANTS), "methods": METHODS,
        "training": "eight_fresh_on_policy_MC_mean_updates_no_PPO_Adam_entropy_or_reward_normalization_preflight_two",
        "credit": source.contract()["credit"], "baseline": source.contract()["baseline"],
        "sampling": "each_round_two_disjoint_batches_16_scenarios_two_independent_noise_replicates_same_exogenous_roster_for_both_methods",
        "noise_mapping": "Stage80_explicit_mapping_fresh_disjoint_Stage83_round_and_evaluation_roles",
        "budget": "same_environment_samples_and_sum_of_level_source_state_conditional_KL_.001_per_round_joint_half_each_lower_only_full",
        "freeze": "both_std_values_source_Adam_forecaster_decoder_unchanged_lower_only_upper_actor_bit_exact_frozen",
        "evaluation": "only_after_registered_last_update_no_best_checkpoint_early_stop_or_radius_allocation_search",
        "statistics": "all12_final_reward_contrasts_equal_root_bootstrap65536_Bonferroni12",
        "artifacts": "server_final_inference_weights_only_no_intermediate_checkpoints_or_raw_traces_local_compact_JSON_and_completion_only",
        "decision": "independent_MC_learning_route_Stage67_critic_credit_HOLD_unchanged_no_source_policy_adoption",
        "limits": "teacher_initialized_fixed_std_decoder_MC_training_not_full_actor_critic_trajectory_KL_or_frequency_superiority"}
