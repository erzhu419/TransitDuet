"""Matched-sample, matched-cumulative-KL native training-order comparison."""

import math
from scripts import pointmaze_iterative_mc_stage83_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
roots, arguments, source_result = source.roots, source.arguments, source.source_result
FISHER_RADIUS = source.FISHER_RADIUS
EXPERIMENT_PROTOCOL = "pointmaze_training_order_stage85_v1"
POLICY = "matched_training_order"
RUNNER_SCRIPT = "scripts/run_pointmaze_training_order_stage85.py"
METHODS = ("paired_joint", "alternating", "staged", "lower_trained")
VARIANTS = ("base", "zero", *METHODS)
CONTRAST_PAIRS = (("alternating","paired_joint"),("staged","paired_joint"),("staged","alternating"),
    *((m,"lower_trained") for m in METHODS if m != "lower_trained"),
    *((m,"zero") for m in METHODS),("base","zero"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (85, 85085)


def options(*, preflight):
    return {"workers":2 if preflight else 8,"credit_chunks":4 if preflight else 16,
        "credit_scenarios_per_batch":2 if preflight else 8,"rollouts_per_scenario":2,
        "evaluation_episodes":4 if preflight else 32}


def training_groups(method, *, preflight):
    n = options(preflight=preflight)["credit_chunks"]
    if method not in METHODS:raise ValueError("Unregistered Stage85 method")
    if method in ("alternating","staged"):
        return [{"chunks":[c],"updates":[{"actor":"lower" if (c%2 == 0 if method == "alternating" else c < n//2)
            else "upper","fraction":.5,"chunks":[c]}]} for c in range(n)]
    return [{"chunks":[c,c+1],"updates":([
        {"actor":"lower","fraction":.5,"chunks":[c]},
        {"actor":"upper","fraction":.5,"chunks":[c+1]}] if method == "paired_joint" else [
        {"actor":"lower","fraction":1.,"chunks":[c,c+1]}])} for c in range(0,n,2)]


def seed_roles(root, *, preflight):
    base = 85_000_000 if preflight else 85_100_000+roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    chunks = [{"credit_"+b:[{"scenario_seed":base+5000*c+offset+i,
        "noise_seeds":[base+5000*c+offset+2001+2*i,base+5000*c+offset+2002+2*i]}
        for i in range(o["credit_scenarios_per_batch"])] for b,offset in (("A",1),("B",1001))}
        for c in range(o["credit_chunks"])]
    return {"training_chunks":chunks,"native_evaluation":list(range(base+95001,base+95001+o["evaluation_episodes"]))}


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0],preflight=preflight).horizon
    n,k,e = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"],o["credit_chunks"],o["evaluation_episodes"]
    credit,evaluation = len(PERIODS)*len(METHODS)*k*n,len(PERIODS)*len(VARIANTS)*e
    forward = fisher = updates = collection = 0
    for p in PERIODS:
        for method in METHODS:
            groups = training_groups(method,preflight=preflight)
            collection += len(groups)
            for group in groups:
                for update in group["updates"]:
                    t = h//p if update["actor"] == "upper" else h
                    samples = n*len(update["chunks"])
                    forward += samples*math.ceil(t/CHUNK_SIZE)
                    fisher += math.ceil(samples*t/CHUNK_SIZE)
                    updates += 1
    count = credit+evaluation
    return {"source_clone_loads":len(PERIODS),"forecaster_loads":1,"decoder_loads":len(PERIODS),
        "training_models_initialized":len(PERIODS)*len(METHODS),"credit_episodes":credit,"evaluation_episodes":evaluation,
        "native_episodes":count,"native_steps":count*h,"native_lower_calls":count*h,
        "native_upper_calls":(len(METHODS)*k*n+len(VARIANTS)*e)*sum(h//p for p in PERIODS),
        "pairing_upper_forward_calls":(len(METHODS)*k*n+len(VARIANTS)*e)*sum(h//p for p in PERIODS),
        "native_network_checks":count,"native_pair_checks":len(PERIODS)*e,"scenario_pair_checks":credit//2,
        "objective_checks":credit,"mc_calls":2*credit,"actor_score_forward_batches":forward,
        "actor_score_backward_batches":3*forward,"fisher_jvp_batches":fisher,"exact_kl_forward_batches":2*fisher,
        "actor_parameter_perturbations":2*updates,"parameter_part_checks":updates,"actor_mean_parameter_updates":updates,
        "policy_updates":updates,"training_freeze_checks":updates,"collection_freeze_checks":collection,
        "frozen_model_checks":len(PERIODS),"checkpoint_writes":0 if preflight else len(PERIODS)*len(METHODS)}


def contract():
    return {"source":source.contract()["source"],"decoder":source.contract()["decoder"],"periods":list(PERIODS),
        "variants":list(VARIANTS),"methods":list(METHODS),"credit":source.contract()["credit"],"baseline":source.contract()["baseline"],
        "training":"16_fresh_credit_chunks_32_episodes_each_no_replay_PPO_Adam_entropy_or_reward_normalization_preflight_four_chunks_eight_episodes",
        "order":"paired_joint_collect_two_chunks_before_both_updates_alternating_lower_upper_staged_lower_first_half_then_upper_lower_only_pool_two_chunks",
        "matching":"dual_methods_eight_updates_per_level_32_episodes_per_gradient_.0005_KL_each_lower_only_eight_64_episode_lower_updates_.001_each_total512_episodes_.008_nominal_KL_per_method_period",
        "paired_joint":"independent_half_batch_per_level_same_pre_update_joint_policy_not_Stage83_shared_full_gradient_batch",
        "freeze":"std_values_source_Adam_forecaster_decoder_bit_exact_frozen_lower_only_upper_frozen",
        "noise_mapping":"Stage80_explicit_mapping_fresh_disjoint_Stage85_chunk_and_evaluation_roles_same_rosters_across_methods",
        "evaluation":"registered_last_update_only_32_fresh_paired_seeds_per_root_period_no_intermediate_selection_preflight_four",
        "statistics":"all22_final_reward_contrasts_equal_root_bootstrap65536_Bonferroni22",
        "artifacts":"server_final_inference_weights_only_no_intermediate_checkpoints_traces_local_compact_JSON_and_completion_only",
        "decision":"test_order_contrasts_and_fair_budget_lower_only_comparator_Stage67_critic_credit_HOLD_unchanged",
        "limits":"teacher_initialized_fixed_std_decoder_MC_route_nominal_sum_of_update_state_conditional_KL_not_trajectory_KL_path_length_or_flat_RL_frequency_superiority"}
