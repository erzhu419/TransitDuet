"""Repair staged actor-dataset assignment without retraining frozen baselines."""

import math
from scripts import pointmaze_training_order_stage85_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
FISHER_RADIUS = source.FISHER_RADIUS
EXPERIMENT_PROTOCOL = "pointmaze_aligned_order_stage86_v1"
POLICY = "aligned_staged_order"
RUNNER_SCRIPT = "scripts/run_pointmaze_aligned_order_stage86.py"
TRAINING_RUN = "pointmaze_training_order_stage85_full_20261002_r1"
BASELINES = {"paired_joint":"paired_joint","alternating":"alternating","staged_original":"staged","lower_trained":"lower_trained"}
VARIANTS = ("base","zero",*BASELINES,"staged_aligned")
CONTRAST_PAIRS = (("staged_aligned","paired_joint"),("staged_aligned","alternating"),
    ("staged_aligned","staged_original"),("staged_aligned","lower_trained"),
    ("paired_joint","lower_trained"),("alternating","paired_joint"),
    ("staged_aligned","zero"),("lower_trained","zero"),("base","zero"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (86, 86086)


def training_result(root):
    return ROOT/"results"/TRAINING_RUN/"cells"/f"replicate_{root}"/"result.json"


def source_chunk_order(*, preflight):
    n = options(preflight=preflight)["credit_chunks"]
    return list(range(0,n,2))+list(range(1,n,2))


def seed_roles(root, *, preflight):
    original = source.seed_roles(root,preflight=False)["training_chunks"]
    o = options(preflight=preflight)
    ordered = [{b:original[c][b][:o["credit_scenarios_per_batch"]] for b in ("credit_A","credit_B")}
        for c in source_chunk_order(preflight=preflight)]
    base = 86_090_000 if preflight else 86_100_000+roots(preflight=False).index(root)*10000
    return {"training_chunks":ordered,"source_chunk_order":source_chunk_order(preflight=preflight),
        "native_evaluation":list(range(base+1,base+1+o["evaluation_episodes"]))}


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0],preflight=preflight).horizon
    k,n,e = o["credit_chunks"],2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"],o["evaluation_episodes"]
    credit,evaluation = len(PERIODS)*k*n,len(PERIODS)*len(VARIANTS)*e
    count = credit+evaluation
    forward = k//2*n*sum(math.ceil(h/CHUNK_SIZE)+math.ceil(h//p/CHUNK_SIZE) for p in PERIODS)
    fisher = k//2*sum(math.ceil(n*h/CHUNK_SIZE)+math.ceil(n*(h//p)/CHUNK_SIZE) for p in PERIODS)
    updates = len(PERIODS)*k
    return {"source_clone_loads":len(PERIODS),"forecaster_loads":1,"decoder_loads":len(PERIODS),
        "training_models_initialized":len(PERIODS),"baseline_checkpoint_loads":len(PERIODS)*len(BASELINES),
        "baseline_freeze_checks":len(PERIODS)*len(BASELINES),"credit_episodes":credit,"evaluation_episodes":evaluation,
        "native_episodes":count,"native_steps":count*h,"native_lower_calls":count*h,
        "native_upper_calls":(k*n+len(VARIANTS)*e)*sum(h//p for p in PERIODS),
        "pairing_upper_forward_calls":(k*n+len(VARIANTS)*e)*sum(h//p for p in PERIODS),
        "native_network_checks":count,"native_pair_checks":len(PERIODS)*e,"scenario_pair_checks":credit//2,
        "objective_checks":credit,"mc_calls":2*credit,"actor_score_forward_batches":forward,
        "actor_score_backward_batches":3*forward,"fisher_jvp_batches":fisher,"exact_kl_forward_batches":2*fisher,
        "actor_parameter_perturbations":2*updates,"parameter_part_checks":updates,"actor_mean_parameter_updates":updates,
        "policy_updates":updates,"training_freeze_checks":updates,"collection_freeze_checks":updates,
        "frozen_model_checks":len(PERIODS),"checkpoint_writes":0 if preflight else len(PERIODS)}


def contract():
    return {"source":source.contract()["source"],"decoder":source.contract()["decoder"],"periods":list(PERIODS),
        "variants":list(VARIANTS),"baseline_training_run":TRAINING_RUN,"baseline_methods":BASELINES,
        "repair":"staged_lower_uses_even_Stage85_chunks_upper_uses_odd_identical_level_rosters_to_paired_joint_and_alternating",
        "training":"only_staged_aligned_retrained_from_same_source_reuse_fixed_Stage85_training_scenarios_and_noise_no_new_selection_or_radius",
        "matching":source.contract()["matching"],"credit":source.contract()["credit"],"baseline":source.contract()["baseline"],
        "freeze":"Stage85_final_baselines_bit_exact_reloaded_both_std_values_source_Adam_forecaster_decoder_frozen",
        "evaluation":"seven_variants_on_32_fresh_paired_Stage86_seeds_per_root_period_preflight_four_no_intermediate_selection",
        "noise_mapping":"Stage80_explicit_mapping_original_Stage85_training_roles_fresh_disjoint_Stage86_evaluation_roles",
        "statistics":"all18_reward_contrasts_equal_root_bootstrap65536_Bonferroni18",
        "artifacts":"read_baseline_weights_server_only_save_only_staged_aligned_final_weights_no_intermediate_checkpoints_or_traces",
        "decision":"correct_real_staged_dataset_confound_keep_Stage85_negative_evidence_Stage67_critic_credit_HOLD_unchanged",
        "limits":"post_Stage85_design_repair_same_fixed_training_rosters_fresh_evaluation_not_new_independent_training_replication_or_frequency_superiority"}
