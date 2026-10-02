"""Staged upper-only learning over both frozen Stage93 U0-trained lowers."""

from scripts import pointmaze_upper_noise_replication_stage93_spec as source
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
EXPERIMENT_PROTOCOL = "pointmaze_staged_upper_stage94_v1"
POLICY = "staged_upper_independent_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_staged_upper_stage94.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_staged_upper_stage94.py"
LOWER_TRAINING_RUN = "pointmaze_upper_noise_replication_stage93_full_20261002_r1"
CHECKPOINT_METHODS = ("source_upper_independent", "source_upper_common", "joint_call")
LOWER_FOR_METHOD = {"staged_independent":"source_upper_independent", "staged_common":"source_upper_common"}
METHODS = {m:("upper",) for m in LOWER_FOR_METHOD}
COMPOSITIONS = {
    "base":("source","source"), "zero":("source","source"),
    "source_independent":("source","source_upper_independent"),
    "source_common":("source","source_upper_common"),
    "fixed_joint_independent":("joint_call","source_upper_independent"),
    "fixed_joint_common":("joint_call","source_upper_common"),
    "staged_independent":("staged_independent","source_upper_independent"),
    "staged_common":("staged_common","source_upper_common"),
    "common_upper_independent_lower":("staged_common","source_upper_independent"),
    "independent_upper_common_lower":("staged_independent","source_upper_common"),
    "joint":("joint_call","joint_call"),
}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (
    ("staged_common","source_common"), ("staged_common","staged_independent"),
    ("staged_independent","source_independent"), ("source_common","source_independent"),
    ("staged_common","fixed_joint_common"), ("staged_independent","fixed_joint_independent"),
    ("fixed_joint_common","source_common"), ("fixed_joint_independent","source_independent"),
    ("staged_common","joint"), ("staged_independent","joint"),
    ("common_upper_independent_lower","staged_independent"),
    ("independent_upper_common_lower","staged_independent"),
    ("staged_common","independent_upper_common_lower"), ("base","zero"),
)
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS[:2])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (94, 94094)


def donor_result(root, method):
    if method in LOWER_FOR_METHOD.values():
        return ROOT/"results"/LOWER_TRAINING_RUN/"cells"/f"replicate_{root}"/"result.json"
    if method == "joint_call":return source.donor_result(root,method)
    raise ValueError("Stage94 donor is not registered")


def allocation(method, period):
    if method not in METHODS:raise ValueError("Stage94 trains only the two registered upper means")
    return {"upper":.5}


def seed_roles(root, *, preflight):
    roles = source.seed_roles(root,preflight=preflight)
    for r in roles["training_rounds"]:
        for b in ("A","B"):
            for s in r["credit_"+b]:
                s["scenario_seed"] += 1_000_000
                s["noise_seeds"] = [n+1_000_000 for n in s["noise_seeds"]]
    roles["native_evaluation"] = [s+1_000_000 for s in roles["native_evaluation"]]
    return roles


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0],preflight=preflight).horizon
    return {**training_budget(options(preflight=preflight),METHODS,VARIANTS,horizon=h,preflight=preflight),
        "checkpoint_loads":len(PERIODS)*len(CHECKPOINT_METHODS),
        "checkpoint_freeze_checks":len(PERIODS)*len(CHECKPOINT_METHODS),
        "training_initialization_checks":len(PERIODS)*len(METHODS),
        "actor_composition_checks":len(PERIODS)*len(VARIANTS), "upper_replay_forward_calls":0}


def contract():
    common = source.contract()
    return {**{k:common[k] for k in ("source","decoder","credit","baseline","evaluation","artifacts")},
        "periods":list(PERIODS), "variants":list(VARIANTS),
        "methods":{str(p):{m:allocation(m,p) for m in METHODS} for p in PERIODS},
        "training_runs":{**{m:LOWER_TRAINING_RUN for m in LOWER_FOR_METHOD.values()},
            "joint_call":common["training_runs"]["joint_call"]},
        "checkpoint_protocols":{**{m:source.EXPERIMENT_PROTOCOL for m in LOWER_FOR_METHOD.values()},
            "joint_call":common["checkpoint_protocols"]["joint_call"]},
        "checkpoint_update":8, "checkpoint_methods":list(CHECKPOINT_METHODS),
        "compositions_upper_lower":{v:list(pair) for v,pair in COMPOSITIONS.items()},
        "training":"U0_initialized_upper_means_with_corresponding_final_Stage93_U0_trained_lower_frozen_eight_updates_preflight_two",
        "sampling":"fresh_Stage94_rosters_same_scenarios_and_independent_upper_lower_noise_pairs_for_both_learners_64_episodes_per_update",
        "noise_mapping":"original_Stage80_noise_mapping_no_upper_replay_in_training_or_evaluation",
        "lower_for_method":LOWER_FOR_METHOD,
        "budget":"new_upper_conditional_KL_.0005_per_update_no_search_combined_nominal_call_weighted_lower_then_upper_proxy_.008_per_branch_not_equal_total_compute_or_trajectory_KL",
        "freeze":"corresponding_Stage93_lower_bit_exact_std_values_source_Adam_forecaster_decoder_original_teacher_unchanged",
        "statistics":"all28_reward_contrasts_equal_root_bootstrap65536_Bonferroni28_seed94_94094",
        "primary_endpoints":list(PRIMARY_ENDPOINTS),
        "decision":"all_four_primary_lower_CI_bounds_positive_required_preflight_mechanical_only_no_cross_stage_pooling_or_root_period_selection",
        "limits":"conditional_staged_mean_learning_same_eight_teachers_not_full_actor_critic_shared_upper_lower_credit_not_used_for_upper_Stage93_global_gate_failed_frequency_superiority_closed_Stage67_HOLD"}
