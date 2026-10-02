"""Lower-only conditional credit with shared upper innovations within pairs."""

from scripts import pointmaze_fixed_upper_lower_stage91_spec as source
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
EXPERIMENT_PROTOCOL = "pointmaze_upper_common_noise_stage92_v1"
POLICY = "lower_credit_upper_common_noise"
RUNNER_SCRIPT = "scripts/run_pointmaze_upper_common_noise_stage92.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_upper_common_noise_stage92.py"
FIXED_TRAINING_RUN = "pointmaze_fixed_upper_lower_stage91_full_20261002_r1"
CHECKPOINT_METHODS = ("joint_call", "lower_matched", "lower_fixed_upper")
METHODS = {"source_upper_common":("lower",), "joint_upper_common":("lower",)}
COMPOSITIONS = {
    "base":("source","source"), "zero":("source","source"),
    "source_matched":("source","lower_matched"), "source_fixed":("source","lower_fixed_upper"),
    "source_common":("source","source_upper_common"), "source_joint_common":("source","joint_upper_common"),
    "source_joint":("source","joint_call"),
    "joint_matched":("joint_call","lower_matched"), "joint_fixed":("joint_call","lower_fixed_upper"),
    "joint_source_common":("joint_call","source_upper_common"), "joint_common":("joint_call","joint_upper_common"),
    "joint":("joint_call","joint_call"),
}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (
    ("source_common","source_matched"), ("joint_common","joint_fixed"),
    ("joint_source_common","joint_matched"), ("source_joint_common","source_fixed"),
    ("joint_common","joint_source_common"), ("source_joint_common","source_common"),
    ("joint_common","joint"), ("source_joint_common","source_joint"),
    ("joint_source_common","source_common"), ("joint_common","source_joint_common"),
    ("joint_common","zero"), ("source_common","zero"), ("base","zero"),
)
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS[:2])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (92, 92092)


def donor_result(root, method):
    if method == "lower_fixed_upper":return ROOT/"results"/FIXED_TRAINING_RUN/"cells"/f"replicate_{root}"/"result.json"
    return source.donor_result(root,method)


def allocation(method, period):
    if method not in METHODS:raise ValueError("Stage92 trains only the two registered lower means")
    return source.allocation("lower_fixed_upper",period)


def seed_roles(root, *, preflight):
    roles = source.seed_roles(root,preflight=preflight)
    base = 92_090_000 if preflight else 92_100_000 + roots(preflight=False).index(root)*10000
    roles["native_evaluation"] = list(range(base+1,base+1+options(preflight=preflight)["evaluation_episodes"]))
    return roles


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0],preflight=preflight).horizon
    o = options(preflight=preflight)
    n = 2*o["credit_scenarios_per_batch"]*o["rollouts_per_scenario"]
    return {**training_budget(o,METHODS,VARIANTS,horizon=h,preflight=preflight),
        "checkpoint_loads":len(PERIODS)*len(CHECKPOINT_METHODS),
        "checkpoint_freeze_checks":len(PERIODS)*len(CHECKPOINT_METHODS),
        "training_initialization_checks":len(PERIODS)*len(METHODS),"actor_composition_checks":len(PERIODS)*len(VARIANTS),
        "upper_replay_forward_calls":len(METHODS)*o["updates"]*(n//2)*sum(h//p for p in PERIODS)}


def contract():
    common = source.contract()
    return {**{k:common[k] for k in ("source","decoder","credit","baseline","evaluation","artifacts")},
        "periods":list(PERIODS),"variants":list(VARIANTS),
        "methods":{str(p):{m:allocation(m,p) for m in METHODS} for p in PERIODS},
        "training_runs":{**common["training_runs"],"lower_fixed_upper":FIXED_TRAINING_RUN},
        "checkpoint_protocols":{**common["checkpoint_protocols"],"lower_fixed_upper":source.EXPERIMENT_PROTOCOL},
        "checkpoint_update":8,"checkpoint_methods":list(CHECKPOINT_METHODS),
        "compositions_upper_lower":{k:list(v) for k,v in COMPOSITIONS.items()},
        "training":"two_new_lower_mean_learners_U0_or_final_UJ_frozen_from_update1_source_L0_init_eight_updates_preflight_two",
        "sampling":"exact_Stage88_scenario_and_lower_noise_rosters_first_replica_unchanged_all_its_upper_gaussian_innovations_replayed_to_second_64_episodes_per_update",
        "noise_mapping":"Stage80_lower_streams_unchanged_upper_replay_only_during_pair_collection_fresh_Stage92_eval_uses_unmodified_legacy_RNG_path",
        "conditional_baseline":"other_rollout_lower_actions_independent_given_shared_upper_innovations_and_scenario_upper_is_not_trained",
        "budget":"lower_.00099_at50_.000995_at100_matches_Stage90_LM_and_Stage91_LC_no_search",
        "freeze":"U0_or_Stage88_update8_UJ_bit_exact_std_values_source_Adam_forecaster_decoder_original_teacher_unchanged",
        "statistics":"all26_reward_contrasts_equal_root_bootstrap65536_Bonferroni26_seed92_92092_initial_credit_covariance_descriptive_only",
        "primary_endpoints":list(PRIMARY_ENDPOINTS),
        "decision":"report_each_primary_CI_global_conditioning_benefit_requires_all_four_lower_bounds_positive_preflight_mechanical_only_no_adoption_or_seed_rescue",
        "limits":"fixed_upper_conditional_lower_intervention_not_joint_upper_learning_sharing_innovations_does_not_fix_upper_actions_and_reduces_independent_upper_draws_same_teachers_not_equal_total_compute_frequency_superiority_closed_Stage67_HOLD"}
