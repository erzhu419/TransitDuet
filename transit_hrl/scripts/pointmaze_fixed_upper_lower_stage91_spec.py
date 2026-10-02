"""Matched lower learning with the final Stage88 upper frozen from update one."""

from scripts import pointmaze_lower_budget_stage90_spec as source
from scripts import pointmaze_call_weighted_replication_stage88_spec as teacher_source
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
EXPERIMENT_PROTOCOL = "pointmaze_fixed_upper_lower_stage91_v1"
POLICY = "final_upper_fixed_matched_lower"
RUNNER_SCRIPT = "scripts/run_pointmaze_fixed_upper_lower_stage91.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_fixed_upper_lower_stage91.py"
MATCHED_TRAINING_RUN = "pointmaze_lower_budget_stage90_full_20261002_r1"
CHECKPOINT_METHODS = ("joint_call", "lower_matched")
METHODS = {"lower_fixed_upper": ("lower",)}
COMPOSITIONS = {
    "base": ("source", "source"), "zero": ("source", "source"),
    "lower_matched": ("source", "lower_matched"),
    "source_upper_joint_lower": ("source", "joint_call"),
    "source_upper_fixed_lower": ("source", "lower_fixed_upper"),
    "joint_call": ("joint_call", "joint_call"),
    "joint_upper_lower_matched": ("joint_call", "lower_matched"),
    "joint_upper_fixed_lower": ("joint_call", "lower_fixed_upper"),
}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (
    ("joint_upper_fixed_lower", "joint_call"), ("source_upper_fixed_lower", "source_upper_joint_lower"),
    ("joint_upper_fixed_lower", "joint_upper_lower_matched"), ("source_upper_fixed_lower", "lower_matched"),
    ("joint_call", "joint_upper_lower_matched"), ("source_upper_joint_lower", "lower_matched"),
    ("joint_upper_fixed_lower", "source_upper_fixed_lower"), ("joint_upper_lower_matched", "lower_matched"),
    ("joint_call", "source_upper_joint_lower"),
    ("joint_upper_fixed_lower", "zero"), ("lower_matched", "zero"), ("joint_call", "zero"), ("base", "zero"),
)
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/joint_upper_fixed_lower_minus_joint_call" for p in PERIODS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (91, 91091)


def donor_result(root, method):
    if method == "joint_call":return source.training_result(root)
    if method == "lower_matched":return ROOT/"results"/MATCHED_TRAINING_RUN/"cells"/f"replicate_{root}"/"result.json"
    raise ValueError("Only Stage88 joint-call and Stage90 matched-lower donors are registered")


def allocation(method, period):
    if method != "lower_fixed_upper":raise ValueError("Only the fixed-final-upper lower is trained in Stage91")
    return {"lower":source.allocation("lower_matched",period)["lower"]}


def seed_roles(root, *, preflight):
    roles = source.seed_roles(root,preflight=preflight)
    base = 91_090_000 if preflight else 91_100_000 + roots(preflight=False).index(root)*10000
    roles["native_evaluation"] = list(range(base+1,base+1+options(preflight=preflight)["evaluation_episodes"]))
    return roles


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0],preflight=preflight).horizon
    return {**training_budget(options(preflight=preflight),METHODS,VARIANTS,horizon=h,preflight=preflight),
        "checkpoint_loads":len(PERIODS)*len(CHECKPOINT_METHODS),
        "checkpoint_freeze_checks":len(PERIODS)*len(CHECKPOINT_METHODS),
        "training_initialization_checks":len(PERIODS), "actor_composition_checks":len(PERIODS)*len(VARIANTS)}


def contract():
    common = teacher_source.contract()
    return {**{k:common[k] for k in ("source","decoder","credit","baseline","evaluation","artifacts")},
        "periods":list(PERIODS),"variants":list(VARIANTS),
        "methods":{str(p):{m:allocation(m,p) for m in METHODS} for p in PERIODS},
        "training_runs":{"joint_call":source.TRAINING_RUN,"lower_matched":MATCHED_TRAINING_RUN},
        "checkpoint_protocols":{"joint_call":teacher_source.EXPERIMENT_PROTOCOL,"lower_matched":source.EXPERIMENT_PROTOCOL},
        "checkpoint_update":8,"checkpoint_methods":list(CHECKPOINT_METHODS),
        "compositions_upper_lower":{k:list(v) for k,v in COMPOSITIONS.items()},
        "training":"one_new_lower_mean_learner_source_lower_init_final_Stage88_upper_frozen_from_update1_eight_updates_preflight_two",
        "sampling":"exact_Stage88_full_training_scenario_noise_rosters_all64_episodes_per_update_preflight_first_two_rounds_two_scenarios_per_batch",
        "noise_mapping":"intentional_Stage88_training_reuse_fresh_disjoint_Stage91_paired_evaluation_roles",
        "budget":"lower_.00099_at50_.000995_at100_matches_Stage88_joint_call_and_Stage90_LM_no_search",
        "freeze":"training_upper_bit_exact_Stage88_update8_std_values_source_Adam_forecaster_decoder_unchanged_original_source_teacher_untouched",
        "statistics":"all26_final_reward_contrasts_equal_root_bootstrap65536_Bonferroni26_seed91_91091",
        "primary_endpoints":list(PRIMARY_ENDPOINTS),
        "decision":"report_fixed_final_upper_LC_minus_joint_LJ_and_LC_minus_source_upper_LM_at_both_evaluation_uppers_signed_CI_no_adoption_or_retuning_Stage67_HOLD_unchanged",
        "limits":"same_teachers_paired_training_conditional_diagnostic_final_upper_uses_prior_joint_training_not_equal_total_compute_or_isolated_temporal_movement_effect_CI_crossing_zero_not_equivalence_frequency_superiority_closed"}
