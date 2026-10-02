"""One fixed-budget lower learner paired with the registered Stage88 donors."""

from scripts import pointmaze_call_weighted_replication_stage88_spec as source
from scripts import pointmaze_call_weighted_actor_swap_stage89_spec as swaps
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
training_result, TRAINING_RUN, CHECKPOINT_METHODS = swaps.training_result, swaps.TRAINING_RUN, swaps.CHECKPOINT_METHODS
EXPERIMENT_PROTOCOL = "pointmaze_lower_budget_stage90_v1"
POLICY = "matched_lower_budget_diagnosis"
RUNNER_SCRIPT = "scripts/run_pointmaze_lower_budget_stage90.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_lower_budget_stage90.py"
METHODS = {"lower_matched": ("lower",)}
COMPOSITIONS = {
    "base": ("source", "source"), "zero": ("source", "source"),
    "lower_matched": ("source", "lower_matched"),
    "lower_full": ("source", "lower_trained"),
    "source_upper_joint_lower": ("source", "joint_call"),
    "joint_call": ("joint_call", "joint_call"),
    "joint_upper_lower_matched": ("joint_call", "lower_matched"),
    "joint_upper_lower_only_lower": ("joint_call", "lower_trained"),
}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (
    ("lower_matched", "lower_full"), ("source_upper_joint_lower", "lower_matched"),
    ("joint_upper_lower_matched", "joint_upper_lower_only_lower"), ("joint_call", "joint_upper_lower_matched"),
    ("joint_call", "source_upper_joint_lower"), ("joint_upper_lower_matched", "lower_matched"),
    ("joint_upper_lower_only_lower", "lower_full"),
    ("source_upper_joint_lower", "lower_full"), ("joint_call", "joint_upper_lower_only_lower"),
    ("joint_call", "lower_full"),
    ("lower_matched", "zero"), ("lower_full", "zero"), ("joint_call", "zero"), ("base", "zero"),
)
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a,b in CONTRAST_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (90, 90090)


def allocation(method, period):
    if method != "lower_matched":raise ValueError("Only the matched-budget lower is trained in Stage90")
    return {"lower": source.allocation("joint_call",period)["lower"]}


def seed_roles(root, *, preflight):
    o = options(preflight=preflight)
    original = source.seed_roles(root,preflight=False)["training_rounds"]
    rounds = [{"credit_"+b: r["credit_"+b][:o["credit_scenarios_per_batch"]] for b in ("A","B")}
        for r in original[:o["updates"]]]
    base = 90_090_000 if preflight else 90_100_000 + roots(preflight=False).index(root)*10000
    return {"training_rounds":rounds,"native_evaluation":list(range(base+1,base+1+o["evaluation_episodes"]))}


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0],preflight=preflight).horizon
    return {**training_budget(options(preflight=preflight),METHODS,VARIANTS,horizon=h,preflight=preflight),
        "checkpoint_loads":len(PERIODS)*len(CHECKPOINT_METHODS),
        "checkpoint_freeze_checks":len(PERIODS)*len(CHECKPOINT_METHODS),
        "actor_composition_checks":len(PERIODS)*len(VARIANTS)}


def contract():
    return {**source.contract(), "variants":list(VARIANTS),
        "methods":{str(p):{m:allocation(m,p) for m in METHODS} for p in PERIODS},
        "training_run":TRAINING_RUN,"checkpoint_protocol":source.EXPERIMENT_PROTOCOL,"checkpoint_update":8,
        "checkpoint_methods":list(CHECKPOINT_METHODS),"compositions_upper_lower":{k:list(v) for k,v in COMPOSITIONS.items()},
        "training":"only_one_new_lower_mean_learner_eight_updates_from_source_no_Stage88_retraining_preflight_two",
        "sampling":"exact_Stage88_full_training_rosters_preflight_first_two_rounds_two_scenarios_per_batch_all64_episodes_per_full_update",
        "noise_mapping":"Stage88_training_roles_reused_intentionally_fresh_disjoint_Stage90_paired_evaluation_roles",
        "budget":"new_lower_.001_minus_.0005_over_period_matches_Stage88_joint_call_lower_not_total_joint_budget_no_search",
        "allocation":"frozen_source_upper_new_lower_.00099_at50_.000995_at100_Stage88_donors_read_only",
        "statistics":"all28_final_reward_contrasts_equal_root_bootstrap65536_Bonferroni28",
        "decision":"decompose_fixed_upper_lower_gap_into_budget_and_matched_budget_training_effects_keep_each_CI_inconclusive_is_not_equivalence_no_tuning_Stage67_HOLD_unchanged",
        "limits":"conditional_diagnosis_same_Stage88_teachers_and_training_rosters_not_new_independent_replication_equal_total_training_budget_or_frequency_superiority_no_checkpoint_adoption"}
