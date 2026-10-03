"""Upper refinement from registered UJ with both fresh matched lowers fixed."""

from scripts import pointmaze_fresh_staged_upper_stage100_spec as reference
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

source = reference.source
ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = reference.ROOT, reference.PERIODS, reference.CHUNK_SIZE, reference.FISHER_RADIUS
roots, arguments, source_result, source_record = reference.roots, reference.arguments, reference.source_result, reference.source_record
options, donor_result = reference.options, reference.donor_result
CHECKPOINT_METHODS = reference.CHECKPOINT_METHODS
LOWER_FOR_METHOD = {"refined_independent": "source_upper_independent", "refined_common": "source_upper_common"}
METHODS = {m: ("upper",) for m in LOWER_FOR_METHOD}
COMPOSITIONS = {
    "base": ("source", "source"), "zero": ("source", "source"),
    "source_independent": ("source", "source_upper_independent"), "source_common": ("source", "source_upper_common"),
    "fixed_joint_independent": ("joint_call", "source_upper_independent"), "fixed_joint_common": ("joint_call", "source_upper_common"),
    "refined_independent": ("refined_independent", "source_upper_independent"), "refined_common": ("refined_common", "source_upper_common"),
    "common_upper_independent_lower": ("refined_common", "source_upper_independent"),
    "independent_upper_common_lower": ("refined_independent", "source_upper_common"), "joint": ("joint_call", "joint_call")}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (
    ("refined_common", "fixed_joint_common"), ("refined_independent", "fixed_joint_independent"),
    ("refined_common", "refined_independent"), ("source_common", "source_independent"),
    ("refined_common", "source_common"), ("refined_independent", "source_independent"),
    ("fixed_joint_common", "source_common"), ("fixed_joint_independent", "source_independent"),
    ("refined_common", "joint"), ("refined_independent", "joint"),
    ("common_upper_independent_lower", "refined_independent"),
    ("independent_upper_common_lower", "refined_independent"),
    ("refined_common", "independent_upper_common_lower"), ("base", "zero"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS[:2])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = reference.BOOTSTRAP_DRAWS, reference.BOOTSTRAP_SEED
EXPERIMENT_PROTOCOL = "pointmaze_upper_refinement_stage101_v1"
POLICY = "UJ_initialized_fixed_lower_refinement"
RUNNER_SCRIPT = "scripts/run_pointmaze_upper_refinement_stage101.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_upper_refinement_stage101.py"
UPPER_INITIALIZATION = "registered_final_Stage98_joint_call_UJ"


def allocation(method, period):
    if method not in METHODS:
        raise ValueError("Upper refinement trains only the two registered upper means")
    return {"upper": .5}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 101_000_000 if preflight else 101_100_000 + roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    rounds = [{"credit_" + name: [{"scenario_seed": base + 10000*j + offset + i,
        "noise_seeds": [base + 10000*j + offset + 2001 + 2*i, base + 10000*j + offset + 2002 + 2*i]}
        for i in range(o["credit_scenarios_per_batch"])] for name, offset in (("A", 1), ("B", 1001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base + 95001, base + 95001 + o["evaluation_episodes"]))}


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    return {**training_budget(options(preflight=preflight), METHODS, VARIANTS, horizon=h, preflight=preflight),
        "checkpoint_loads": len(PERIODS)*len(CHECKPOINT_METHODS), "checkpoint_freeze_checks": len(PERIODS)*len(CHECKPOINT_METHODS),
        "training_initialization_checks": len(PERIODS)*len(METHODS), "upper_initialization_checks": len(PERIODS)*len(METHODS),
        "actor_composition_checks": len(PERIODS)*len(VARIANTS), "upper_replay_forward_calls": 0}


def contract():
    c = {k: v for k, v in reference.contract().items() if k not in ("confirmation_of", "reference_run", "confirmation")}
    return {**c, "variants": list(VARIANTS), "methods": {str(p): {m: allocation(m, p) for m in METHODS} for p in PERIODS},
        "lower_for_method": LOWER_FOR_METHOD, "compositions_upper_lower": {v: list(pair) for v, pair in COMPOSITIONS.items()},
        "upper_initialization": UPPER_INITIALIZATION, "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "training": "final_Stage98_UJ_initialized_upper_means_given_same_final_Stage99_U0_trained_lowers_eight_additional_updates_preflight_two",
        "sampling": "fresh_Stage101_disjoint_training_and_eval_roles_same_original_independent_upper_lower_pairs_for_both_learners",
        "budget": "incremental_upper_conditional_KL_.0005_per_update_after_UJ_no_radius_search_not_equal_total_compute_or_trajectory_KL",
        "statistics": "all28_reward_contrasts_equal_root_bootstrap65536_Bonferroni28_same_Stage95_bootstrap_seed",
        "decision": "all_four_refined_minus_fixed_UJ_primary_lower_CI_bounds_positive; crossed_upper_specialization_separate_four_sign_checks; preflight_mechanical_only",
        "limits": "conditional_UJ_refinement_extra_training_compute_same_new_teacher_cohort_not_full_actor_critic_or_frequency_superiority_Stage100_negatives_and_Stage67_HOLD_preserved"}
