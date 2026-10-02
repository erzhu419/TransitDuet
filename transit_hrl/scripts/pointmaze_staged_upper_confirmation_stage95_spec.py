"""Independent upper-training/evaluation confirmation with fixed Stage93 lowers."""

from scripts import pointmaze_staged_upper_stage94_spec as reference

source = reference.source
ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = reference.ROOT, reference.PERIODS, reference.CHUNK_SIZE, reference.FISHER_RADIUS
roots, arguments, source_result, options = reference.roots, reference.arguments, reference.source_result, reference.options
METHODS, CHECKPOINT_METHODS, LOWER_FOR_METHOD = reference.METHODS, reference.CHECKPOINT_METHODS, reference.LOWER_FOR_METHOD
COMPOSITIONS, VARIANTS, CONTRAST_PAIRS = reference.COMPOSITIONS, reference.VARIANTS, reference.CONTRAST_PAIRS
ENDPOINTS, PRIMARY_ENDPOINTS = reference.ENDPOINTS, reference.PRIMARY_ENDPOINTS
allocation, donor_result, budget = reference.allocation, reference.donor_result, reference.budget
EXPERIMENT_PROTOCOL = "pointmaze_staged_upper_confirmation_stage95_v1"
POLICY = "staged_upper_independent_confirmation"
RUNNER_SCRIPT = "scripts/run_pointmaze_staged_upper_confirmation_stage95.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_staged_upper_confirmation_stage95.py"
REFERENCE_RUN = "pointmaze_staged_upper_stage94_full_20261002_r1"
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (95, 95095)


def seed_roles(root, *, preflight):
    roles = reference.seed_roles(root,preflight=preflight)
    for r in roles["training_rounds"]:
        for b in ("A","B"):
            for s in r["credit_"+b]:
                s["scenario_seed"] += 1_000_000
                s["noise_seeds"] = [n+1_000_000 for n in s["noise_seeds"]]
    roles["native_evaluation"] = [s+1_000_000 for s in roles["native_evaluation"]]
    return roles


def contract():
    return {**reference.contract(), "confirmation_of":reference.EXPERIMENT_PROTOCOL, "reference_run":REFERENCE_RUN,
        "sampling":"fresh_Stage95_upper_training_and_evaluation_rosters_all_roles_shift_Stage94_by_1000000_same_independent_pairs_for_both_learners",
        "statistics":"all28_reward_contrasts_equal_root_bootstrap65536_Bonferroni28_seed95_95095",
        "confirmation":"both_uppers_retrained_from_U0_no_Stage94_upper_reuse_same_Stage93_lower_donors_and_Stage88_UJ_diagnostic_no_cross_stage_pooling",
        "limits":"conditional_upper_sample_confirmation_same_eight_teachers_and_fixed_Stage93_lowers_not_fresh_lower_training_or_unseen_teacher_validation_frequency_superiority_closed_Stage67_HOLD"}
