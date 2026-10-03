"""Matched lower donors for the fresh teacher cohort and its final joint upper."""

from scripts import pointmaze_upper_noise_replication_stage93_spec as reference
from scripts import pointmaze_fresh_joint_stage98_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, reference.PERIODS, reference.CHUNK_SIZE
FISHER_RADIUS = reference.FISHER_RADIUS
roots, arguments, options, source_result = source.roots, source.arguments, reference.options, source.source_result
METHODS, NOISE_MODES, CHECKPOINT_METHODS = reference.METHODS, reference.NOISE_MODES, reference.CHECKPOINT_METHODS
COMPOSITIONS, VARIANTS = reference.COMPOSITIONS, reference.VARIANTS
CONTRAST_PAIRS, ENDPOINTS, PRIMARY_ENDPOINTS = reference.CONTRAST_PAIRS, reference.ENDPOINTS, reference.PRIMARY_ENDPOINTS
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = reference.BOOTSTRAP_DRAWS, reference.BOOTSTRAP_SEED
allocation, budget = reference.allocation, reference.budget
EXPERIMENT_PROTOCOL = "pointmaze_fresh_lower_stage99_v1"
POLICY = "fresh_teacher_fixed_upper_lower"
RUNNER_SCRIPT = "scripts/run_pointmaze_fresh_lower_stage99.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_fresh_lower_stage99.py"
JOINT_RUN = "pointmaze_fresh_joint_stage98_full_20261003_r1"


def donor_result(root, method):
    if method != "joint_call":
        raise ValueError("fresh lower requires only the registered fresh joint-call donor")
    return ROOT / "results" / JOINT_RUN / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 99_000_000 if preflight else 99_100_000 + roots(preflight=False).index(root) * 100000
    o = options(preflight=preflight)
    rounds = [{"credit_" + name: [{"scenario_seed": base + 10000*j + offset + i,
        "noise_seeds": [base + 10000*j + offset + 2001 + 2*i, base + 10000*j + offset + 2002 + 2*i]}
        for i in range(o["credit_scenarios_per_batch"])] for name, offset in (("A", 1), ("B", 1001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base + 95001, base + 95001 + o["evaluation_episodes"]))}


def contract():
    c = reference.contract()
    return {**c, "source": source.decoder.source.EXPERIMENT_PROTOCOL,
        "decoder": source.contract()["decoder"], "source_run": source.SOURCE_RUN,
        "training_runs": {"joint_call": JOINT_RUN},
        "checkpoint_protocols": {"joint_call": source.EXPERIMENT_PROTOCOL},
        "training": "same_four_Stage93_lower_mean_learners_new_L0_U0_and_final_Stage98_UJ_fixed_eight_updates_preflight_two",
        "sampling": "fresh_Stage99_rosters_all_four_methods_same64_episodes_per_update_common_only_replays_first_upper_innovations_to_second_replica",
        "noise_mapping": "unchanged_Stage80_independent_mapping_common_upper_replay_training_only_no_replay_in_eval",
        "replication": "all_eight_new_Stage96_teachers_full_Stage97_decoders_Stage98_final_UJ_no_old_lower_weight_reuse",
        "freeze": "U0_or_final_Stage98_UJ_bit_exact_all_std_values_source_Adam_forecaster_decoder_original_teacher_unchanged",
        "statistics": "same_all26_reward_contrasts_equal_root_bootstrap65536_Bonferroni26_same_Stage93_bootstrap_indices",
        "downstream": "retain_all_registered_final_lowers_then_unchanged_staged_upper_rule_no_Stage94_95_pooling",
        "decision": "preflight_mechanical_only_all_four_positive_primary_CIs_for_conditioning_claim_no_performance_admission_or_root_rescue",
        "limits": "new_teacher_fixed_upper_conditional_lower_MC_mean_learning_not_full_actor_critic_or_frequency_superiority"}
