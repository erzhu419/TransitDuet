"""Repeat the staged-upper rule with fresh teachers and rebuilt lower donors."""

from scripts import pointmaze_staged_upper_confirmation_stage95_spec as reference
from scripts import pointmaze_fresh_lower_stage99_spec as source

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, reference.PERIODS, reference.CHUNK_SIZE, reference.FISHER_RADIUS
roots, arguments, source_result, source_record = source.roots, source.arguments, source.source_result, source.source.source_record
options, budget, allocation = reference.options, reference.budget, reference.allocation
METHODS, CHECKPOINT_METHODS, LOWER_FOR_METHOD = reference.METHODS, reference.CHECKPOINT_METHODS, reference.LOWER_FOR_METHOD
COMPOSITIONS, VARIANTS, CONTRAST_PAIRS = reference.COMPOSITIONS, reference.VARIANTS, reference.CONTRAST_PAIRS
ENDPOINTS, PRIMARY_ENDPOINTS = reference.ENDPOINTS, reference.PRIMARY_ENDPOINTS
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = reference.BOOTSTRAP_DRAWS, reference.BOOTSTRAP_SEED
EXPERIMENT_PROTOCOL = "pointmaze_fresh_staged_upper_stage100_v1"
POLICY = "fresh_teacher_staged_upper_independent_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_fresh_staged_upper_stage100.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_fresh_staged_upper_stage100.py"
LOWER_TRAINING_RUN = "pointmaze_fresh_lower_stage99_full_20261003_r1"


def donor_result(root, method):
    if method in LOWER_FOR_METHOD.values():
        return ROOT / "results" / LOWER_TRAINING_RUN / "cells" / f"replicate_{root}" / "result.json"
    if method == "joint_call":
        return source.donor_result(root, method)
    raise ValueError("Fresh staged-upper donor is not registered")


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 100_000_000 if preflight else 100_100_000 + roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    rounds = [{"credit_" + name: [{"scenario_seed": base + 10000*j + offset + i,
        "noise_seeds": [base + 10000*j + offset + 2001 + 2*i, base + 10000*j + offset + 2002 + 2*i]}
        for i in range(o["credit_scenarios_per_batch"])] for name, offset in (("A", 1), ("B", 1001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base + 95001, base + 95001 + o["evaluation_episodes"]))}


def contract():
    c = reference.contract()
    parent = source.contract()
    return {**c, "source": parent["source"], "decoder": parent["decoder"], "source_run": parent["source_run"],
        "confirmation_of": reference.EXPERIMENT_PROTOCOL,
        "training_runs": {**{m: LOWER_TRAINING_RUN for m in LOWER_FOR_METHOD.values()}, "joint_call": source.JOINT_RUN},
        "checkpoint_protocols": {**{m: source.EXPERIMENT_PROTOCOL for m in LOWER_FOR_METHOD.values()},
            "joint_call": source.source.EXPERIMENT_PROTOCOL},
        "training": "new_U0_upper_means_given_final_Stage99_U0_trained_independent_or_common_lower_frozen_eight_updates_preflight_two",
        "sampling": "fresh_Stage100_training_eval_disjoint_roles_same_original_independent_upper_lower_pairs_for_both_learners",
        "freeze": "corresponding_Stage99_lower_std_values_source_Adam_forecaster_decoder_original_Stage96_teacher_unchanged",
        "statistics": "unchanged_all28_equal_root_bootstrap65536_Bonferroni28_same_Stage95_seed95_95095",
        "confirmation": "both_uppers_retrained_from_new_U0_no_old_upper_lower_weights_all_eight_new_roots_no_Stage94_95_pooling",
        "limits": "new_teacher_staged_MC_mean_learning_not_full_actor_critic_unseen_task_or_frequency_superiority_Stage67_HOLD"}
