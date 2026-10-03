"""Rebuild joint/lower mean donors on the new Stage96/97 cohort."""

from scripts import pointmaze_call_weighted_replication_stage88_spec as reference
from scripts import pointmaze_fresh_decoder_stage97_spec as decoder

ROOT, PERIODS, CHUNK_SIZE = decoder.ROOT, reference.PERIODS, reference.CHUNK_SIZE
FISHER_RADIUS = reference.FISHER_RADIUS
METHODS, VARIANTS = reference.METHODS, reference.VARIANTS
CONTRAST_PAIRS, ENDPOINTS = reference.CONTRAST_PAIRS, reference.ENDPOINTS
PRIMARY_ENDPOINTS = reference.PRIMARY_ENDPOINTS
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = reference.BOOTSTRAP_DRAWS, reference.BOOTSTRAP_SEED
EXPERIMENT_PROTOCOL = "pointmaze_fresh_joint_stage98_v1"
POLICY = "fresh_teacher_call_weighted_mc"
RUNNER_SCRIPT = "scripts/run_pointmaze_fresh_joint_stage98.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_fresh_joint_stage98.py"
SOURCE_RUN = "pointmaze_fresh_decoder_stage97_full_20261003_r1"
roots, arguments, options = decoder.roots, decoder.arguments, reference.options
allocation, budget = reference.allocation, reference.budget


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def teacher_result(root):
    return decoder.source_result(root)


def teacher_raw(root):
    path = teacher_result(root).parent
    return path.with_name(path.name + "_raw")


def clone_checkpoint(root, period):
    return teacher_raw(root) / f"clone_{period}_final.pt"


def source_record(root):
    return {"teacher_result": str(teacher_result(root)), "decoder_result": str(source_result(root)),
        "teacher_protocol": decoder.source.EXPERIMENT_PROTOCOL, "decoder_protocol": decoder.EXPERIMENT_PROTOCOL,
        "checkpoints": {str(p): str(clone_checkpoint(root, p)) for p in PERIODS},
        "forecaster": str(teacher_raw(root) / "forecaster.npz"), "teacher_preflight": False,
        "decoder_preflight": False, "historical_artifact_loads": 0}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 98_000_000 if preflight else 98_100_000 + roots(preflight=False).index(root) * 100000
    o = options(preflight=preflight)
    rounds = [{"credit_" + name: [{"scenario_seed": base + 10000*j + offset + i,
        "noise_seeds": [base + 10000*j + offset + 2001 + 2*i, base + 10000*j + offset + 2002 + 2*i]}
        for i in range(o["credit_scenarios_per_batch"])] for name, offset in (("A", 1), ("B", 1001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base + 95001, base + 95001 + o["evaluation_episodes"]))}


def contract():
    return {**reference.source.contract(), "source": decoder.source.EXPERIMENT_PROTOCOL,
        "decoder": "full_Stage97_new_teacher_BC_response_first_feasible_scales_and_envelopes_no_recalibration",
        "source_cohort": "all_eight_Stage96_new_teachers_full_artifacts_even_in_preflight_no_old_weights",
        "source_run": SOURCE_RUN,
        "native_task": "same_environment_parameters_and_history_states_as_Stage88_new_optimizer_roots",
        "noise_mapping": "unchanged_Stage80_mapping_fresh_Stage98_training_and_evaluation_roles",
        "replication": "rebuild_joint_call_joint_level_lower_trained_from_new_L0_U0_not_old_joint_or_lower_donors",
        "statistics": "unchanged_all20_reward_contrasts_equal_root_bootstrap65536_Bonferroni20_same_bootstrap_indices",
        "confirmation": "all_four_primary_CI_lower_bounds_strictly_positive_no_period_root_or_checkpoint_selection",
        "downstream": "use_all_fixed_final_joint_call_donors_rebuild_matched_lowers_then_unchanged_staged_upper_rule",
        "decision": "preflight_mechanical_only_no_return_admission_keep_negative_donors_no_cross_stage_pooling",
        "limits": "new_teacher_fixed_std_decoder_MC_mean_learning_not_full_actor_critic_or_frequency_superiority"}


def confirmation(summary):
    return {"status": "mechanical_only" if summary["status"] == "preflight_passed" else
        ("confirmed" if all(summary["endpoints"][k]["ci"][0] > 0 for k in PRIMARY_ENDPOINTS) else "not_confirmed"),
        "primary_endpoints": list(PRIMARY_ENDPOINTS), "population": "eight_new_Stage96_teachers_no_old_cohort_pooling"}
