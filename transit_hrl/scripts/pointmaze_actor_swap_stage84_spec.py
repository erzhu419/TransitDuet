"""Frozen final-checkpoint actor interventions, without additional training."""

from scripts import pointmaze_iterative_mc_stage83_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
roots, arguments, source_result = source.roots, source.arguments, source.source_result
EXPERIMENT_PROTOCOL = "pointmaze_actor_swap_stage84_v1"
POLICY = "final_actor_swap"
RUNNER_SCRIPT = "scripts/run_pointmaze_actor_swap_stage84.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_actor_swap_stage84.py"
TRAINING_RUN = "pointmaze_iterative_mc_stage83_full_20261002_r1"
CHECKPOINT_METHODS = tuple(source.METHODS)
COMPOSITIONS = {
    "base": ("source", "source"), "zero": ("source", "source"),
    "joint_trained": ("joint_trained", "joint_trained"),
    "source_upper_joint_lower": ("source", "joint_trained"),
    "lower_trained": ("source", "lower_trained"),
    "joint_upper_lower_only_lower": ("joint_trained", "lower_trained"),
    "joint_upper_source_lower": ("joint_trained", "source"),
}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (
    ("joint_trained", "source_upper_joint_lower"),
    ("joint_upper_lower_only_lower", "lower_trained"),
    ("joint_upper_source_lower", "base"),
    ("source_upper_joint_lower", "lower_trained"),
    ("joint_trained", "joint_upper_lower_only_lower"),
    ("joint_trained", "lower_trained"),
    ("joint_trained", "zero"), ("lower_trained", "zero"),
    ("joint_upper_lower_only_lower", "zero"), ("base", "zero"),
)
ENDPOINTS = tuple(k for p in PERIODS for k in (
    *(f"{p}/{a}_minus_{b}" for a, b in CONTRAST_PAIRS), f"{p}/upper_by_lower_interaction"))
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (84, 84084)


def training_result(root):
    return ROOT / "results" / TRAINING_RUN / "cells" / f"replicate_{root}" / "result.json"


def options(*, preflight):
    return {"workers": 2 if preflight else 8, "evaluation_episodes": 4 if preflight else 32}


def seed_roles(root, *, preflight):
    base = 84_090_000 if preflight else 84_100_000 + roots(preflight=False).index(root)*10000
    return {"native_evaluation": list(range(base+1,base+1+options(preflight=preflight)["evaluation_episodes"]))}


def budget(*, preflight):
    h = arguments(roots(preflight=preflight)[0],preflight=preflight).horizon
    e = options(preflight=preflight)["evaluation_episodes"]
    count = len(PERIODS)*len(VARIANTS)*e
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "decoder_loads": len(PERIODS),
        "checkpoint_loads": len(PERIODS)*len(source.METHODS), "checkpoint_freeze_checks": len(PERIODS)*len(source.METHODS),
        "actor_composition_checks": len(PERIODS)*len(VARIANTS), "frozen_model_checks": len(PERIODS),
        "evaluation_episodes": count, "native_episodes": count, "native_steps": count*h,
        "native_lower_calls": count*h, "native_upper_calls": len(VARIANTS)*e*sum(h//p for p in PERIODS),
        "pairing_upper_forward_calls": len(VARIANTS)*e*sum(h//p for p in PERIODS),
        "native_network_checks": count, "native_pair_checks": len(PERIODS)*e,
        "credit_episodes": 0, "policy_updates": 0, "checkpoint_writes": 0}


def contract():
    return {"source": source.contract()["source"], "decoder": source.contract()["decoder"],
        "training_run": TRAINING_RUN, "checkpoint_protocol": source.EXPERIMENT_PROTOCOL,
        "checkpoint_update": 8, "periods": list(PERIODS), "variants": list(VARIANTS),
        "compositions_upper_lower": {k:list(v) for k,v in COMPOSITIONS.items()},
        "intervention": "2x2_source_or_joint_upper_cross_joint_or_lower_only_lower_plus_source_lower_and_zero_controls",
        "freeze": "bit_exact_actor_composition_both_std_and_values_source_Adam_forecaster_decoder_unchanged",
        "sampling": "32_fresh_paired_evaluation_seeds_per_root_period_preflight_four_no_training_or_checkpoint_selection",
        "noise_mapping": "Stage80_explicit_mapping_on_fresh_disjoint_Stage84_environment_noise_roles",
        "statistics": "all22_reward_and_interaction_contrasts_equal_root_bootstrap65536_Bonferroni22",
        "decision": "isolate_direct_upper_effect_and_lower_training_difference_Stage67_critic_credit_HOLD_unchanged",
        "artifacts": "read_final_weights_server_only_no_trace_or_checkpoint_writes_local_compact_JSON_and_completion_only",
        "limits": "fixed_checkpoint_interventions_not_retraining_counterfactual_or_flat_RL_comparison_not_frequency_superiority"}
