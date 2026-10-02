"""Frozen actor attribution for the confirmed Stage88 final checkpoints."""

from scripts import pointmaze_actor_swap_stage84_spec as previous
from scripts import pointmaze_call_weighted_replication_stage88_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
roots, arguments, source_result = source.roots, source.arguments, source.source_result
# Same seven interventions, two donors and evaluation budget as Stage84.
options, budget = previous.options, previous.budget
EXPERIMENT_PROTOCOL = "pointmaze_call_weighted_actor_swap_stage89_v1"
POLICY = "call_weighted_final_actor_swap"
RUNNER_SCRIPT = "scripts/run_pointmaze_call_weighted_actor_swap_stage89.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_call_weighted_actor_swap_stage89.py"
TRAINING_RUN = "pointmaze_call_weighted_replication_stage88_full_20261002_r1"
CHECKPOINT_METHODS = ("joint_call", "lower_trained")
COMPOSITIONS = {
    "base": ("source", "source"), "zero": ("source", "source"),
    "joint_call": ("joint_call", "joint_call"),
    "source_upper_joint_lower": ("source", "joint_call"),
    "lower_trained": ("source", "lower_trained"),
    "joint_upper_lower_only_lower": ("joint_call", "lower_trained"),
    "joint_upper_source_lower": ("joint_call", "source"),
}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (
    ("joint_call", "source_upper_joint_lower"),
    ("joint_upper_lower_only_lower", "lower_trained"),
    ("joint_upper_source_lower", "base"),
    ("source_upper_joint_lower", "lower_trained"),
    ("joint_call", "joint_upper_lower_only_lower"),
    ("joint_call", "lower_trained"),
    ("joint_call", "zero"), ("lower_trained", "zero"),
    ("joint_upper_lower_only_lower", "zero"), ("base", "zero"),
)
ENDPOINTS = tuple(k for p in PERIODS for k in (
    *(f"{p}/{a}_minus_{b}" for a,b in CONTRAST_PAIRS), f"{p}/upper_by_lower_interaction"))
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (89, 89089)


def training_result(root):
    return ROOT / "results" / TRAINING_RUN / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    base = 89_090_000 if preflight else 89_100_000 + roots(preflight=False).index(root)*10000
    return {"native_evaluation": list(range(base+1,base+1+options(preflight=preflight)["evaluation_episodes"]))}


def contract():
    return {**previous.contract(), "training_run": TRAINING_RUN,
        "checkpoint_protocol": source.EXPERIMENT_PROTOCOL,
        "checkpoint_methods": list(CHECKPOINT_METHODS), "variants": list(VARIANTS),
        "compositions_upper_lower": {k:list(v) for k,v in COMPOSITIONS.items()},
        "noise_mapping": "Stage80_explicit_mapping_on_fresh_disjoint_Stage89_environment_noise_roles",
        "decision": "direct_upper_requires_positive_joint_call_minus_source_upper_joint_lower_CI_in_both_periods_transfer_and_interaction_reported_separately_no_tuning_Stage67_HOLD_unchanged",
        "limits": "fixed_Stage88_checkpoint_actor_attribution_not_retraining_counterfactual_or_equal_training_budget_claim_for_composed_policies_not_frequency_superiority_no_cross_stage_teacher_pooling"}
