"""Isolate frozen-level execution during critic-only pretraining."""

import numpy as np
from scripts import pointmaze_frozen_execution_stage40_spec as previous

ROOT, source = previous.ROOT, previous.source
EXPERIMENT_PROTOCOL = "pointmaze_warmup_alignment_stage41_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_warmup_alignment_stage41.py"
METHODS, MODES, METRICS = previous.METHODS, previous.MODES, previous.METRICS
LOWER_CREDIT = previous.LOWER_CREDIT
ENDPOINTS = tuple(f"{m}:frozen:final_return" for m in METHODS[1:]) + (
    "intrinsic:warmup_alignment:final_return", "task:warmup_alignment:final_return",
    "intrinsic:warmup_alignment:first_update_return", "task:warmup_alignment:first_update_return")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (41, 41041)
roots, options, iterations, snapshots = previous.roots, previous.options, previous.iterations, previous.snapshots
source_result, budget, verification_budget = previous.source_result, previous.budget, previous.verification_budget
update_kind = previous.update_kind


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered warmup alignment root")
    base = 10_590_000 if preflight else 10_600_000 + roster.index(root) * 10000
    opt = options(preflight=preflight)
    count = iterations(preflight=preflight) * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)), "probe": [base + 2001],
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([41, root, seed, 41017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([41, root, iteration]).generate_state(1)[0])


def rollout_arguments(root, method, seed, *, phase, mode):
    sampled = phase in ("train", "probe")
    upper = phase == "probe" or (phase == "train" and mode == "warmup" and not method.endswith("_aligned"))
    lower = sampled or mode == "lower_sampled"
    lower_seed = int(np.random.SeedSequence([41, root, seed, 41019]).generate_state(1)[0]) if lower else None
    gate_seed = int(np.random.SeedSequence([41, root, seed, 41029]).generate_state(1)[0]) if upper else None
    return {"sample": sampled, "upper_sample": upper, "gate_sample": upper,
            "lower_sample": lower, "lower_seed": lower_seed, "gate_seed": gate_seed}


def contrasts(means, *, preflight=False):
    return previous.contrasts(means, preflight=preflight)


def contract():
    return {**previous.contract(), "warmup": "critic_only_sampled_or_deterministic_upper_gate_lower_always_sampled",
            "training_sampling": "lower_always_sampled_upper_and_gate_deterministic_in_all_learning_arms",
            "method_suffix": "sampled_or_aligned_describes_warmup_only_not_learning_execution",
            "learning_execution": {m: "deterministic_upper_gate" for m in METHODS},
            "critic_initialization": "inherited_then_sixteen_iteration_critic_only_warmup_no_reset_or_rescale",
            "warmup_pair": "actors_and_frozen_values_identical_lower_critic_and_its_adam_may_differ",
            "first_learning_pair": "native_state_action_reward_duration_done_logp_equal_old_value_may_differ",
            "lower_sampling_seed": "numpy_seedsequence_41_root_environment_seed_41019_plus_primitive_step",
            "gate_sampling_seed": "numpy_seedsequence_41_root_environment_seed_41029_plus_gate_step",
            "evaluation_policy_seed": "numpy_seedsequence_41_root_environment_seed_41017",
            "shuffle_seed": "numpy_seedsequence_41_root_global_iteration",
            "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED),
            "decision": "critic_warmup_distribution_diagnosis_not_algorithm_superiority"}
