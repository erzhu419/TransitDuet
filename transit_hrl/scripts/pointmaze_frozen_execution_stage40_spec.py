"""Frozen upper/gate execution alignment during lower-only learning."""

import numpy as np
from scripts import pointmaze_critic_calibration_stage39_spec as previous

ROOT, source = previous.ROOT, previous.source
EXPERIMENT_PROTOCOL = "pointmaze_frozen_execution_stage40_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_frozen_execution_stage40.py"
METHODS = ("frozen", "intrinsic_sampled", "intrinsic_aligned", "task_sampled", "task_aligned")
MODES, METRICS = previous.MODES, previous.METRICS
LOWER_CREDIT = {m: "task_option" if m.startswith("task_") else "intrinsic_option" for m in METHODS}
ENDPOINTS = tuple(f"{m}:frozen:final_return" for m in METHODS[1:]) + (
    "intrinsic:alignment:final_return", "task:alignment:final_return",
    "intrinsic:alignment:first_update_return", "task:alignment:first_update_return")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (40, 40040)
roots, options, iterations, snapshots = previous.roots, previous.options, previous.iterations, previous.snapshots
source_result, budget = previous.source_result, previous.budget


def verification_budget(*, preflight):
    horizon = source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    count = len(snapshots(preflight=preflight)) * len(MODES) + 3
    return {"native_episodes_per_cell": count,
            "total_primitive_steps": count * horizon * len(roots(preflight=preflight)) * len(METHODS)}


def update_kind(method, iteration, *, preflight):
    if method not in METHODS or not 1 <= iteration <= iterations(preflight=preflight):
        raise ValueError("unregistered execution method/iteration")
    if method == "frozen":
        return "none"
    return "critic" if iteration <= options(preflight=preflight)["warmup_iterations"] else "actor_critic"


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered execution root")
    base = 10_490_000 if preflight else 10_500_000 + roster.index(root) * 10000
    opt = options(preflight=preflight)
    count = iterations(preflight=preflight) * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)), "probe": [base + 2001],
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([40, root, seed, 40017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([40, root, iteration]).generate_state(1)[0])


def rollout_arguments(root, method, seed, *, phase, mode):
    sampled = phase in ("train", "probe")
    upper = sampled and not (phase == "train" and mode == "learning" and method.endswith("_aligned"))
    lower = sampled or mode == "lower_sampled"
    lower_seed = int(np.random.SeedSequence([40, root, seed, 40019]).generate_state(1)[0]) if lower else None
    gate_seed = int(np.random.SeedSequence([40, root, seed, 40029]).generate_state(1)[0]) if upper else None
    return {"sample": sampled, "upper_sample": upper, "gate_sample": upper,
            "lower_sample": lower, "lower_seed": lower_seed, "gate_seed": gate_seed}


def contrasts(means, *, preflight=False):
    final = means[str(iterations(preflight=preflight))]["deterministic"]
    first = means[str(options(preflight=preflight)["warmup_iterations"] + 1)]["deterministic"]
    values = [final[m]["episode_return"] - final["frozen"]["episode_return"] for m in METHODS[1:]]
    for stage in (final, first):
        values.extend(stage[s + "_aligned"]["episode_return"] - stage[s + "_sampled"]["episode_return"]
                      for s in ("intrinsic", "task"))
    return values


def contract():
    return {**previous.contract(), "methods": list(METHODS), "lower_credit": LOWER_CREDIT,
            "warmup": "identical_all_level_sampled_critic_only_data_and_updates_within_each_reward_pair",
            "training_sampling": "lower_always_sampled_upper_and_gate_deterministic_only_in_aligned_learning_phase",
            "learning_execution": {m: "deterministic_upper_gate" if m.endswith("_aligned") else "sampled_upper_gate" for m in METHODS},
            "critic_initialization": "inherited_then_common_sixteen_iteration_critic_only_warmup_no_reset_or_rescale",
            "lower_sampling_seed": "numpy_seedsequence_40_root_environment_seed_40019_plus_primitive_step",
            "gate_sampling_seed": "numpy_seedsequence_40_root_environment_seed_40029_plus_gate_step",
            "evaluation_policy_seed": "numpy_seedsequence_40_root_environment_seed_40017",
            "training_policy_seed": "environment_seed_plus_optimizer_root",
            "shuffle_seed": "numpy_seedsequence_40_root_global_iteration",
            "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED),
            "decision": "frozen_level_training_execution_diagnosis_not_algorithm_superiority"}
