"""Matched-capacity lower critic clock versus zero-clock control."""

import numpy as np
from scripts import pointmaze_warmup_alignment_stage41_spec as previous

ROOT, source = previous.ROOT, previous.source
EXPERIMENT_PROTOCOL = "pointmaze_critic_clock_stage42_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_critic_clock_stage42.py"
METHODS = ("frozen", "intrinsic_sham", "intrinsic_clock", "task_sham", "task_clock")
MODES, METRICS = previous.MODES, previous.METRICS
LOWER_CREDIT = {m: "task_option" if m.startswith("task_") else "intrinsic_option" for m in METHODS}
VALUE_CLOCK = {m: m.endswith("_clock") for m in METHODS}
ENDPOINTS = tuple(f"{m}:frozen:final_return" for m in METHODS[1:]) + (
    "intrinsic:clock:final_return", "task:clock:final_return",
    "intrinsic:clock:first_update_return", "task:clock:first_update_return")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (42, 42042)
roots, options, iterations, snapshots = previous.roots, previous.options, previous.iterations, previous.snapshots
source_result, budget, verification_budget = previous.source_result, previous.budget, previous.verification_budget


def update_kind(method, iteration, *, preflight):
    if method not in METHODS or not 1 <= iteration <= iterations(preflight=preflight):
        raise ValueError("unregistered critic-clock method/iteration")
    if method == "frozen":
        return "none"
    return "critic" if iteration <= options(preflight=preflight)["warmup_iterations"] else "actor_critic"


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered critic-clock root")
    base = 10_690_000 if preflight else 10_700_000 + roster.index(root) * 10000
    opt = options(preflight=preflight)
    count = iterations(preflight=preflight) * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)), "probe": [base + 2001],
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([42, root, seed, 42017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([42, root, iteration]).generate_state(1)[0])


def rollout_arguments(root, method, seed, *, phase, mode):
    sampled = phase in ("train", "probe")
    upper, lower = phase == "probe", sampled or mode == "lower_sampled"
    lower_seed = int(np.random.SeedSequence([42, root, seed, 42019]).generate_state(1)[0]) if lower else None
    gate_seed = int(np.random.SeedSequence([42, root, seed, 42029]).generate_state(1)[0]) if upper else None
    return {"sample": sampled, "upper_sample": upper, "gate_sample": upper,
            "lower_sample": lower, "lower_seed": lower_seed, "gate_seed": gate_seed}


def contrasts(means, *, preflight=False):
    final = means[str(iterations(preflight=preflight))]["deterministic"]
    first = means[str(options(preflight=preflight)["warmup_iterations"] + 1)]["deterministic"]
    values = [final[m]["episode_return"] - final["frozen"]["episode_return"] for m in METHODS[1:]]
    for stage in (final, first):
        values.extend(stage[s + "_clock"]["episode_return"] - stage[s + "_sham"]["episode_return"] for s in ("intrinsic", "task"))
    return values


def contract():
    inherited = previous.contract()
    for key in ("method_suffix", "warmup_pair", "first_learning_pair"):
        inherited.pop(key)
    return {**inherited, "methods": list(METHODS), "lower_credit": LOWER_CREDIT,
            "warmup": "all_learning_arms_critic_only_with_deterministic_upper_gate_stochastic_lower",
            "training_sampling": "upper_gate_always_deterministic_lower_sampled_in_warmup_and_learning",
            "learning_execution": {m: "deterministic_upper_gate" for m in METHODS},
            "lower_value_state": "unchanged_actor_state_plus_current_option_age_over_100_and_remaining_episode_fraction",
            "value_clock": VALUE_CLOCK, "critic_capacity": "two_extra_inputs_in_all_arms_sham_and_frozen_receive_zeros",
            "critic_initialization": "inherited_MLP_weights_plus_zero_clock_columns_fresh_Adam_no_reset_or_rescale",
            "warmup_pair": "identical_actor_data_and_frozen_networks_lower_critic_inputs_weights_Adam_may_differ",
            "first_learning_pair": "native_state_action_reward_duration_done_logp_equal_old_value_and_value_context_may_differ",
            "clock_timing": "current_option_age_after_any_renewal_before_lower_action_no_future_observations",
            "lower_sampling_seed": "numpy_seedsequence_42_root_environment_seed_42019_plus_primitive_step",
            "gate_sampling_seed": "numpy_seedsequence_42_root_environment_seed_42029_plus_gate_step",
            "evaluation_policy_seed": "numpy_seedsequence_42_root_environment_seed_42017",
            "shuffle_seed": "numpy_seedsequence_42_root_global_iteration",
            "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED),
            "decision": "critic_only_control_clock_information_diagnosis_not_algorithm_superiority"}
