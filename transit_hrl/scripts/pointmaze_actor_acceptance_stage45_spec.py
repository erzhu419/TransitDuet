"""Training-objective-only acceptance of the original lower PPO update."""

import numpy as np
from scripts import pointmaze_update_direction_stage43_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_actor_acceptance_stage45_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_actor_acceptance_stage45.py"
POLICY = "actor_acceptance"
METHODS, MODES, METRICS = source.METHODS, source.MODES, source.METRICS
TREATMENTS = ("vanilla", "accepted")
POLICIES = ("frozen", *(f"{m}:{t}" for m in METHODS for t in TREATMENTS))
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = source.SOURCE_FULL_RUN, source.SOURCE_PREFLIGHT_RUN
source_result, roots = source.source_result, source.roots
ENDPOINTS = tuple(f"{m}:{k}" for m in METHODS for k in
                  ("accepted_vanilla_first", "accepted_vanilla_final", "accepted_frozen_final", "vanilla_frozen_final"))
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (45, 45045)


def options(*, preflight):
    original = source.options(preflight=preflight)
    return {k: original[k] for k in ("learning_iterations", "rollouts_per_iteration", "evaluation_paths", "workers")}


def snapshots(*, preflight):
    return (1, options(preflight=preflight)["learning_iterations"])


def arguments(root, *, preflight):
    return source.source.source.arguments(root, preflight=preflight)


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 10_990_000 if preflight else 11_000_000 + index * 10000
    opt = options(preflight=preflight)
    count = opt["learning_iterations"] * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([45, root, seed, 45017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([45, root, iteration]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    sampled = phase == "train" or mode == "lower_sampled"
    return {"sample": phase == "train", "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([45, root, seed, 45019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    training = len(METHODS) * len(TREATMENTS) * opt["learning_iterations"] * opt["rollouts_per_iteration"]
    evaluation = (1 + len(METHODS) * len(TREATMENTS) * len(snapshots(preflight=preflight))) * len(MODES) * opt["evaluation_paths"]
    return {"training_primitive_steps": training * horizon, "evaluation_primitive_steps": evaluation * horizon,
            "total_primitive_steps": (training + evaluation) * horizon, "native_trace_audits": training + evaluation}


def contrasts(means, *, preflight):
    first, final = (means[str(i)]["lower_sampled"] for i in snapshots(preflight=preflight))
    reference = means["0"]["lower_sampled"]["frozen"]["episode_return"]
    values = {}
    for method in METHODS:
        accepted, vanilla = (final[f"{method}:{t}"]["episode_return"] for t in ("accepted", "vanilla"))
        effects = (first[method + ":accepted"]["episode_return"] - first[method + ":vanilla"]["episode_return"],
                   accepted - vanilla, accepted - reference, vanilla - reference)
        keys = ("accepted_vanilla_first", "accepted_vanilla_final", "accepted_frozen_final", "vanilla_frozen_final")
        values.update({f"{method}:{key}": value for key, value in zip(keys, effects)})
    return values


def contract():
    return {"source_protocol": source.source.EXPERIMENT_PROTOCOL, "methods": list(METHODS), "treatments": list(TREATMENTS),
            "initialization": "fixed_last_warmup_checkpoint_including_actor_and_critic_Adam_no_new_warmup",
            "learning": "original_lower_PPO_epochs_minibatches_GAE_entropy_gradient_clip_learning_rate_unchanged",
            "acceptance": "tentative_full_batch_clipped_surrogate_plus_original_entropy_not_below_before",
            "rejection": "restore_lower_actor_and_its_entire_Adam_state_keep_updated_lower_critic",
            "acceptance_inputs": "current_training_batch_only_no_evaluation_data_or_step_search",
            "frozen_levels": "upper_gate_actors_and_values_unchanged_parameters_state_mediated_actions_may_differ",
            "paired_initialization": "exact_same_actor_critic_and_optimizer_state_within_each_method_pair",
            "first_batch": "all_training_arrays_equal_within_method_pair_before_first_update",
            "sampling": "deterministic_upper_gate_sampled_lower_training_two_lower_deployment_modes",
            "lower_sampling_seed": "numpy_seedsequence_45_root_environment_seed_45019_plus_primitive_step",
            "shuffle_seed": "numpy_seedsequence_45_root_learning_iteration",
            "snapshots": "first_and_fixed_final_only_shared_frozen_before_reference",
            "cost": "count_executed_actor_steps_separately_from_retained_steps_critic_steps_always_retained",
            "primary_mode": "lower_sampled", "descriptive_mode": "deterministic",
            "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED), "bootstrap_draws": BOOTSTRAP_DRAWS,
            "interval": "two_sided_percentile_bonferroni_16_equal_root_paired_means",
            "checkpoint_selection": "none", "root_exclusion": "forbidden", "sequential_extension": "forbidden",
            "decision": "accepted_minus_vanilla_positive_is_relative_benefit_accepted_minus_frozen_positive_also_needed_for_learning_utility",
            "evidence_role": "conditional_development_intervention_on_reused_roots_not_independent_confirmation"}
