"""Isolate full-episode task actor credit from the original option GAE."""

import numpy as np
from scripts import pointmaze_actor_acceptance_stage45_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_episode_credit_stage46_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_episode_credit_stage46.py"
POLICY = "episode_credit"
METHODS = ("task_sham", "task_clock")
TREATMENTS = ("gae", "episode_mc")
POLICIES = ("frozen", *(f"{m}:{t}" for m in METHODS for t in TREATMENTS))
MODES, METRICS = previous.MODES, previous.METRICS
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments = previous.source_result, previous.roots, previous.arguments
ENDPOINT_KEYS = ("mc_gae_first", "mc_gae_final", "mc_frozen_final", "gae_frozen_final")
ENDPOINTS = tuple(f"{m}:{k}" for m in METHODS for k in ENDPOINT_KEYS)
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (46, 46046)


def options(*, preflight):
    opt = previous.options(preflight=preflight)
    return {**opt, "rollouts_per_iteration": 2 if preflight else opt["rollouts_per_iteration"]}


def snapshots(*, preflight):
    return (1, options(preflight=preflight)["learning_iterations"])


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 11_090_000 if preflight else 11_100_000 + index * 10000
    opt = options(preflight=preflight)
    count = opt["learning_iterations"] * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([46, root, seed, 46017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([46, root, iteration]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    sampled = phase == "train" or mode == "lower_sampled"
    return {"sample": phase == "train", "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([46, root, seed, 46019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    train = len(METHODS) * len(TREATMENTS) * opt["learning_iterations"] * opt["rollouts_per_iteration"]
    evaluate = (1 + len(METHODS) * len(TREATMENTS) * len(snapshots(preflight=preflight))) * len(MODES) * opt["evaluation_paths"]
    return {"training_primitive_steps": train * horizon, "evaluation_primitive_steps": evaluate * horizon,
            "total_primitive_steps": (train + evaluate) * horizon, "native_trace_audits": train + evaluate}


def contrasts(means, *, preflight):
    first, final = (means[str(i)]["lower_sampled"] for i in snapshots(preflight=preflight))
    reference = means["0"]["lower_sampled"]["frozen"]["episode_return"]
    values = {}
    for method in METHODS:
        mc, gae = (final[f"{method}:{t}"]["episode_return"] for t in ("episode_mc", "gae"))
        effects = (first[method + ":episode_mc"]["episode_return"] - first[method + ":gae"]["episode_return"],
                   mc - gae, mc - reference, gae - reference)
        values.update({f"{method}:{key}": value for key, value in zip(ENDPOINT_KEYS, effects)})
    return values


def contract():
    return {"source_protocol": previous.source.source.EXPERIMENT_PROTOCOL,
            "methods": list(METHODS), "treatments": list(TREATMENTS),
            "initialization": "fixed_task_sham_and_task_clock_warmup16_including_Adam_no_new_warmup",
            "actor_credit": "original_normalized_option_GAE_vs_normalized_undiscounted_episode_RTG_with_leave_one_episode_out_time_baseline",
            "episode_credit": "native_task_rewards_only_cross_all_option_renewals_no_critic_bootstrap",
            "baseline": "other_complete_training_episodes_at_same_time_no_evaluation_inputs",
            "critic": "original_task_option_GAE_targets_and_updates_unchanged",
            "learning": "original_PPO_learning_rate_epochs_minibatches_clip_entropy_gradclip_unchanged_no_acceptance",
            "frozen_levels": "upper_gate_actors_and_values_unchanged_parameters_state_mediated_actions_may_differ",
            "pairing": "same_actor_critic_Adam_initialization_and_all_first_batch_arrays_within_source_method",
            "sampling": "deterministic_upper_gate_sampled_lower_training_both_lower_deployment_modes",
            "lower_sampling_seed": "numpy_seedsequence_46_root_environment_seed_46019_plus_primitive_step",
            "shuffle_seed": "numpy_seedsequence_46_root_learning_iteration",
            "snapshots": "first_and_fixed_final_only_shared_frozen_before",
            "primary_mode": "lower_sampled", "descriptive_mode": "deterministic",
            "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED),
            "bootstrap_draws": BOOTSTRAP_DRAWS, "interval": "two_sided_percentile_bonferroni_8_equal_root_paired_means",
            "decision": "positive_mc_minus_gae_is_relative_only_mc_minus_frozen_also_needed_for_learning_utility",
            "checkpoint_selection": "none", "root_exclusion": "forbidden", "sequential_extension": "forbidden",
            "evidence_role": "conditional_development_on_reused_roots_not_independent_confirmation",
            "limits": "LOO_centering_before_normalization_is_a_task_score_estimator_not_a_finite_PPO_improvement_guarantee"}
