"""Training-path episode-scale KL backoff for full-task actor credit."""

import numpy as np
from scripts import pointmaze_state_baseline_stage47_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_episode_kl_stage48_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_episode_kl_stage48.py"
POLICY = "episode_kl"
METHODS = ("task_clock",)
TREATMENTS = ("gae", "episode_mc", "episode_kl")
CREDITS = {"gae": "gae", "episode_mc": "episode_mc", "episode_kl": "episode_mc"}
BOUNDED_TREATMENTS = ("episode_kl",)
CANDIDATE_PAIRS = (("episode_mc", "episode_kl"),)
POLICIES = ("frozen", *(f"{m}:{t}" for m in METHODS for t in TREATMENTS))
MODES, METRICS = previous.MODES, previous.METRICS
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments, options, snapshots, warmup_iterations = (
    previous.source_result, previous.roots, previous.arguments, previous.options, previous.snapshots, previous.warmup_iterations)
KL_BUDGET, MAX_BACKTRACKS = .1, 12
ENDPOINTS = ("bounded_mc_first", "bounded_mc_final", "bounded_gae_final", "bounded_frozen_final",
             "mc_frozen_final", "gae_frozen_final")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (48, 48048)


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 11_290_000 if preflight else 11_300_000 + index * 10000
    opt = options(preflight=preflight)
    count = opt["learning_iterations"] * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([48, root, seed, 48017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([48, root, iteration]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    sampled = phase == "train" or mode == "lower_sampled"
    return {"sample": phase == "train", "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([48, root, seed, 48019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    # Same number of source arms, paths and snapshots as Stage47; no auxiliary fits.
    return previous.budget(preflight=preflight)


def contrasts(means, *, preflight, diagnostics=None):
    first, final = (means[str(i)]["lower_sampled"] for i in snapshots(preflight=preflight))
    method = METHODS[0]
    bounded, mc, gae = (final[f"{method}:{t}"]["episode_return"] for t in ("episode_kl", "episode_mc", "gae"))
    reference = means["0"]["lower_sampled"]["frozen"]["episode_return"]
    return dict(zip(ENDPOINTS, (first[method + ":episode_kl"]["episode_return"] - first[method + ":episode_mc"]["episode_return"],
        bounded - mc, bounded - gae, bounded - reference, mc - reference, gae - reference)))


def contract():
    return {"source_protocol": previous.contract()["source_protocol"], "methods": list(METHODS), "treatments": list(TREATMENTS),
        "initialization": "fixed_task_clock_Stage42_warmup16_and_Adam_no_new_warmup",
        "credit": "original_GAE_vs_time_LOO_full_task_MC_vs_same_MC_with_episode_KL_backoff",
        "budget": "maximum_training_episode_sum_of_exact_old_to_new_conditional_Gaussian_KL",
        "kl_budget": KL_BUDGET, "max_backtracks": MAX_BACKTRACKS, "candidate_scales": "1_then_successive_halves",
        "trial": "restore_all_lower_actor_critic_and_Adam_states_reseed_same_shuffle_rerun_original_PPO_scaled_actor_lr_only",
        "decision": "first_candidate_with_max_episode_KL_at_most_budget_no_objective_floor_or_evaluation_access",
        "rejection": "restore_actor_and_Adam_keep_first_original_critic_update_if_all_13_trials_fail",
        "critic": "retain_exact_first_original_GAE_critic_and_Adam_update_once",
        "optimizer": "accepted_actual_scaled_lr_Adam_state_retained_original_lr_restored_for_next_round",
        "other_settings": "original_PPO_epochs_minibatches_clip_entropy_gradclip_unchanged_no_state_baseline",
        "cost": "charge_all_executed_actor_and_critic_trial_steps_distinguish_retained_steps_and_KL_checks",
        "pairing": "exact_three_arm_initial_actor_critic_Adam_batches_rewards_and_first_critic_update",
        "frozen_levels": "upper_gate_actor_and_value_parameters_fixed_state_mediated_actions_may_differ",
        "sampling": "deterministic_upper_gate_sampled_lower_training_both_lower_evaluation_modes",
        "seeds": "policy_48_root_env_48017_lower_noise_48019_shuffle_48_root_round_each_trial",
        "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED), "bootstrap_draws": BOOTSTRAP_DRAWS,
        "interval": "two_sided_percentile_bonferroni_6_equal_root_paired_means",
        "utility_decision": "bounded_minus_MC_final_and_bounded_minus_frozen_positive_needed_for_learning_repair",
        "checkpoint_selection": "none", "root_exclusion": "forbidden", "sequential_extension": "forbidden",
        "evidence_role": "conditional_development_on_reused_roots_not_independent_confirmation",
        "limits": "empirical_training_history_KL_not_population_trajectory_KL_or_native_reward_guarantee"}
