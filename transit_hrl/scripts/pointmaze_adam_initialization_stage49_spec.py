"""Actor Adam initialization crossed with GAE/MC under fixed episode KL."""

import numpy as np
from scripts import pointmaze_episode_kl_stage48_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_adam_initialization_stage49_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_adam_initialization_stage49.py"
POLICY = "adam_initialization"
METHODS = previous.METHODS
TREATMENTS = ("gae", "gae_fresh", "episode_mc", "mc_fresh")
CREDITS = {"gae": "gae", "gae_fresh": "gae", "episode_mc": "episode_mc", "mc_fresh": "episode_mc"}
FRESH_TREATMENTS = ("gae_fresh", "mc_fresh")
BOUNDED_TREATMENTS, CANDIDATE_PAIRS = TREATMENTS, ()
POLICIES = ("frozen", *(f"{m}:{t}" for m in METHODS for t in TREATMENTS))
MODES, METRICS = previous.MODES, previous.METRICS
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments, options, snapshots, warmup_iterations = (
    previous.source_result, previous.roots, previous.arguments, previous.options, previous.snapshots, previous.warmup_iterations)
KL_BUDGET, MAX_BACKTRACKS = previous.KL_BUDGET, previous.MAX_BACKTRACKS
ENDPOINTS = ("fresh_mc_inherited_mc_first", "fresh_mc_inherited_mc_final", "fresh_gae_inherited_gae_final",
             "fresh_mc_fresh_gae_final", "credit_reset_interaction_final", "fresh_mc_frozen_final",
             "inherited_mc_frozen_final", "fresh_gae_frozen_final", "inherited_gae_frozen_final")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (49, 49049)


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 11_390_000 if preflight else 11_400_000 + index * 10000
    opt = options(preflight=preflight)
    count = opt["learning_iterations"] * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([49, root, seed, 49017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([49, root, iteration]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    sampled = phase == "train" or mode == "lower_sampled"
    return {"sample": phase == "train", "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([49, root, seed, 49019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    train = len(TREATMENTS) * opt["learning_iterations"] * opt["rollouts_per_iteration"]
    evaluate = (1 + len(TREATMENTS) * len(snapshots(preflight=preflight))) * len(MODES) * opt["evaluation_paths"]
    return {"training_primitive_steps": train * horizon, "evaluation_primitive_steps": evaluate * horizon,
            "total_primitive_steps": (train + evaluate) * horizon, "native_trace_audits": train + evaluate}


def contrasts(means, *, preflight, diagnostics=None):
    first, final = (means[str(i)]["lower_sampled"] for i in snapshots(preflight=preflight))
    m = METHODS[0] + ":"
    mc, fresh_mc, gae, fresh_gae = (final[m + t]["episode_return"] for t in ("episode_mc", "mc_fresh", "gae", "gae_fresh"))
    frozen = means["0"]["lower_sampled"]["frozen"]["episode_return"]
    return dict(zip(ENDPOINTS, (first[m + "mc_fresh"]["episode_return"] - first[m + "episode_mc"]["episode_return"],
        fresh_mc - mc, fresh_gae - gae, fresh_mc - fresh_gae, (fresh_mc - mc) - (fresh_gae - gae),
        fresh_mc - frozen, mc - frozen, fresh_gae - frozen, gae - frozen)))


def contract():
    return {**previous.contract(), "methods": list(METHODS), "treatments": list(TREATMENTS),
        "initialization": "fixed_task_clock_Stage42_warmup16_pre2_no_new_warmup",
        "credit": "2x2_original_GAE_vs_time_LOO_full_task_MC_by_inherited_vs_fresh_actor_Adam",
        "optimizer_initialization": "clear_actor_Adam_state_once_at_first_update_fresh_arms_only_keep_parameter_groups",
        "critic": "never_reset_critic_Adam_retain_first_original_GAE_update_each_round",
        "budget": "all_four_arms_same_maximum_training_episode_Gaussian_KL_at_most_0.1",
        "pairing": "same_initial_actor_critic_critic_Adam_first_native_batches_rewards_and_first_critic_update_all_four",
        "optimizer": "no_later_resets_accept_actual_scaled_lr_Adam_path_restore_nominal_lr_next_round",
        "seeds": "policy_49_root_env_49017_lower_noise_49019_shuffle_49_root_round_each_trial",
        "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED), "bootstrap_draws": BOOTSTRAP_DRAWS,
        "interval": "two_sided_percentile_bonferroni_9_equal_root_paired_means",
        "utility_decision": "fresh_MC_minus_inherited_MC_final_and_fresh_MC_minus_frozen_positive_needed_for_reset_repair",
        "factorial_decision": "credit_specific_reset_effect_requires_positive_registered_difference_in_differences",
        "limits": "same_KL_upper_bound_not_equal_realized_KL_empirical_history_not_population_bound_initial_reset_not_per_round"}
