"""Episode-held-out causal state baselines for full-task actor credit."""

import numpy as np
from scripts import pointmaze_episode_credit_stage46_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_state_baseline_stage47_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_state_baseline_stage47.py"
POLICY = "state_baseline"
METHODS = ("task_clock",)
TREATMENTS = ("gae", "episode_mc", "state_mc")
POLICIES = ("frozen", *(f"{m}:{t}" for m in METHODS for t in TREATMENTS))
MODES, METRICS = previous.MODES, previous.METRICS
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments, snapshots, warmup_iterations = (
    previous.source_result, previous.roots, previous.arguments, previous.snapshots, previous.warmup_iterations)
ENDPOINTS = ("state_mc_first", "state_mc_final", "state_gae_final", "state_frozen_final",
             "mc_frozen_final", "gae_frozen_final", "first_gradient_dispersion_reduction")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (47, 47047)


def options(*, preflight):
    return {**previous.options(preflight=preflight), "rollouts_per_iteration": 4 if preflight else 8}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 11_190_000 if preflight else 11_200_000 + index * 10000
    opt = options(preflight=preflight)
    count = opt["learning_iterations"] * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([47, root, seed, 47017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([47, root, iteration]).generate_state(1)[0])


def baseline_seed(root, iteration, fold):
    return int(np.random.SeedSequence([47, root, iteration, fold, 47031]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    sampled = phase == "train" or mode == "lower_sampled"
    return {"sample": phase == "train", "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([47, root, seed, 47019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    train = len(TREATMENTS) * opt["learning_iterations"] * opt["rollouts_per_iteration"]
    evaluate = (1 + len(TREATMENTS) * len(snapshots(preflight=preflight))) * len(MODES) * opt["evaluation_paths"]
    return {"training_primitive_steps": train * horizon, "evaluation_primitive_steps": evaluate * horizon,
            "total_primitive_steps": (train + evaluate) * horizon, "native_trace_audits": train + evaluate}


def contrasts(means, *, preflight, diagnostics):
    first, final = (means[str(i)]["lower_sampled"] for i in snapshots(preflight=preflight))
    method = METHODS[0]
    state, mc, gae = (final[f"{method}:{t}"]["episode_return"] for t in ("state_mc", "episode_mc", "gae"))
    reference = means["0"]["lower_sampled"]["frozen"]["episode_return"]
    gradient = diagnostics["training"][method]["state_mc"][0]["gradient_dispersion"]
    return dict(zip(ENDPOINTS, (first[method + ":state_mc"]["episode_return"] - first[method + ":episode_mc"]["episode_return"],
        state - mc, state - gae, state - reference, mc - reference, gae - reference,
        gradient["episode_mc"]["trace_dispersion"] - gradient["state_mc"]["trace_dispersion"])))


def contract():
    return {"source_protocol": previous.previous.source.source.EXPERIMENT_PROTOCOL,
        "methods": list(METHODS), "treatments": list(TREATMENTS),
        "initialization": "task_clock_fixed_Stage42_warmup16_and_Adam_no_new_warmup",
        "actor_credit": "original_GAE_vs_time_LOO_MC_vs_episode_held_out_state_MC_all_original_normalization",
        "state_inputs": "causal_pre_action_lower_history_plus_option_age_and_remaining_episode_horizon",
        "fit": "exclude_entire_query_episode_from_labels_time_mean_and_feature_target_normalization",
        "baseline": "time_LOO_plus_normalized_RTG_residual_ValueNet_original_hidden_dim_zero_output_head",
        "fit_optimizer": "fresh_Adam_original_lower_lr_epochs_minibatch_value_coef_gradclip_each_fold_each_round",
        "fit_randomness": "isolated_torch_fork_rng_and_local_numpy_seedsequence_47_root_round_fold_47031",
        "critic": "original_task_option_GAE_targets_and_updates_unchanged",
        "learning": "original_PPO_settings_no_acceptance_no_baseline_selection_or_early_stopping",
        "frozen_levels": "upper_gate_actor_and_value_parameters_fixed_state_mediated_actions_may_differ",
        "pairing": "all_three_first_native_batches_task_rewards_and_first_critic_Adam_updates_exact",
        "sampling": "deterministic_upper_gate_sampled_lower_training_two_lower_evaluation_modes",
        "seeds": "policy_47_root_env_47017_lower_noise_47019_shuffle_47_root_round",
        "gradient": "pre_update_episode_mean_logp_score_with_global_normalized_advantage_trace_sample_dispersion",
        "gradient_limits": "crossfit_coupled_episode_dispersion_not_IID_variance_or_native_utility",
        "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED), "bootstrap_draws": BOOTSTRAP_DRAWS,
        "interval": "two_sided_percentile_bonferroni_7_equal_root_paired_means",
        "decision": "state_minus_MC_final_and_state_minus_frozen_positive_needed_for_learning_repair",
        "checkpoint_selection": "none", "root_exclusion": "forbidden", "sequential_extension": "forbidden",
        "evidence_role": "conditional_development_on_reused_roots_not_independent_confirmation"}
