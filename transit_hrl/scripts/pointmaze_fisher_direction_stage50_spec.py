"""Fixed episode-KL comparison of Adam, Euclidean and natural score directions."""

import numpy as np
from scripts import pointmaze_episode_kl_stage48_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_fisher_direction_stage50_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_fisher_direction_stage50.py"
POLICY = "fisher_direction"
METHODS, MODES, METRICS = previous.METHODS, previous.MODES, previous.METRICS
TREATMENTS = ("gae", "episode_mc", "euclidean_mc", "fisher_mc")
CREDITS = {t: "gae" if t == "gae" else "episode_mc" for t in TREATMENTS}
BOUNDED_TREATMENTS, CANDIDATE_PAIRS = TREATMENTS, ()
POLICIES = ("frozen", *(f"{m}:{t}" for m in METHODS for t in TREATMENTS))
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments, options, snapshots, warmup_iterations = (
    previous.source_result, previous.roots, previous.arguments, previous.options, previous.snapshots, previous.warmup_iterations)
KL_BUDGET, MAX_BACKTRACKS = .1, 12
CG_ITERATIONS, CG_RTOL, FISHER_DAMPING = 10, 1e-6, .1
ENDPOINTS = ("fisher_adam_first", "fisher_adam_final", "fisher_euclidean_final", "fisher_frozen_final",
             "euclidean_adam_final", "euclidean_frozen_final", "adam_frozen_final", "gae_frozen_final")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (50, 50050)


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 11_690_000 if preflight else 11_700_000 + index * 10000
    opt = options(preflight=preflight)
    count = opt["learning_iterations"] * opt["rollouts_per_iteration"]
    return {"training": list(range(base + 1, base + 1 + count)),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([50, root, seed, 50017]).generate_state(1)[0])


def shuffle_seed(root, iteration):
    return int(np.random.SeedSequence([50, root, iteration]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    sampled = phase == "train" or mode == "lower_sampled"
    return {"sample": phase == "train", "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([50, root, seed, 50019]).generate_state(1)[0]) if sampled else None}


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
    fisher, euclidean, adam, gae = (final[m + t]["episode_return"] for t in ("fisher_mc", "euclidean_mc", "episode_mc", "gae"))
    frozen = means["0"]["lower_sampled"]["frozen"]["episode_return"]
    return dict(zip(ENDPOINTS, (first[m + "fisher_mc"]["episode_return"] - first[m + "episode_mc"]["episode_return"],
        fisher - adam, fisher - euclidean, fisher - frozen, euclidean - adam, euclidean - frozen, adam - frozen, gae - frozen)))


def contract():
    return {**previous.contract(), "treatments": list(TREATMENTS),
        "credit": "GAE_Adam_vs_time_LOO_full_task_MC_Adam_vs_same_MC_Euclidean_vs_same_MC_Fisher",
        "budget": "all_four_arms_maximum_training_episode_sum_exact_Gaussian_KL_at_most_0.1",
        "score": "one_full_batch_gradient_of_normalized_MC_log_probability_plus_original_entropy",
        "fisher": "float64_Hessian_of_full_batch_mean_old_to_new_conditional_Gaussian_KL_at_old_actor",
        "solver": "scipy_matrix_free_CG", "cg_iterations": CG_ITERATIONS, "cg_rtol": CG_RTOL, "fisher_damping": FISHER_DAMPING,
        "direction": "Euclidean_g_or_truncated_solve_of_F_plus_damping_I_times_d_equals_g",
        "initial_scale": "sqrt_2_KL_budget_over_horizon_times_undamped_d_F_d",
        "trial": "Adam_transactional_scaled_lr_or_direct_theta0_plus_halved_scaled_direction",
        "decision": "first_exact_max_episode_KL_feasible_no_surrogate_floor_or_evaluation_access",
        "critic": "original_GAE_critic_and_Adam_update_once_for_direct_directions_same_first_update_as_Adam_controls",
        "optimizer": "no_Adam_reset_direct_directions_leave_actor_Adam_untouched",
        "other_settings": "Adam_controls_original_PPO_direct_arms_one_full_batch_score_step_critic_epochs_minibatches_unchanged",
        "cost": "all_executed_Adam_steps_score_backward_HVP_CG_iterations_parameter_proposals_and_KL_checks_separate_from_retained",
        "pairing": "same_four_arm_initial_weights_native_training_batch_rewards_and_first_critic_update",
        "seeds": "policy_50_root_env_50017_lower_noise_50019_shuffle_50_root_round",
        "primary_endpoints": list(ENDPOINTS), "bootstrap_seed": list(BOOTSTRAP_SEED),
        "interval": "two_sided_percentile_bonferroni_8_equal_root_paired_means",
        "utility_decision": "Fisher_minus_Adam_and_Fisher_minus_frozen_final_positive_required_for_update_repair",
        "curvature_decision": "also_Fisher_minus_Euclidean_final_positive_required_for_curvature_specific_benefit",
        "limits": "reused_roots_development_not_confirmation_equal_KL_upper_bound_not_equal_realized_KL_no_population_guarantee"}
