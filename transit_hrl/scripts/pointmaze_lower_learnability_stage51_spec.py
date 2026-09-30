"""Frozen native feedback, behavioral-cloning and independent-credit controls."""

import numpy as np
from scripts import pointmaze_fisher_direction_stage50_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_lower_learnability_stage51_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_lower_learnability_stage51.py"
POLICY = "lower_learnability"
METHODS = ("task_clock",)
MODES, METRICS = previous.MODES, previous.METRICS
POLICIES = ("frozen", "teacher_waypoint", "teacher_task", "clone_waypoint", "sham_waypoint", "clone_task", "sham_task")
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments, warmup_iterations = previous.source_result, previous.roots, previous.arguments, previous.warmup_iterations
BC_MINIBATCH, BC_LR, ACTION_LIMIT = 1024, 3e-4, .95
Q_DIAGONAL, R_DIAGONAL = (25., 25., 1., 1.), (1., 1.)
RETURN_PAIRS = (("teacher_waypoint", "frozen"), ("teacher_task", "frozen"),
                ("clone_waypoint", "frozen"), ("clone_task", "frozen"),
                ("clone_waypoint", "sham_waypoint"), ("clone_task", "sham_task"))
ENDPOINTS = (*(f"{a}_minus_{b}" for a, b in RETURN_PAIRS), "mc_independent_cosine", "gae_independent_cosine")
CI_FAMILY_SIZE = len(ENDPOINTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (51, 51051)


def options(*, preflight):
    return {"batch_paths": 2 if preflight else 8, "evaluation_paths": 2 if preflight else 16,
            "bc_epochs": 4 if preflight else 32, "workers": 1 if preflight else 8}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 11_890_000 if preflight else 11_900_000 + index * 10000
    opt = options(preflight=preflight)
    return {"A": list(range(base + 1, base + 1 + opt["batch_paths"])),
            "B": list(range(base + 1001, base + 1001 + opt["batch_paths"])),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([51, root, seed, 51017]).generate_state(1)[0])


def shuffle_seed(root, goal):
    return int(np.random.SeedSequence([51, root, 51023, int(goal == "task")]).generate_state(1)[0])


def label_seed(root, goal):
    return int(np.random.SeedSequence([51, root, 51029, int(goal == "task")]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    sampled = phase in ("A", "B") or mode == "lower_sampled"
    return {"sample": phase in ("A", "B"), "upper_sample": False, "gate_sample": False, "lower_sample": sampled,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([51, root, seed, 51019]).generate_state(1)[0]) if sampled else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    fitting = 2 * opt["batch_paths"]
    evaluate = len(POLICIES) * len(MODES) * opt["evaluation_paths"]
    return {"fitting_primitive_steps": fitting * horizon, "evaluation_primitive_steps": evaluate * horizon,
            "total_primitive_steps": (fitting + evaluate) * horizon, "native_trace_audits": fitting + evaluate}


def contrasts(means, credit):
    returns = means["deterministic"]
    result = {f"{a}_minus_{b}": returns[a]["episode_return"] - returns[b]["episode_return"] for a, b in RETURN_PAIRS}
    result.update(mc_independent_cosine=credit["mc"]["cosine"], gae_independent_cosine=credit["gae"]["cosine"])
    return result


def contract():
    return {"source_protocol": "pointmaze_critic_clock_stage42_v1", "source_method": "task_clock",
        "source_iteration": "warmup16_full_warmup2_preflight_not_post_actor_update",
        "policies": list(POLICIES), "feedback": "continuous_LQR_unit_mass_double_integrator_scipy_CARE_no_native_model_fit",
        "Q_diagonal": list(Q_DIAGONAL), "R_diagonal": list(R_DIAGONAL), "action_limit": ACTION_LIMIT,
        "teacher_inputs": "waypoint_error_or_visible_current_target_minus_position_velocity_current_force_only",
        "task_teacher": "bypasses_waypoint_diagnostic_not_hierarchical_policy_claim",
        "action": "clip_requested_normalized_torque_then_atanh_Gaussian_mean_source_std_unchanged",
        "cloning": "original_lower_MLP_mean_raw_action_MSE_on_frozen_A_states_only",
        "bc_lr": BC_LR, "bc_minibatch": BC_MINIBATCH, "sham": "fixed_A_label_permutation_same_optimizer_budget_shuffle",
        "selection": "fixed_final_no_early_stopping_eval_selection_gain_search_root_exclusion_or_seed_extension",
        "frozen": "all_other_networks_source_std_and_original_RL_optimizers_unchanged",
        "credit": "A_normalized_MC_or_original_GAE_plus_entropy_vs_B_reward_only_full_episode_LOO_MC_gradient",
        "credit_scaling": "A_mean_logp_advantage_B_sum_logp_raw_LOO_RTG_per_episode",
        "independence": "disjoint_A_B_evaluation_paths_same_source_actor_critic",
        "secondary_credit": "A_negative_BC_loss_gradient_vs_B_reward_only_gradient",
        "primary_mode": "deterministic", "secondary_mode": "lower_sampled",
        "endpoints": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
        "interval": "two_sided_percentile_Bonferroni8_equal_root_paired_means",
        "decision": "waypoint_teacher_positive_for_plan_headroom_clone_vs_frozen_and_sham_positive_for_same_network_learnability",
        "limits": "conditional_development_reused_roots_teacher_failure_not_ceiling_BC_failure_not_unlearnability_cosine_not_finite_return_proof"}
