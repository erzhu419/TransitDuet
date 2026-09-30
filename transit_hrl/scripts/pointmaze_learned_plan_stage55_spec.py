"""Native learned residual-plan upper and velocity-conditioned lower PPO."""

import numpy as np
from scripts import pointmaze_forecast_tracking_stage54_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_learned_plan_stage55_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_learned_plan_stage55.py"
POLICY, METHODS = "learned_plan", ("task_clock",)
POLICIES = ("frozen", "teacher", "clone", "sham", "lower_ppo", "joint_ppo")
PERIODS, MODES, METRICS = previous.PERIODS, previous.MODES, previous.METRICS
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result, roots, arguments, warmup_iterations = previous.source_result, previous.roots, previous.arguments, previous.warmup_iterations
BC_LR, BC_MINIBATCH, PLAN_BASIS = 3e-4, 1024, 3
RETURN_PAIRS = (("teacher", "frozen"), ("clone", "frozen"), ("clone", "sham"),
                ("lower_ppo", "clone"), ("joint_ppo", "lower_ppo"), ("joint_ppo", "frozen"), ("joint_ppo", "clone"))
ENDPOINTS = tuple(f"period{p}:{a}_minus_{b}" for p in PERIODS for a, b in RETURN_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (55, 55055)


def options(*, preflight):
    return {"fitting_paths": 2 if preflight else 32, "label_paths": 2 if preflight else 8,
            "evaluation_paths": 2 if preflight else 16, "bc_epochs": 4 if preflight else 64,
            "critic_warmup_iterations": 2 if preflight else 16,
            "learning_iterations": 2 if preflight else 32, "rollouts_per_iteration": 2 if preflight else 8,
            "workers": 1 if preflight else 8}


def seed_roles(root, *, preflight):
    index, opt = roots(preflight=preflight).index(root), options(preflight=preflight)
    base = 14_090_000 if preflight else 14_100_000 + index * 10000
    counts = {"fitting": opt["fitting_paths"], "labels": opt["label_paths"],
              "warmup": opt["critic_warmup_iterations"] * opt["rollouts_per_iteration"],
              "training": opt["learning_iterations"] * opt["rollouts_per_iteration"], "evaluation": opt["evaluation_paths"]}
    return {k: list(range(base + offset, base + offset + counts[k])) for k, offset in
            (("fitting", 1), ("labels", 1001), ("warmup", 2001), ("training", 3001), ("evaluation", 5001))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([55, root, seed, 55017]).generate_state(1)[0])


def shuffle_seed(root, period, iteration=0):
    return int(np.random.SeedSequence([55, root, period, iteration, 55023]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    lower = phase in ("warmup", "train") or mode == "lower_sampled"
    return {"sample": phase in ("labels", "warmup", "train"), "upper_sample": phase in ("warmup", "train"), "gate_sample": False,
            "lower_sample": lower, "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([55, root, seed, 55019]).generate_state(1)[0]) if lower else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    train = 2 * opt["learning_iterations"] * opt["rollouts_per_iteration"]
    evaluate = len(POLICIES) * len(MODES) * opt["evaluation_paths"]
    episodes = {"labels": opt["label_paths"], "warmup": opt["critic_warmup_iterations"] * opt["rollouts_per_iteration"],
                "train": train, "eval": evaluate}
    return {"primitive_steps": {k: len(PERIODS) * n * horizon for k, n in episodes.items()},
        "total_primitive_steps": len(PERIODS) * sum(episodes.values()) * horizon,
        "native_trace_audits": len(PERIODS) * sum(episodes.values()),
        "upper_calls": {k: sum(horizon // p for p in PERIODS) * n for k, n in episodes.items()},
        "fitting_rows": opt["fitting_paths"] * (horizon - previous.FORECAST_STEPS),
        "fitting_observations": opt["fitting_paths"] * (horizon + 1),
        "supervised_steps": len(PERIODS) * 2 * opt["bc_epochs"] * int(np.ceil(opt["label_paths"] * horizon / BC_MINIBATCH))}


def contrasts(means):
    return {f"period{p}:{a}_minus_{b}": means[str(p)]["deterministic"][a]["episode_return"]
            - means[str(p)]["deterministic"][b]["episode_return"] for p in PERIODS for a, b in RETURN_PAIRS}


def contract():
    return {"source": "Stage42_task_clock_warmup16_full_2_preflight", "policies": list(POLICIES), "periods": list(PERIODS),
        "forecast": "unchanged_Stage54_ridge_fit_once_on_fresh_disjoint_fitting_paths_not_eval_paths",
        "upper": "four_Gaussian_actions_existing_anchored_quadratic_Bernstein_residual_on_clipped_ridge_plan",
        "residual_scale": "inherited_maximum_subgoal_delta_no_tuning", "upper_initialization": "inherited_MLP_trunk_zero_four_output_mean_repeated_source_std",
        "lower": "inherited_MLP_plus_two_zero_velocity_input_columns_native_Gaussian_actor_no_feedback_at_learned_deployment",
        "critic": "base390_plus_plan_velocity2_plus_causal_clock2_inherited_clock_weights_zero_velocity_columns",
        "labels": "deterministic_native_teacher_trajectories_each_period_only_disjoint_label_paths",
        "bc": "bounded_action_tanh_mean_MSE_Adam_net_only_source_std_fixed_final_epochs",
        "bc_lr": BC_LR, "bc_minibatch": BC_MINIBATCH, "sham": "fixed_label_permutation_same_data_and_optimizer_budget",
        "warmup": "shared_critic_only_calibration_on_cloned_policy_current_plan_inputs_actor_std_frozen_before_PPO_fork",
        "training": "common_calibrated_clone_standard_shared_SMDP_PPO_native_task_option_reward_same_inherited_hyperparameters",
        "lower_ppo": "upper_actor_and_value_fixed_lower_actor_value_updated",
        "joint_ppo": "upper_actor_value_and_lower_actor_value_updated_no_gate_or_custom_PG",
        "sampling": "both_PPO_arms_sample_upper_lower_during_training_same_seed_streams_deterministic_upper_at_eval",
        "clock": "current_option_age_over100_and_remaining_episode_fraction_no_future_data",
        "credit": "lower_native_reward_upper_discounted_sum_native_reward_minus_equal_call_cost",
        "selection": "fixed_final_no_validation_checkpoint_period_mode_gain_root_or_seed_selection",
        "primary_mode": "deterministic", "secondary_mode": "lower_sampled", "endpoints": list(ENDPOINTS),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
        "interval": "two_sided_percentile_Bonferroni14_equal_root_paired_means",
        "decision": "clone_vs_frozen_and_sham_positive_both_periods_for_learnability_joint_vs_frozen_clone_lower_ppo_positive_both_for_joint_gain",
        "limits": "residual_HRL_on_fixed_learned_forecast_with_teacher_initialization_conditional_development_not_frequency_or_promotion_confirmation"}
