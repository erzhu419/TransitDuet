"""Train normal joint and zero-execution lower policies from one clone."""

import numpy as np
from scripts import pointmaze_upper_execution_stage56_spec as previous

ROOT = previous.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_matched_upper_stage57_v1"
RUNNER_SCRIPT = "scripts/run_pointmaze_matched_upper_stage57.py"
POLICY, METHODS = "matched_upper", ("task_clock",)
POLICIES, TRAIN_POLICIES = ("clone", "zero_train", "joint_ppo"), ("zero_train", "joint_ppo")
PERIODS, MODES, METRICS = previous.PERIODS, previous.MODES, previous.METRICS
roots, arguments = previous.roots, previous.arguments
SOURCE_FULL_RUN, SOURCE_PREFLIGHT_RUN = previous.SOURCE_FULL_RUN, previous.SOURCE_PREFLIGHT_RUN
source_result = previous.source_result
SOURCE_SPEC = previous.previous
RETURN_PAIRS = (("joint_ppo", "zero_train"), ("joint_ppo", "clone"), ("zero_train", "clone"))
ENDPOINTS = tuple(f"period{p}:{a}_minus_{b}" for p in PERIODS for a, b in RETURN_PAIRS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (57, 57057)


def options(*, preflight):
    return {"evaluation_paths": 2 if preflight else 16, "critic_warmup_iterations": 2 if preflight else 16,
            "learning_iterations": 2 if preflight else 32, "rollouts_per_iteration": 2 if preflight else 8,
            "workers": 1 if preflight else 8}


def execution(policy):
    return "normal" if policy == "joint_ppo" else "zero_residual"


def seed_roles(root, *, preflight):
    index, opt = roots(preflight=preflight).index(root), options(preflight=preflight)
    base = 16_090_000 if preflight else 16_100_000 + index * 10000
    counts = {"warmup": opt["critic_warmup_iterations"] * opt["rollouts_per_iteration"],
              "training": opt["learning_iterations"] * opt["rollouts_per_iteration"], "evaluation": opt["evaluation_paths"]}
    return {k: list(range(base + offset, base + offset + counts[k])) for k, offset in
            (("warmup", 2001), ("training", 3001), ("evaluation", 5001))}


def policy_seed(root, seed):
    return int(np.random.SeedSequence([57, root, seed, 57017]).generate_state(1)[0])


def shuffle_seed(root, period, iteration, *, phase, level):
    return int(np.random.SeedSequence([57, root, period, iteration,
        int(phase == "train"), int(level == "lower"), 57023]).generate_state(1)[0])


def rollout_arguments(root, seed, *, phase, mode):
    training = phase in ("warmup", "train")
    lower = training or mode == "lower_sampled"
    return {"sample": training, "upper_sample": training, "gate_sample": False, "lower_sample": lower,
            "gate_seed": None,
            "lower_seed": int(np.random.SeedSequence([57, root, seed, 57019]).generate_state(1)[0]) if lower else None}


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n = {"warmup": len(TRAIN_POLICIES) * opt["critic_warmup_iterations"] * opt["rollouts_per_iteration"],
         "train": len(TRAIN_POLICIES) * opt["learning_iterations"] * opt["rollouts_per_iteration"],
         "eval": len(POLICIES) * len(MODES) * opt["evaluation_paths"]}
    return {"primitive_steps": {k: len(PERIODS) * v * horizon for k, v in n.items()},
        "upper_calls": {k: sum(horizon // p for p in PERIODS) * v for k, v in n.items()},
        "total_primitive_steps": len(PERIODS) * sum(n.values()) * horizon,
        "native_trace_audits": len(PERIODS) * sum(n.values()),
        "checkpoint_loads": len(PERIODS), "forecaster_loads": 1, "new_forecaster_fits": 0,
        "supervised_steps": 0, "verification_primitive_steps": 0}


def contrasts(means):
    return {f"period{p}:{a}_minus_{b}": means[str(p)]["deterministic"][a]["episode_return"]
            - means[str(p)]["deterministic"][b]["episode_return"] for p in PERIODS for a, b in RETURN_PAIRS}


def contract():
    return {"source_protocol": SOURCE_SPEC.EXPERIMENT_PROTOCOL, "source_runs": [SOURCE_PREFLIGHT_RUN, SOURCE_FULL_RUN],
        "initialization": "fixed_Stage55_clone_final_each_root_period_same_four_networks_and_optimizer_states",
        "forecaster": "same_saved_Stage55_forecaster_no_fitting_or_BC", "policies": list(POLICIES), "periods": list(PERIODS),
        "zero_train": "infer_upper_but_execute_zero_residual_from_first_warmup_and_training_rollout_upper_actor_fixed_lower_PPO_learned",
        "joint_ppo": "normal_residual_execution_shared_native_SMDP_PPO_upper_and_lower_learned",
        "calibration": "each_arm_same_budget_critic_only_on_its_own_execution_distribution_both_actors_std_frozen",
        "optimizer": "unchanged_inherited_PPO_hyperparameters_and_matched_lower_updates_upper_updates_only_for_joint",
        "pairing": "fresh_shared_environment_and_stepwise_lower_noise_seeds_same_initial_upper_proposals_not_identical_lower_trajectories",
        "credit": "lower_native_task_option_reward_upper_discounted_native_reward_minus_equal_call_cost",
        "deployment": "fixed_final_same_execution_rule_as_training_no_analytic_feedback_no_checkpoint_selection",
        "audit": "actual_proposed_and_executed_actions_native_credit_plan_velocity_reconstruction_rollout_networks_frozen_parent_updates_nonnull",
        "primary_mode": "deterministic", "secondary_mode": "lower_sampled", "endpoints": list(ENDPOINTS),
        "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": list(BOOTSTRAP_SEED),
        "interval": "two_sided_percentile_Bonferroni6_equal_root_paired_means",
        "decision": "matched_upper_gate_joint_vs_zero_positive_both_periods_training_gain_gate_joint_vs_clone_positive_both_periods",
        "selection": "no_checkpoint_period_mode_root_seed_or_budget_selection_after_outcomes",
        "limits": "reused_training_roots_teacher_initialized_fixed_forecast_development_not_frequency_promotion_OOD_or_equal_total_FLOPs"}
