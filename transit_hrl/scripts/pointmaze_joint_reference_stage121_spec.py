"""Fresh, equal-environment-budget joint PPO reference-tracking development."""

import math

from scripts import pointmaze_upper_residual_train_stage113_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
roots, arguments = source.roots, source.arguments
source_result, lower_checkpoint = source.source_result, source.lower_checkpoint
SOURCE_RUN, LOWER_RUN = source.SOURCE_RUN, source.LOWER_RUN
EXPERIMENT_PROTOCOL = "pointmaze_joint_reference_stage121_v1"
POLICY = "joint_smdp_ppo_frozen_teacher_bounded_reference_tracking"
RUNNER_SCRIPT = "scripts/run_pointmaze_joint_reference_stage121.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_joint_reference_stage121.py"
METHODS = {"flat": "lower", "forecast": "lower", "joint": "upper_lower"}
ARMS = ("source_flat", "source_forecast", "flat", "forecast", "joint", "joint_blinded", "upper_transfer")
CONTRASTS = (("joint", "flat"), ("joint", "forecast"), ("joint", "joint_blinded"),
             ("joint", "source_forecast"), ("upper_transfer", "forecast"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRASTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (121, 121121)
REFERENCE_LIMIT, RESIDUAL_LIMIT, MINIMUM_GAIN = .05, .05, .5
UPPER_STD, LEARNING_RATE, EPOCHS, MINIBATCH = .15, 3e-4, 2, 1024
UPPER_STATE_DIM, LOWER_STATE_DIM, UPPER_ACTION_DIM = 392, 402, 8


def options(*, preflight):
    return {"workers": 2 if preflight else 4, "updates": 2 if preflight else 8,
            "scenarios_per_update": 2 if preflight else 8, "rollouts_per_scenario": 2,
            "evaluation_episodes": 4 if preflight else 32}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 121000000 if preflight else 121100000 + index * 100000
    o = options(preflight=preflight)
    return {"training_rounds": [[{"scenario_seed": base + j * 10000 + i + 1,
        "noise_seeds": [base + j * 10000 + 2001 + i * 2, base + j * 10000 + 2002 + i * 2]}
        for i in range(o["scenarios_per_update"])] for j in range(o["updates"])],
        "native_evaluation": list(range(base + 95001, base + 95001 + o["evaluation_episodes"]))}


def optimizer_seed(root, period, iteration):
    return root + 121000000 + period * 100 + iteration


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    paths = o["scenarios_per_update"] * o["rollouts_per_scenario"]
    train = o["updates"] * paths
    evaluation = o["evaluation_episodes"]
    episodes = len(PERIODS) * (len(METHODS) * train + len(ARMS) * evaluation)
    lower_steps = len(PERIODS) * len(METHODS) * o["updates"] * EPOCHS * math.ceil(paths * h / MINIBATCH)
    upper_steps = sum(o["updates"] * EPOCHS * math.ceil(paths * (h // p) / MINIBATCH) for p in PERIODS)
    return {"source_cell_loads": 1, "source_clone_loads": len(PERIODS),
        "lower_checkpoint_loads": len(PERIODS), "training_episodes": len(PERIODS) * len(METHODS) * train,
        "evaluation_episodes": len(PERIODS) * len(ARMS) * evaluation,
        "native_episodes": episodes, "native_steps": episodes * h, "native_lower_calls": episodes * h,
        "native_upper_calls": sum((train + 2 * evaluation) * (h // p) for p in PERIODS),
        "native_donor_response_calls": 2 * episodes * h,
        "planning_renewals": sum((2 * train + 5 * evaluation) * (h // p) for p in PERIODS),
        "planning_fits": sum((2 * train + 5 * evaluation) * (h // p - 1) for p in PERIODS),
        "planning_reference_calls": len(PERIODS) * (2 * train + 5 * evaluation) * h,
        "lower_actor_optimizer_steps": lower_steps, "lower_value_optimizer_steps": lower_steps,
        "upper_actor_optimizer_steps": upper_steps, "upper_value_optimizer_steps": upper_steps,
        "evaluation_pair_groups": len(PERIODS) * evaluation,
        "ppo_updates": len(PERIODS) * len(METHODS) * o["updates"],
        "credit_checks": len(PERIODS) * len(METHODS) * train,
        "checkpoint_writes": 0 if preflight else len(PERIODS) * len(METHODS)}


def contract():
    return {"source": SOURCE_RUN, "lower_source": LOWER_RUN,
        "architecture": "upper392_to8_zero_mean_Bernstein5_lower402_to2_zero_bounded_residual",
        "feedback": "all392_lower_feedback_plus4_causal_forecast_advice_plus4_curve_minus_forecast_plus2_clock",
        "execution": "frozen_Stage112_teacher_mean_plus_bounded_Stage120_donor_reference_response_plus_bounded_learned_residual",
        "reference_limit": REFERENCE_LIMIT, "residual_limit": RESIDUAL_LIMIT,
        "initial_policy": "zero_upper_mean_and_lower_readout_exact_source_forecast_mean_std",
        "training": "shared_SMDP_PPO_both_actors_and_reward_critics_gamma1_lambda1_true_option_durations",
        "critic_initialization": "remaining_steps_upper_bound_of_exp_minus_distance_return_no_fitted_evaluation_prior",
        "ppo": {"epochs": EPOCHS, "minibatch": MINIBATCH, "learning_rate": LEARNING_RATE,
                "clip_ratio": .2, "entropy": 0., "upper_std": UPPER_STD, "lower_std": "source_frozen"},
        "budget": "equal_fresh_native_paths_for_flat_forecast_joint_not_equal_parameters_or_total_compute",
        "arms": list(ARMS), "endpoints": list(ENDPOINTS), "minimum_episode_gain": MINIMUM_GAIN,
        "statistics": "all8_equal_root_bootstrap65536_Bonferroni10_fresh_rosters_fixed_update8",
        "decision": "both_period_joint_minus_trained_flat_forecast_CI_above0.5_and_own_blinded_source_forecast_CI_above0",
        "freeze": "teacher_function_forecaster_task_std_all_source_weights_no_stage120_gate_change",
        "artifacts": "final_inference_weights_server_only_scalar_JSON_no_native_trace_writes",
        "limits": "new_joint_control_development_not_Stage120_confirmation_promotion_strict_routing_or_domain_general_proof_Stage67_HOLD"}
