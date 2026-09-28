"""Frozen development test of jointly trained, proposal-free plan renewal."""

from pathlib import Path

from scripts import pointmaze_root_response_stage33_spec as source

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PROTOCOL = "pointmaze_joint_renewal_stage35_v1"
POLICY = source.POLICY
OPTIMIZER_ROOTS = source.OPTIMIZER_ROOTS
PREFLIGHT_ROOTS = source.PREFLIGHT_ROOTS
METHODS = ("learned_history", "learned_current", "fixed50", "fixed100")
RUNNER_SCRIPT = "scripts/run_pointmaze_joint_renewal_stage35.py"
CHECK_STEPS = 25
MAX_AGE_STEPS = 100
CALL_COST = 1.0
ENDPOINTS = ("return:fixed50", "ise:fixed50", "calls:fixed50",
             "utility:fixed100", "utility:learned_current")
BOOTSTRAP_DRAWS = 65536
BOOTSTRAP_SEED = (35, 35039)


def roots(*, preflight):
    return PREFLIGHT_ROOTS if preflight else OPTIMIZER_ROOTS


def options(*, preflight):
    return {"iterations": 2 if preflight else 128,
            "rollouts_per_iteration": 1 if preflight else 8,
            "selection_paths": 2 if preflight else 8,
            "evaluation_paths": 2 if preflight else 32,
            "selection_interval": 1 if preflight else 32,
            "workers": 1 if preflight else 8,
            "call_cost": CALL_COST, "check_steps": CHECK_STEPS,
            "max_age_steps": MAX_AGE_STEPS}


def seed_roles(root, *, preflight):
    roster = roots(preflight=preflight)
    if root not in roster:
        raise ValueError("unregistered Stage-35 root")
    base = 6_390_000 if preflight else 6_400_000 + roster.index(root) * 10000
    opt = options(preflight=preflight)
    return {"training": list(range(base + 1, base + 1 + opt["iterations"] * opt["rollouts_per_iteration"])),
            "selection": list(range(base + 2001, base + 2001 + opt["selection_paths"])),
            "evaluation": list(range(base + 3001, base + 3001 + opt["evaluation_paths"]))}


def source_result(root, *, preflight):
    run = ("pointmaze_root_response_stage33_v1_preflight_20260928_r1" if preflight
           else "pointmaze_root_response_stage33_v1_controllers_20260928_r1")
    filename = "controller.json" if preflight else "result.json"
    return ROOT / "results" / run / "cells" / POLICY / f"replicate_{root}" / filename


def budget(*, preflight):
    opt = options(preflight=preflight)
    horizon = source.arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    training = opt["iterations"] * opt["rollouts_per_iteration"] * horizon
    selection = (1 + opt["iterations"] // opt["selection_interval"]) * opt["selection_paths"] * horizon
    evaluation = opt["evaluation_paths"] * horizon
    return {"training_primitive_steps": training, "selection_primitive_steps": selection,
            "evaluation_primitive_steps": evaluation, "factual_replay_primitive_steps": horizon,
            "total_primitive_steps": training + selection + evaluation + horizon}


def contract():
    return {"source_protocol": source.EXPERIMENT_PROTOCOL, "methods": list(METHODS),
            "training": "joint_on_policy_upper_lower_and_renewal_smdp_ppo",
            "initialization": "same_stage33_weights_per_root_fresh_optimizers",
            "gate_input": "causal_physical_target_error_measurement_history_waypoint_error_age_remaining_time",
            "current_control": "repeat_current_measurement_to_same_history_width_gate_only",
            "gate_check_steps": CHECK_STEPS, "max_plan_age_steps": MAX_AGE_STEPS,
            "upper_call_cost_in_reward_units": CALL_COST,
            "upper_and_gate_reward": "native_dense_reward_minus_actual_call_cost",
            "lower_reward": "unchanged_goal_conditioned_intrinsic_reward",
            "candidate_previews": 0, "lower_feedback": "every_primitive_step",
            "checkpoint_selection": "maximum_selection_return_minus_call_cost_then_minimum_ise",
            "checkpoint_candidates": "initial_and_every_32_iterations_full_or_every_iteration_preflight",
            "primary_endpoints": list(ENDPOINTS), "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": list(BOOTSTRAP_SEED), "familywise_alpha": .05,
            "interval": "two_sided_percentile_bonferroni_5_endpoints",
            "statistical_unit": "optimizer_seed_root", "root_weighting": "equal",
            "gate": "all_5_adjusted_lower_bounds_strictly_positive",
            "evidence_role": "conditional_joint_training_development_not_independent_confirmation",
            "root_exclusion": "forbidden", "sequential_extension": "forbidden"}
