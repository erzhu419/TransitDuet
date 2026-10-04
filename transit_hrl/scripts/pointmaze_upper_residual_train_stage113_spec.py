"""Forecast-anchored upper residual training above a frozen Stage112 lower branch."""
import math

from scripts import pointmaze_option_residual_stage111_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
EXPERIMENT_PROTOCOL = "pointmaze_upper_residual_train_stage113_v1"
POLICY = "forecast_anchored_upper_residual_mc_mean_frozen_lower"
RUNNER_SCRIPT = "scripts/run_pointmaze_upper_residual_train_stage113.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_upper_residual_train_stage113.py"
SOURCE_RUN = "pointmaze_option_residual_stage111_full_20261004_r1"
LOWER_RUN = "pointmaze_option_residual_train_stage112_full_20261004_r2"
ARMS = ("forecast", "learned", "learned_blinded")
CONTRASTS = (("learned", "forecast"), ("learned", "learned_blinded"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRASTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (113, 113113)
CHUNK_SIZE = 1024


def roots(*, preflight):
    return (410011,) if preflight else (410011, 410023, 410037, 410049, 410061, 410073, 410089, 410101)


def arguments(root, *, preflight):
    return source.arguments(root, preflight=preflight)


def options(*, preflight):
    return {"workers": 2 if preflight else 4, "credit_scenarios_per_batch": 2 if preflight else 16,
        "rollouts_per_scenario": 2, "evaluation_episodes": 4 if preflight else 32,
        "updates": 2 if preflight else 8}


def source_result(root):
    return ROOT / "results" / SOURCE_RUN / "cells" / f"replicate_{root}" / "result.json"


def lower_checkpoint(root, period):
    return ROOT / "results" / LOWER_RUN / "cells" / f"replicate_{root}" / "final_weights" / f"period_{period}_learned.pt"


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 113000000 if preflight else 113100000 + roots(preflight=False).index(root) * 100000
    o = options(preflight=preflight)
    rounds = [{name: [{"scenario_seed": base + 10000 * j + offset + i,
        "noise_seeds": [base + 10000 * j + offset + 2001 + 2 * i, base + 10000 * j + offset + 2002 + 2 * i]}
        for i in range(o["credit_scenarios_per_batch"])]
        for name, offset in (("credit_A", 1), ("credit_B", 1001))} for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base + 95001, base + 95001 + o["evaluation_episodes"]))}


def budget(*, preflight):
    o, h = options(preflight=preflight), arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    train_per_period = o["updates"] * 2 * o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"]
    eval_per_period = o["evaluation_episodes"] * len(ARMS)
    episodes = len(PERIODS) * (train_per_period + eval_per_period)
    upper_train_calls = sum(train_per_period * (h // p) for p in PERIODS)
    upper_eval_calls = sum(o["evaluation_episodes"] * (h // p) for p in PERIODS)
    plan_episodes = episodes
    plan_renewals = sum((train_per_period + eval_per_period) * (h // p) for p in PERIODS)
    plan_fits = sum((train_per_period + eval_per_period) * (h // p - 1) for p in PERIODS)
    plan_calls = plan_episodes * h
    score_forward = sum(o["updates"] * 2 * math.ceil(
        o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"] * (h // p) / CHUNK_SIZE) for p in PERIODS)
    fisher = sum(o["updates"] * math.ceil(
        2 * o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"] * (h // p) / CHUNK_SIZE) for p in PERIODS)
    return {"source_cell_loads": 1, "source_clone_loads": len(PERIODS), "lower_checkpoint_loads": len(PERIODS),
        "upper_branch_initializations": len(PERIODS), "training_episodes": len(PERIODS) * train_per_period,
        "evaluation_episodes": len(PERIODS) * eval_per_period, "native_episodes": episodes,
        "native_steps": episodes * h, "native_lower_calls": episodes * h,
        "native_upper_calls": upper_train_calls + upper_eval_calls, "native_network_checks": episodes,
        "native_pair_checks": len(PERIODS) * o["evaluation_episodes"],
        "scenario_pair_checks": len(PERIODS) * o["updates"] * 2 * o["credit_scenarios_per_batch"],
        "objective_checks": len(PERIODS) * o["updates"] * 2 * o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"],
        "mc_calls": len(PERIODS) * o["updates"] * 2 * o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"],
        "actor_score_forward_batches": score_forward, "actor_score_backward_batches": score_forward * 2,
        "residual_fisher_batches": fisher, "residual_kl_checks": len(PERIODS) * o["updates"],
        "residual_parameter_updates": len(PERIODS) * o["updates"], "training_freeze_checks": len(PERIODS) * o["updates"],
        "frozen_model_checks": len(PERIODS), "checkpoint_writes": 0 if preflight else len(PERIODS),
        "planning_renewals": plan_renewals, "planning_fits": plan_fits, "planning_predictions": plan_fits,
        "planning_reference_calls": plan_calls, "planning_context_calls": plan_calls}


def contract():
    return {"source": SOURCE_RUN, "lower_source": LOWER_RUN,
        "architecture": "forecast_anchored_Bernstein_residual_upper_readout390_to4_zero_initialized",
        "training": "MC_mean_upper_residual_updates_only_lower_branch_critic_std_and_upper_base_frozen",
        "representation": "alpha1_forecast_base_plus_upper_residual_no_calibration_shrinkage",
        "arms": list(ARMS), "primary_endpoints": [f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRASTS],
        "statistics": "all4_equal_root_bootstrap65536_Bonferroni4_seed113_113113_no_selection",
        "decision": "positive_CI_for_learned_minus_forecast_and_learned_minus_learned_blinded_both_periods",
        "freeze": "Stage112_learned_lower_branch_Stage111_source_upper_base_upper_std_values_and_critic",
        "artifacts": "server_final_upper_weights_only_compact_JSON_no_native_trace_pull",
        "limits": "upper_branch_only_not_full_joint_actor_critic_until_gate"}
