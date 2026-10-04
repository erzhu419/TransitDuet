"""Matched option-versus-suffix credit for the complete Stage116 plan head."""

import math

from scripts import pointmaze_upper_full_plan_train_stage116_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
EXPERIMENT_PROTOCOL = "pointmaze_upper_suffix_credit_stage118_v1"
POLICY = "complete_plan_head_matched_option_versus_decision_suffix_credit"
RUNNER_SCRIPT = "scripts/run_pointmaze_upper_suffix_credit_stage118.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_upper_suffix_credit_stage118.py"
SOURCE_RUN, LOWER_RUN = source.SOURCE_RUN, source.LOWER_RUN
PREREQUISITE_RUN = "pointmaze_local_plan_gain_stage117_full_20261004_r1"
METHODS = ("option", "suffix")
ARMS = ("forecast", *METHODS, "suffix_blinded")
CONTRASTS = (("suffix", "forecast"), ("suffix", "option"))
ENDPOINTS = tuple(f"{period}/{a}_minus_{b}" for period in PERIODS for a, b in CONTRASTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CHUNK_SIZE = 65536, (118, 118118), source.CHUNK_SIZE
roots, arguments, options = source.roots, source.arguments, source.options
source_result, lower_checkpoint = source.source_result, source.lower_checkpoint


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 118000000 if preflight else 118100000 + index * 100000
    o = options(preflight=preflight)
    rounds = [{name: [{"scenario_seed": base + j * 10000 + offset + i,
        "noise_seeds": [base + j * 10000 + offset + 2001 + 2 * i,
                        base + j * 10000 + offset + 2002 + 2 * i]}
        for i in range(o["credit_scenarios_per_batch"])]
        for name, offset in (("credit_A", 1), ("credit_B", 1001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base + 95001, base + 95001 + o["evaluation_episodes"]))}


def prerequisite_summary():
    return ROOT / "results" / PREREQUISITE_RUN / "qualification_summary.json"


def budget(*, preflight):
    o = options(preflight=preflight)
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    periods, methods = len(PERIODS), len(METHODS)
    train = methods * o["updates"] * 2 * o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"]
    evaluate = o["evaluation_episodes"] * len(ARMS)
    episodes = periods * (train + evaluate)
    forward = methods * sum(o["updates"] * 2 * math.ceil(
        o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"] * (horizon // p) / CHUNK_SIZE)
        for p in PERIODS)
    fisher = methods * sum(o["updates"] * math.ceil(
        2 * o["credit_scenarios_per_batch"] * o["rollouts_per_scenario"] * (horizon // p) / CHUNK_SIZE)
        for p in PERIODS)
    updates = periods * methods * o["updates"]
    fits = sum((train + evaluate) * (horizon // p - 1) for p in PERIODS)
    return {"source_cell_loads": 1, "source_clone_loads": periods, "lower_checkpoint_loads": periods,
        "upper_branch_initializations": periods * methods, "training_episodes": periods * train,
        "evaluation_episodes": periods * evaluate, "native_episodes": episodes,
        "native_steps": episodes * horizon, "native_lower_calls": episodes * horizon,
        "native_upper_calls": sum((train + methods * o["evaluation_episodes"]) * (horizon // p) for p in PERIODS),
        "native_network_checks": episodes, "native_pair_checks": periods * o["evaluation_episodes"],
        "scenario_pair_checks": periods * methods * o["updates"] * 2 * o["credit_scenarios_per_batch"],
        "initial_credit_pair_checks": periods * 2 * o["credit_scenarios_per_batch"],
        "objective_checks": periods * train, "mc_calls": periods * train,
        "actor_score_forward_batches": forward, "actor_score_backward_batches": forward * 2,
        "residual_fisher_batches": fisher, "residual_kl_checks": updates,
        "residual_parameter_updates": updates, "training_freeze_checks": updates,
        "frozen_model_checks": periods, "checkpoint_writes": 0 if preflight else periods * methods,
        "planning_renewals": sum((train + evaluate) * (horizon // p) for p in PERIODS),
        "planning_fits": fits, "planning_predictions": fits,
        "planning_reference_calls": episodes * horizon, "planning_context_calls": episodes * horizon}


def contract():
    return {"source": SOURCE_RUN, "lower_source": LOWER_RUN, "prerequisite": PREREQUISITE_RUN,
        "architecture": "unchanged_Stage116_complete_readout390_to8_Bernstein_basis5_alpha1",
        "factor": "option_reward_versus_undiscounted_decision_to_episode_end_reward_before_leave_other_out",
        "pairing": "same_initial_weights_round_scenarios_action_noise_rosters_update_count_and_Fisher_radius",
        "credit": "reverse_cumulative_option_rewards_aligned_to_each_upper_decision_no_episode_score_broadcast",
        "training": "same_score_gradient_Fisher_update_final8_updates_no_checkpoint_selection",
        "freeze": "Stage112_final_learned_lower_Stage111_source_donor_upper_std_values_critic_and_plan_clock",
        "evaluation": "new_scenarios_final_deterministic_upper_mean_stochastic_lower_common_innovations",
        "arms": list(ARMS), "primary_endpoints": list(ENDPOINTS),
        "statistics": "all8_equal_root_bootstrap65536_Bonferroni4_no_root_exclusion",
        "decision": "positive_CI_suffix_minus_forecast_and_suffix_minus_option_both_periods",
        "blinding": "suffix_blinded_matches_forecast_execution_not_a_duplicate_primary_endpoint",
        "artifacts": "compact_JSON_final_upper_weights_server_only_no_raw_native_trace_writes",
        "limits": "learned_upper_credit_intervention_not_joint_actor_critic_or_learned_promotion"}
