"""Native common-noise action-coordinate gradients for a learned upper plan."""

import math

from scripts import pointmaze_upper_suffix_credit_stage118_spec as source
from scripts import pointmaze_local_plan_gain_stage117_spec as probe

ROOT, PERIODS = source.ROOT, source.PERIODS
EXPERIMENT_PROTOCOL = "pointmaze_native_plan_gradient_stage119_v1"
POLICY = "deterministic_upper_mean_native_plan_gradient_full_suffix"
RUNNER_SCRIPT = "scripts/run_pointmaze_native_plan_gradient_stage119.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_native_plan_gradient_stage119.py"
SOURCE_RUN, LOWER_RUN = source.SOURCE_RUN, source.LOWER_RUN
REFERENCE_RUN = "pointmaze_upper_suffix_credit_stage118_full_20261005_r1"
METHODS = ("native_fd",)
ARMS = ("forecast", "native_fd", "stage118_suffix", "native_fd_blinded")
CONTRASTS = (("native_fd", "forecast"), ("native_fd", "stage118_suffix"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRASTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, CHUNK_SIZE = 65536, (119, 119119), source.CHUNK_SIZE
EPSILON, ACTION_DIM, PANELS, DIRECTIONS = probe.EPSILON, probe.ACTION_DIM, probe.PANELS, probe.DIRECTIONS
VARIANTS = ("zero", *DIRECTIONS)
roots, arguments, source_result, lower_checkpoint = source.roots, source.arguments, source.source_result, source.lower_checkpoint


def options(*, preflight):
    return {"workers": 2 if preflight else 8, "queries_per_update": 2 if preflight else 12,
        "updates": 2 if preflight else 8, "evaluation_episodes": 4 if preflight else 32}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 119000000 if preflight else 119100000 + index * 100000
    o, horizon = options(preflight=preflight), arguments(root, preflight=preflight).horizon
    rounds = []
    for j in range(o["updates"]):
        groups = {}
        for k, period in enumerate(PERIODS):
            b = base + j * 10000 + k * 1000
            groups[str(period)] = [{"scenario_seed": b + 1 + i,
                "start": (0 if i == 0 else horizon - period) if preflight else
                    ((j * o["queries_per_update"] + i) % (horizon // period)) * period,
                "prefix_noise_seed": b + 2001 + i,
                "suffix_noise_seeds": {"A": b + 4001 + i, "B": b + 6001 + i}}
                for i in range(o["queries_per_update"])]
        rounds.append(groups)
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base + 95001, base + 95001 + o["evaluation_episodes"]))}


def reference_checkpoint(root, period):
    return ROOT / "results" / REFERENCE_RUN / "cells" / f"replicate_{root}" / "final_weights" / f"period_{period}_suffix_upper.pt"


def budget(*, preflight):
    o = options(preflight=preflight)
    h, periods = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon, len(PERIODS)
    queries = periods * o["updates"] * o["queries_per_update"]
    train = o["updates"] * o["queries_per_update"] * len(PANELS) * len(VARIANTS)
    evaluate = o["evaluation_episodes"] * len(ARMS)
    episodes = periods * (train + evaluate)
    updates = periods * o["updates"]
    return {"source_cell_loads": 1, "source_clone_loads": periods, "lower_checkpoint_loads": periods,
        "reference_checkpoint_loads": periods, "upper_branch_initializations": periods,
        "training_queries": queries, "training_episodes": periods * train, "evaluation_episodes": periods * evaluate,
        "native_episodes": episodes, "native_steps": episodes * h, "native_lower_calls": episodes * h,
        "native_upper_calls": sum((train + 2 * o["evaluation_episodes"]) * (h // p) for p in PERIODS),
        "native_pair_checks": queries + periods * o["evaluation_episodes"], "native_network_checks": periods * evaluate,
        "prefix_pair_checks": queries * (len(PANELS) * len(VARIANTS) - 1),
        "external_pair_checks": queries * (len(PANELS) * len(VARIANTS) - 1),
        "innovation_pair_checks": queries * len(PANELS) * (len(VARIANTS) - 1),
        "suffix_credit_checks": periods * train, "training_policy_freeze_checks": queries,
        "actor_pullback_forward_batches": updates * len(PANELS), "actor_pullback_backward_batches": updates * len(PANELS),
        "residual_fisher_batches": updates * math.ceil(2 * o["queries_per_update"] / CHUNK_SIZE),
        "residual_kl_checks": updates, "residual_parameter_updates": updates, "training_freeze_checks": updates,
        "frozen_model_checks": periods, "checkpoint_writes": 0 if preflight else periods,
        "planning_renewals": sum((train + evaluate) * (h // p) for p in PERIODS),
        "planning_fits": sum((train + evaluate) * (h // p - 1) for p in PERIODS),
        "planning_predictions": sum((train + evaluate) * (h // p - 1) for p in PERIODS),
        "planning_reference_calls": episodes * h, "planning_context_calls": episodes * h}


def contract():
    return {"source": SOURCE_RUN, "lower_source": LOWER_RUN, "reference": REFERENCE_RUN,
        "architecture": "same_Stage116_complete_eight_coordinate_readout_and_forecast_anchored_Bernstein_basis5",
        "gradient": "central_native_action_coordinate_differences_plus_minus0.05_full_suffix_same_prefix_and_lower_noise",
        "continuation": "current_deterministic_upper_mean_at_every_other_decision_full_native_lower_feedback_to_episode_end",
        "pullback": "negative_mean_of_native_Q_action_gradient_times_actor_mean_Jacobian_on_causal390_state",
        "update": "all_queries_and_both_noise_panels_same_Fisher_radius_final8_updates_no_direction_filtering_or_iteration_selection",
        "coverage": "fresh_queries_uniform_cyclic_over_all_upper_decisions_including_first_and_last",
        "freeze": "Stage112_learned_lower_donor_upper_base_and_std_values_critic_forecaster_and_plan_clock",
        "evaluation": "new_scenarios_deterministic_upper_stochastic_lower_common_innovations_no_native_branch_search_at_deployment",
        "arms": list(ARMS), "primary_endpoints": list(ENDPOINTS),
        "statistics": "all8_equal_root_bootstrap65536_Bonferroni4_no_seed_extension",
        "decision": "positive_CI_native_fd_minus_forecast_and_native_fd_minus_stage118_suffix_both_periods",
        "artifacts": "final_native_fd_upper_weights_and_compact_JSON_server_only_no_native_trace_writes",
        "limits": "more_simulation_than_Stage118_not_equal_budget_estimator_superiority_not_joint_HRL_or_promotion"}
