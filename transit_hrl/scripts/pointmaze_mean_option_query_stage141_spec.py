"""Compare stochastic credit with deployment-mean option queries."""

import math

from scripts import pointmaze_exploration_match_stage140_spec as source

ROOT, ROOTS, PERIODS = source.ROOT, source.ROOTS, source.PERIODS
PROTOCOL = "pointmaze_mean_option_query_stage141_v1"
EXPERIMENT_PROTOCOL = PROTOCOL
POLICY = "replayed_std005_score_vs_mean_option_antithetic_query_fixed_mean_step"
RUNNER_SCRIPT = "scripts/run_pointmaze_mean_option_query_stage141.py"
SOURCE_RUN = "pointmaze_exploration_match_stage140_probe_20261008_r1"
WARM_SOURCE_RUN = source.WARM_SOURCE_RUN
WORKERS, EVALUATION_EPISODES = 16, 32
STD, RADIUS, MEAN_STEP_RMS = .05, source.radius("reduced"), source.MEAN_STEP_RMS
PANELS, MODES, METHODS = source.PANELS, source.MODES, ("sampled_credit", "mean_query")
VARIANTS = ("source_forecast", "warm_start",
    *tuple(m+"_"+s for m in METHODS for s in ("plus", "minus")))
CONTRASTS = (("warm_start", "source_forecast"),
    *tuple((m+"_"+s, "warm_start") for m in METHODS for s in ("plus", "minus")),
    ("mean_query_plus", "sampled_credit_plus"),
    *tuple((m+"_plus", m+"_minus") for m in METHODS),
    *tuple((m+"_plus", "source_forecast") for m in METHODS))
METRICS, arguments, warm_result = source.METRICS, source.arguments, source.warm_result


def source_result(root):
    return ROOT/"results"/SOURCE_RUN/"cells"/f"replicate_{root}"/"result.json"


def seed_roles(root):
    base = 141100000+ROOTS.index(root)*100000
    return {"replayed_training": source.seed_roles(root)["training"],
        "evaluation": list(range(base+90001, base+90001+EVALUATION_EPISODES))}


def budget():
    h, paths, e = arguments(ROOTS[0]).horizon, source.SCENARIOS*len(PANELS), EVALUATION_EPISODES
    actual_eval = 2*len(VARIANTS)-1
    episodes = [2*paths+2*paths*(h//p)+actual_eval*e for p in PERIODS]
    chunks = [math.ceil(paths*(h//p)/source.ppo.MINIBATCH) for p in PERIODS]
    return {"source_cell_loads": 2, "lower_checkpoint_loads": len(PERIODS), "upper_checkpoint_loads": len(PERIODS),
        "replay_episodes": len(PERIODS)*paths, "collection_episodes": len(PERIODS)*paths,
        "counterfactual_episodes": 2*paths*sum(h//p for p in PERIODS),
        "evaluation_episodes": len(PERIODS)*actual_eval*e, "evaluation_alias_assignments": len(PERIODS)*e,
        "native_episodes": sum(episodes), "native_steps": sum(episodes)*h, "native_lower_calls": sum(episodes)*h,
        "native_upper_calls": sum((n-e)*(h//p) for n,p in zip(episodes, PERIODS)),
        "native_donor_response_calls": 2*sum(episodes)*h, "planning_reference_calls": sum(episodes)*h,
        "planning_renewals": sum(n*(h//p) for n,p in zip(episodes, PERIODS)),
        "planning_fits": sum(n*(h//p-1) for n,p in zip(episodes, PERIODS)),
        "credit_checks": 2*len(PERIODS)*paths+2*paths*sum(h//p for p in PERIODS),
        "mean_query_label_pairs": paths*sum(h//p for p in PERIODS),
        "score_gradient_batches": sum(chunks), "mean_score_forward_batches": sum(chunks),
        "empirical_fisher_solves": len(METHODS)*len(PERIODS),
        "fisher_jvp_batches": len(METHODS)*sum(chunks), "exact_kl_forward_batches": 2*len(METHODS)*sum(chunks),
        "policy_geometry_forward_batches": 3*len(METHODS)*len(PERIODS),
        "upper_candidate_weight_steps": 2*len(METHODS)*len(PERIODS),
        "upper_actor_optimizer_steps": 0, "upper_value_optimizer_steps": 0,
        "lower_actor_optimizer_steps": 0, "lower_value_optimizer_steps": 0,
        "checkpoint_writes": 0, "native_trace_writes": 0}


def contract():
    return {"source": SOURCE_RUN, "warm_source": WARM_SOURCE_RUN, "std": STD,
        "query": "mean_prefix_single_option_mean_plus_minus_std_epsilon_mean_future_common_lower_noise",
        "gradient": "paired_half_return_difference_times_epsilon_over_std_no_PPO_or_critic_labels",
        "sampled_comparator": "exact_Stage140_reduced_std_path_credit_replay_same_warm_mean_and_training_roster",
        "geometry": "same_standardized_damped_mean_geometry_damping1_fixed_mean_step",
        "mean_step_RMS": MEAN_STEP_RMS, "KL_radius": RADIUS, "authority": .05,
        "freeze": "all_lower_values_teacher_forecaster_std_and_warm_source",
        "evaluation": "new_paired_mean_and_sampled_scenes_both_signs_no_winner_or_CI",
        "artifacts": "compact_JSON_no_checkpoint_or_trace_writes_full_replay_and_added_query_cost_counted",
        "limits": "finite_probe_smoothed_option_gradient_query_augmented_not_equal_cost_PPO_not_joint_HRL"}
