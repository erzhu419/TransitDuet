"""Local plan interventions through the final Stage112 lower controller."""

from scripts import pointmaze_upper_full_plan_train_stage116_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
EXPERIMENT_PROTOCOL = "pointmaze_local_plan_gain_stage117_v1"
POLICY = "one_option_plan_intervention_full_suffix_frozen_stage112_lower"
METHODS = {}
RUNNER_SCRIPT = "scripts/run_pointmaze_local_plan_gain_stage117.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_local_plan_gain_stage117.py"
SOURCE_RUN, LOWER_RUN = source.SOURCE_RUN, source.LOWER_RUN
roots, arguments, lower_checkpoint = source.roots, source.arguments, source.lower_checkpoint
source_result = source.source_result
EPSILON, ACTION_DIM = .05, 8
PANELS = ("A", "B")
DIRECTIONS = tuple(f"axis{i}_{sign}" for i in range(ACTION_DIM) for sign in ("plus", "minus"))
VARIANTS = ("forecast", "zero", *DIRECTIONS)
METRICS = ("crossfit_suffix_gain", "suffix_gradient_dot", "suffix_gradient_cosine",
           "suffix_minus_option_selection")
ENDPOINTS = tuple(f"{period}/{metric}" for period in PERIODS for metric in METRICS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (117, 117117)


def options(*, preflight):
    return {"workers": 2 if preflight else 4, "queries": 1 if preflight else 12}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 117000000 if preflight else 117100000 + index * 100000
    horizon = arguments(root, preflight=preflight).horizon
    starts = (100,) if preflight else (horizon // 4, horizon // 2, 3 * horizon // 4)
    return {"queries": [{"scenario_seed": base + 95001 + i, "start": starts[i % len(starts)],
        "prefix_noise_seed": base + 1001 + i,
        "suffix_noise_seeds": {name: base + 10001 + 1000 * j + i for j, name in enumerate(PANELS)}}
        for i in range(options(preflight=preflight)["queries"])]}


def budget(*, preflight):
    queries = len(PERIODS) * options(preflight=preflight)["queries"]
    horizon = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    episodes = queries * len(PANELS) * len(VARIANTS)
    per_period = options(preflight=preflight)["queries"] * len(PANELS) * len(VARIANTS)
    return {"source_cell_loads": 1, "source_clone_loads": len(PERIODS),
        "lower_checkpoint_loads": len(PERIODS), "native_episodes": episodes,
        "native_steps": episodes * horizon, "native_lower_calls": episodes * horizon,
        "native_pair_groups": queries, "prefix_pair_checks": queries * (len(PANELS) * len(VARIANTS) - 1),
        "external_pair_checks": queries * (len(PANELS) * len(VARIANTS) - 1),
        "innovation_pair_checks": queries * len(PANELS) * (len(VARIANTS) - 1),
        "zero_forecast_checks": queries * len(PANELS), "suffix_credit_checks": episodes,
        "frozen_source_checks": len(PERIODS), "plan_renewals": per_period * sum(horizon // p for p in PERIODS),
        "plan_fits": per_period * sum(horizon // p - 1 for p in PERIODS),
        "reference_evaluations": episodes * horizon, "actor_context_evaluations": episodes * horizon}


def contract():
    return {"source": SOURCE_RUN, "lower_source": LOWER_RUN,
        "lower": "all8_final_Stage112_learned_lower_branches_full392_feedback_and_fixed_std",
        "plan": "Stage116_basis5_eight_coordinate_decoder_alpha1_forecast_anchor",
        "intervention": "one_interior_option_one_latent_coordinate_plus_minus0.05_all_other_options_zero_residual",
        "baseline": "zero_residual_matches_direct_forecast_not_a_second_primary_endpoint",
        "continuation": "full_native_episode_feedback_control_after_the_intervened_option",
        "noise": "same_scenario_and_prefix_independent_suffix_A_B_common_across_variants_within_panel",
        "credit": "undiscounted_suffix_option_and_post_option_returns_no_critic_or_option_truncation",
        "selection": "A_selects_best_suffix_or_option_direction_including_zero_B_scores_then_reverse_average",
        "primary": "both_periods_positive_corrected_CI_crossfit_suffix_gain_and_suffix_gradient_dot",
        "statistics": "all8_equal_root_bootstrap65536_Bonferroni8_no_root_or_direction_exclusion",
        "freeze": "all_policy_parameters_values_std_forecaster_task_and_fixed_plan_clock_no_training",
        "artifacts": "scalar_JSON_only_no_checkpoint_or_native_trace_writes",
        "limits": "conditional_local_plan_headroom_and_noise_replication_not_a_causal_deployable_selector_or_learned_HRL"}
