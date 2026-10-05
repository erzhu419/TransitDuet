"""Bounded structural plan-authority test before further upper training."""

from scripts import pointmaze_local_plan_gain_stage117_spec as source

ROOT, PERIODS = source.ROOT, source.PERIODS
EXPERIMENT_PROTOCOL = "pointmaze_plan_authority_stage120_v1"
POLICY = "one_option_plan_through_advice_or_bounded_donor_tracking"
RUNNER_SCRIPT = "scripts/run_pointmaze_plan_authority_stage120.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_plan_authority_stage120.py"
METHODS = {}
SOURCE_RUN, LOWER_RUN = source.SOURCE_RUN, source.LOWER_RUN
roots, arguments = source.roots, source.arguments
source_result, lower_checkpoint = source.source_result, source.lower_checkpoint
PANELS, DIRECTIONS, ACTION_DIM = source.PANELS, source.DIRECTIONS, source.ACTION_DIM
AMPLITUDES = {"small": .05, "large": 1.0}
CHANNELS = ("advice", "reference")
REFERENCE_LIMIT, MINIMUM_GAIN = .05, .5
METRICS = tuple(f"{channel}_{size}_gain" for channel in CHANNELS for size in AMPLITUDES) + (
    "reference_large_minus_advice_large",)
ENDPOINTS = tuple(f"{p}/{metric}" for p in PERIODS for metric in METRICS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (120, 120120)


def options(*, preflight):
    return {"workers": 1 if preflight else 4, "queries": 1 if preflight else 4}


def seed_roles(root, *, preflight):
    index = roots(preflight=preflight).index(root)
    base = 120000000 if preflight else 120100000 + index * 100000
    horizon = arguments(root, preflight=preflight).horizon
    starts = (0,) if preflight else (0, horizon // 4, horizon // 2, 3 * horizon // 4)
    return {"queries": [{"scenario_seed": base + 95001 + i, "start": starts[i % len(starts)],
        "prefix_noise_seed": base + 1001 + i,
        "suffix_noise_seeds": {p: base + 10001 + 1000 * j + i for j, p in enumerate(PANELS)}}
        for i in range(options(preflight=preflight)["queries"])]}


def budget(*, preflight):
    queries = len(PERIODS) * options(preflight=preflight)["queries"]
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    interventions = len(DIRECTIONS) * len(AMPLITUDES)
    variants = 1 + len(CHANNELS) * (1 + interventions)
    episodes = queries * len(PANELS) * variants
    per_period = options(preflight=preflight)["queries"] * len(PANELS) * variants
    return {"source_cell_loads": 1, "source_clone_loads": len(PERIODS), "lower_checkpoint_loads": len(PERIODS),
        "native_pair_groups": queries, "native_episodes": episodes, "native_steps": episodes * h,
        "native_lower_calls": episodes * h,
        "reference_donor_calls": queries * len(PANELS) * (1 + interventions) * 2 * h,
        "prefix_pair_checks": queries * (len(PANELS) * variants - 1),
        "external_pair_checks": queries * (len(PANELS) * variants - 1),
        "innovation_pair_checks": queries * len(PANELS) * (variants - 1),
        "zero_forecast_checks": queries * len(PANELS) * len(CHANNELS),
        "suffix_credit_checks": episodes, "frozen_source_checks": len(PERIODS),
        "plan_renewals": per_period * sum(h // p for p in PERIODS),
        "plan_fits": per_period * sum(h // p - 1 for p in PERIODS),
        "reference_evaluations": episodes * h, "actor_context_evaluations": episodes * h}


def contract():
    return {"source": SOURCE_RUN, "lower_source": LOWER_RUN,
        "freeze": "all8_Stage112_final_learned_lowers_full392_feedback_std_forecaster_task_and_clock",
        "plan": "forecast_anchored_basis5_eight_coordinate_curve_one_option_all_other_options_zero",
        "amplitudes": AMPLITUDES, "channels": list(CHANNELS),
        "reference": "unchanged_lower_mean_plus_bounded_frozen_donor_response_to_curve_minus_forecast_position_velocity",
        "reference_limit": REFERENCE_LIMIT,
        "zero": "both_channels_exactly_reproduce_forecast_commands_and_returns",
        "queries": "fresh_scenarios_first_quarter_half_three_quarter_option_independent_A_B_suffix_noise",
        "selection": "A_selects_direction_including_zero_B_scores_full_suffix_then_reverse_not_deployable",
        "statistics": "all8_equal_root_bootstrap65536_Bonferroni10_no_seed_extension",
        "minimum_gain": MINIMUM_GAIN,
        "decision": "train_new_channel_only_if_both_period_large_reference_gain_CI_above0.5_and_advantage_CI_above0",
        "artifacts": "scalar_JSON_only_no_training_checkpoint_or_native_trace_writes",
        "limits": "finite_candidate_conditional_headroom_not_global_upper_bound_causal_policy_or_joint_HRL_equal_compute"}
