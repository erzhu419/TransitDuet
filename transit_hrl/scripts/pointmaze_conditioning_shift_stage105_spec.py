"""Frozen lower-conditioning test under faster hidden-regime switching."""

from scripts import pointmaze_joint_conditioned_stage102_spec as source
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, options, source_record = source.roots, source.options, source.source_record
METHODS, allocation = source.METHODS, source.allocation
COMPOSITIONS = {k: source.COMPOSITIONS[k] for k in ("base", "zero", *METHODS)}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (("joint_conditioned", "joint_independent"), ("joint_conditioned", "base"),
    ("joint_independent", "base"), ("joint_conditioned", "zero"),
    ("joint_independent", "zero"), ("base", "zero"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS[:2])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (105, 105105)
EXPERIMENT_PROTOCOL = "pointmaze_conditioning_shift_stage105_v1"
POLICY = "fast_regime_lower_conditioning_joint_mean"
RUNNER_SCRIPT = "scripts/run_pointmaze_conditioning_shift_stage105.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_conditioning_shift_stage105.py"
REGIME_DWELL_SECONDS = (.40, .80)


def arguments(root, *, preflight):
    args = source.arguments(root, preflight=preflight)
    args.regime_dwell_seconds = REGIME_DWELL_SECONDS
    return args


def task_options(root, *, preflight):
    from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
    return {k: list(v) if isinstance(v, tuple) else v
        for k, v in _task_options(arguments(root, preflight=preflight)).items()}


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 105_000_000 if preflight else 105_100_000 + roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    rounds = [{name: [{"scenario_seed": base+10000*j+offset+i,
        "noise_seeds": [base+10000*j+offset+2001+2*i, base+10000*j+offset+2002+2*i]}
        for i in range(o["credit_scenarios_per_batch"])]
        for name, offset in (("credit_A", 1), ("credit_B", 1001), ("lower_credit_A", 4001), ("lower_credit_B", 5001))}
        for j in range(o["updates"])]
    return {"training_rounds": rounds,
        "native_evaluation": list(range(base+95001, base+95001+o["evaluation_episodes"]))}


def budget(*, preflight):
    o = options(preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    b = training_budget(o, METHODS, VARIANTS, horizon=h, preflight=preflight)
    extra = b["credit_episodes"]
    for k in ("credit_episodes", "objective_checks", "mc_calls", "scenario_pair_checks"):
        b[k] *= 2
    for k in ("native_episodes", "native_network_checks"):
        b[k] += extra
    for k in ("native_steps", "native_lower_calls"):
        b[k] += extra*h
    for k in ("native_upper_calls", "pairing_upper_forward_calls"):
        b[k] += extra//len(PERIODS)*sum(h//p for p in PERIODS)
    pairs = len(PERIODS)*o["updates"]*2*o["credit_scenarios_per_batch"]
    return {**b, "upper_independent_pair_checks": len(METHODS)*pairs,
        "lower_independent_pair_checks": pairs, "lower_common_pair_checks": pairs,
        "actor_composition_checks": len(PERIODS)*len(VARIANTS),
        "upper_replay_forward_calls": o["updates"]*2*o["credit_scenarios_per_batch"]*sum(h//p for p in PERIODS)}


def contract():
    inherited = source.contract()
    return {k: inherited[k] for k in ("source", "decoder", "credit", "artifacts")} | {
        "periods": list(PERIODS), "variants": list(VARIANTS), "methods": list(METHODS),
        "task_change": {"regime_dwell_seconds": list(REGIME_DWELL_SECONDS), "source_regime_dwell_seconds": [.8, 1.6],
            "other_environment_parameters": "unchanged_Stage96", "training_and_evaluation": "same_shifted_distribution"},
        "source_cohort": "all_eight_original_Stage96_U0_L0_teachers_Stage97_decoder_no_Stage98_to104_trained_weights",
        "training": "eight_simultaneous_mean_updates_per_actor_preflight_two_separate_actor_credit_pools",
        "sampling": "same_actor_specific_scenario_noise_rosters_across_arms_original_independent_upper_credit",
        "conditioning": "common_upper_innovations_only_in_conditioned_lower_training_independent_lower_noise",
        "freeze": "source_std_values_Adam_forecaster_and_decoder_fixed_no_OOD_recalibration",
        "budget": "matched_native_paths_actor_updates_and_nominal_call_weighted_KL_.001_per_round_extra_replay_reported",
        "evaluation": "fresh_paired_original_independent_noise_after_fixed_final_update_no_selection",
        "statistics": "all12_contrasts_equal_root_bootstrap65536_Bonferroni12_seed105_105105_no_pooling",
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "decision": "all_four_conditioned_minus_independent_and_base_primary_CI_lower_bounds_positive_preflight_mechanical_only",
        "limits": "shifted_distribution_adaptation_from_original_teachers_not_new_initialization_cross_domain_zero_shot_full_actor_critic_or_frequency_superiority"}
