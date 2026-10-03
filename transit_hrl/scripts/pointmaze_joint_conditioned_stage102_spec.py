"""Joint mean learning with actor-specific paired MC credit."""

from scripts import pointmaze_fresh_joint_stage98_spec as source
from scripts.pointmaze_iterative_mc_stage83_spec import training_budget

ROOT, PERIODS, CHUNK_SIZE, FISHER_RADIUS = source.ROOT, source.PERIODS, source.CHUNK_SIZE, source.FISHER_RADIUS
roots, arguments, options, source_result, source_record = source.roots, source.arguments, source.options, source.source_result, source.source_record
METHODS = {m: ("upper", "lower") for m in ("joint_independent", "joint_conditioned")}
COMPOSITIONS = {
    "base": ("source", "source"), "zero": ("source", "source"),
    "joint_independent": ("joint_independent", "joint_independent"),
    "joint_conditioned": ("joint_conditioned", "joint_conditioned"),
    "conditioned_upper_independent_lower": ("joint_conditioned", "joint_independent"),
    "independent_upper_conditioned_lower": ("joint_independent", "joint_conditioned")}
VARIANTS = tuple(COMPOSITIONS)
CONTRAST_PAIRS = (
    ("joint_conditioned", "joint_independent"), ("joint_conditioned", "base"),
    ("joint_independent", "base"), ("joint_conditioned", "zero"), ("joint_independent", "zero"),
    ("conditioned_upper_independent_lower", "joint_independent"),
    ("independent_upper_conditioned_lower", "joint_independent"),
    ("joint_conditioned", "independent_upper_conditioned_lower"),
    ("joint_conditioned", "conditioned_upper_independent_lower"), ("base", "zero"))
ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS)
PRIMARY_ENDPOINTS = tuple(f"{p}/{a}_minus_{b}" for p in PERIODS for a, b in CONTRAST_PAIRS[:2])
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (102, 102102)
EXPERIMENT_PROTOCOL = "pointmaze_joint_conditioned_stage102_v1"
POLICY = "separate_upper_lower_credit_joint_mean"
RUNNER_SCRIPT = "scripts/run_pointmaze_joint_conditioned_stage102.py"
ANALYZER_SCRIPT = "scripts/analyze_pointmaze_joint_conditioned_stage102.py"


def allocation(method, period):
    if method not in METHODS:
        raise ValueError("Stage102 trains only the two registered joint learners")
    return source.allocation("joint_call", period)


def seed_roles(root, *, preflight):
    roots(preflight=preflight).index(root)
    base = 102_000_000 if preflight else 102_100_000 + roots(preflight=False).index(root)*100000
    o = options(preflight=preflight)
    rounds = [{name: [{"scenario_seed": base+10000*j+offset+i,
        "noise_seeds": [base+10000*j+offset+2001+2*i, base+10000*j+offset+2002+2*i]}
        for i in range(o["credit_scenarios_per_batch"])]
        for name, offset in (("credit_A", 1), ("credit_B", 1001),
            ("lower_credit_A", 4001), ("lower_credit_B", 5001))}
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
    calls = extra//len(PERIODS)*sum(h//p for p in PERIODS)
    for k in ("native_upper_calls", "pairing_upper_forward_calls"):
        b[k] += calls
    pairs = len(PERIODS)*o["updates"]*2*o["credit_scenarios_per_batch"]
    return {**b, "upper_independent_pair_checks": len(METHODS)*pairs,
        "lower_independent_pair_checks": pairs, "lower_common_pair_checks": pairs,
        "actor_composition_checks": len(PERIODS)*len(VARIANTS),
        "upper_replay_forward_calls": o["updates"]*2*o["credit_scenarios_per_batch"]*sum(h//p for p in PERIODS)}


def contract():
    inherited = {k: v for k, v in source.contract().items() if k != "downstream"}
    return {**inherited, "variants": list(VARIANTS),
        "methods": {str(p): {m: allocation(m, p) for m in METHODS} for p in PERIODS},
        "compositions_upper_lower": {v: list(pair) for v, pair in COMPOSITIONS.items()},
        "training": "both_actor_means_from_original_Stage96_U0_L0_updated_simultaneously_eight_rounds_preflight_two",
        "sampling": "per_method_per_update64_upper_and64_lower_paths_preflight8_each_separate_registered_roles_before_either_actor_update",
        "baseline": "upper_same_scenario_independent_upper_lower_noise_lower_same_scenario_independent_lower_noise_common_upper_innovations_only_in_conditioned_arm",
        "noise_mapping": "original_Stage80_mapping_fresh_Stage102_roles_upper_replay_only_lower_credit_never_upper_credit_or_evaluation",
        "budget": "both_arms_equal_native_paths_and_call_weighted_KL_.001_per_update_extra_conditioned_lower_replay_reported_separately",
        "freeze": "std_values_source_Adam_forecaster_decoder_original_Stage96_teachers_unchanged_no_trained_donor_reuse",
        "statistics": "all20_reward_contrasts_equal_root_bootstrap65536_Bonferroni20_seed102_102102_no_stage_pooling",
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "decision": "all_four_conditioned_minus_independent_and_original_teacher_primary_lower_CI_bounds_positive_preflight_mechanical_only",
        "replication": "joint_training_integration_not_unchanged_Stage98_99_101_replication",
        "limits": "same_task_teacher_initialized_fixed_std_decoder_joint_MC_mean_learning_not_full_actor_critic_frequency_superiority_or_equal_prior_stage_compute"}
