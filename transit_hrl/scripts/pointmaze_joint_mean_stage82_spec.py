"""Test joint mean updates with a fixed sum-of-level conditional KL budget."""

from scripts import pointmaze_actor_parts_stage81_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
roots, arguments, source_result, options = source.roots, source.arguments, source.source_result, source.options
FISHER_RADIUS = source.FISHER_RADIUS
EXPERIMENT_PROTOCOL = "pointmaze_joint_mean_stage82_v1"
POLICY = "joint_mean"
RUNNER_SCRIPT = "scripts/run_pointmaze_joint_mean_stage82.py"
ALLOCATIONS = {"full": 1., "half": .5}
JOINT_SIGNS = {"joint_plus": ("plus", "plus"), "joint_minus": ("minus", "minus"),
    "joint_upper_plus_lower_minus": ("plus", "minus"), "joint_upper_minus_lower_plus": ("minus", "plus")}
DIRECTIONS = ("upper_full", "lower_full", "upper_half", "lower_half", "joint")
VARIANTS = ("base", "zero", *tuple(f"{d}_{s}" for d in DIRECTIONS for s in ("plus", "minus")),
    "joint_upper_plus_lower_minus", "joint_upper_minus_lower_plus")
CONTRAST_PAIRS = (("base", "zero"), *tuple((d + "_" + s, b) for d in DIRECTIONS
    for s, b in (("plus", d + "_minus"), ("plus", "base"), ("minus", "base"), ("plus", "zero"))),
    *tuple(("joint_plus", f"{a}_{budget}_plus") for a in ("upper", "lower") for budget in ALLOCATIONS),
    *tuple((v, "base") for v in JOINT_SIGNS if v not in ("joint_plus", "joint_minus")),
    *tuple(("joint_plus", v) for v in JOINT_SIGNS if v not in ("joint_plus", "joint_minus")))
ENDPOINTS = tuple(k for p in PERIODS for k in (*tuple(f"{p}/{a}_minus_{b}" for a, b in CONTRAST_PAIRS),
    f"{p}/joint_plus_interaction"))
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (82, 82082)


def seed_roles(root, *, preflight):
    base = 82_090_000 if preflight else 82_100_000 + roots(preflight=False).index(root) * 10000
    n, e = options(preflight=preflight)["credit_scenarios_per_batch"], options(preflight=preflight)["evaluation_episodes"]
    return {"credit_" + name: [{"scenario_seed": base + offset + i,
        "noise_seeds": [base + offset + 2001 + 2 * i, base + offset + 2002 + 2 * i]} for i in range(n)]
        for name, offset in (("A", 1), ("B", 1001))} | {
        "native_evaluation": list(range(base + 5001, base + 5001 + e))}


def budget(*, preflight):
    b = source.budget(preflight=preflight)
    b["fisher_jvp_batches"] = b["fisher_jvp_batches"] // 3 * 2
    b["exact_kl_forward_batches"] = 2 * b["fisher_jvp_batches"]
    b["actor_parameter_perturbations"] = b["parameter_part_checks"] = 8 * len(PERIODS)
    b["joint_composition_checks"] = len(JOINT_SIGNS) * len(PERIODS)
    return b


def contract():
    return {**{k: v for k, v in source.contract().items() if k not in
        ("variants", "direction", "perturbation", "parameter_freeze", "statistics", "limits")},
        "variants": list(VARIANTS), "direction": "same_fresh_scenario_MC_mean_gradient_std_bit_exact_frozen",
        "noise_mapping": "Stage80_explicit_environment_and_action_noise_mapping_on_fresh_Stage82_seed_roles",
        "perturbation": "fixed_sum_of_level_mean_conditional_Gaussian_KL_.001_single_full_.001_joint_.0005_each_no_radius_sweep",
        "allocation": "equal_half_budget_joint_plus_minus_and_cross_signs_same_half_step_actors_as_single_level_controls",
        "parameter_freeze": "both_log_std_critics_forecaster_source_Adam_decoder_frozen_joint_composition_bit_exact",
        "statistics": "all60_reward_and_interaction_contrasts_equal_root_bootstrap65536_Bonferroni60",
        "limits": "sum_of_per_level_source_state_conditional_KL_not_full_trajectory_KL_single_step_not_joint_training_not_frequency_superiority"}
