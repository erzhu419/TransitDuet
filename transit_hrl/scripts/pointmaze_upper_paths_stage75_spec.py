"""Frozen-clone factorial intervention on reference and velocity paths."""

from scripts import pointmaze_crossed_direction_stage74_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_upper_paths_stage75_v1"
POLICY = "upper_paths"
RUNNER_SCRIPT = "scripts/run_pointmaze_upper_paths_stage75.py"
PERIODS, roots, arguments = source.PERIODS, source.roots, source.arguments
MODES = ("R0V0", "R1V0", "R0V1", "R1V1")
METRICS = ("episode_return", "tracking_squared_error_integral")
CONTRASTS = ("reference_at_base_velocity", "reference_at_residual_velocity",
    "velocity_at_base_reference", "velocity_at_residual_reference", "normal_minus_zero", "interaction")
ENDPOINTS = tuple(f"{p}/{m}/{c}" for p in PERIODS for m in METRICS for c in CONTRASTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (75, 75075)


def options(*, preflight):
    return {"evaluation_episodes": 4 if preflight else 32, "workers": 2 if preflight else 8}


def seed_roles(root, *, preflight):
    old = source.seed_roles(root, preflight=preflight)
    base = 75_090_000 if preflight else 75_100_000 + roots(preflight=False).index(root) * 10000
    seeds = list(range(base + 5001, base + 5001 + options(preflight=preflight)["evaluation_episodes"]))
    if set(seeds).intersection([*old["calibration"], *old["native_evaluation"]]):
        raise ValueError("Stage75 evaluation overlaps historical calibration or Stage74 probes")
    return {"native_evaluation": seeds}


def source_result(root, *, preflight):
    run = "pointmaze_crossed_direction_stage74_" + ("preflight" if preflight else "full") + "_20261001_r1"
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def budget(*, preflight):
    n = options(preflight=preflight)["evaluation_episodes"]
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    extra_modes = 2 if preflight else 0
    count = len(PERIODS) * n * (len(MODES) + extra_modes)
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1,
        "native_episodes": count, "native_steps": count * h, "native_lower_calls": count * h,
        "native_upper_calls": n * (len(MODES) + extra_modes) * sum(h // p for p in PERIODS),
        "native_network_checks": count, "native_pair_checks": len(PERIODS) * n,
        "production_equivalence_checks": len(PERIODS) * n * extra_modes,
        "frozen_model_checks": len(PERIODS)}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "modes": list(MODES), "periods": list(PERIODS),
        "parameters": "Stage55_teacher_clones_frozen_no_historical_direction_fitting_or_perturbation",
        "intervention": "R_selects_reference_position_V_selects_actor_and_value_planned_velocity_0_base_forecast_1_residual_curve",
        "planning": "one_causal_ridge_forecast_per_option_anchored_Bernstein_residual_same_fixed_scale_and_world_clipping",
        "native_pairing": "fresh_common_environment_seeds_stepwise_lower_noise_initial_policy_rng_upper_proposals_all_four_modes",
        "sampling": "both_actors_sampled_fixed50_100_no_batch_or_raw_trace",
        "production_equivalence": "preflight_only_16_extra_native_episodes_exact_reward_proposal_noise_call_match_R0V0_zero_R1V1_normal",
        "statistics": "all24_reward_and_tracking_error_factorial_contrasts_equal_root_percentile_bootstrap_Bonferroni24",
        "decision": "mechanism_diagnosis_only_no_path_selection_or_policy_adoption_Stage67_HOLD_unchanged",
        "artifacts": "compact_JSON_no_new_native_traces_or_checkpoints",
        "limits": "mixed_paths_are_diagnostic_not_coherent_deployment_plans_teacher_initialized_development_roots_not_joint_training_or_frequency_proof"}
