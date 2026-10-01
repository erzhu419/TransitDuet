"""Historical command-response calibration followed by fresh native probes."""

import math
from scripts import pointmaze_velocity_support_stage76_spec as source

ROOT = source.ROOT
EXPERIMENT_PROTOCOL = "pointmaze_calibrated_residual_stage77_v1"
POLICY = "calibrated_residual"
RUNNER_SCRIPT = "scripts/run_pointmaze_calibrated_residual_stage77.py"
PERIODS, roots, arguments = source.PERIODS, source.roots, source.arguments
CHUNK_SIZE = source.CHUNK_SIZE
MODES = ("zero", "original", "calibrated")
METRICS = ("episode_return", "tracking_squared_error_integral")
CONTRASTS = ("original_minus_zero", "calibrated_minus_original", "calibrated_minus_zero")
ENDPOINTS = tuple(f"{p}/{m}/{c}" for p in PERIODS for m in METRICS for c in CONTRASTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (77, 77077)


def options(*, preflight):
    return {"evaluation_episodes": 4 if preflight else 32, "workers": 2 if preflight else 8}


def source_result(root, *, preflight):
    run = "pointmaze_velocity_support_stage76_" + ("preflight_20261001_r2" if preflight else "full_20261001_r1")
    return ROOT / "results" / run / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    base = 77_090_000 if preflight else 77_100_000 + roots(preflight=False).index(root) * 10000
    n = options(preflight=preflight)["evaluation_episodes"]
    return {"calibration_labels": source.seed_roles(root, preflight=preflight)["calibration_labels"],
        "native_evaluation": list(range(base + 5001, base + 5001 + n))}


def budget(*, preflight):
    roles = seed_roles(roots(preflight=preflight)[0], preflight=preflight)
    h = arguments(roots(preflight=preflight)[0], preflight=preflight).horizon
    n, e = len(roles["calibration_labels"]), len(roles["native_evaluation"])
    modes = len(MODES) + (2 if preflight else 0)
    count = len(PERIODS) * e * modes
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "layout_loads": 1,
        "label_archive_loads": len(PERIODS) * n, "label_state_rows": len(PERIODS) * n * h,
        "label_plan_checks": len(PERIODS) * n, "bc_mse_reproductions": len(PERIODS),
        "offline_actor_rows": 3 * len(PERIODS) * n * h,
        "offline_actor_forward_batches": 3 * len(PERIODS) * n * math.ceil(h / CHUNK_SIZE),
        "calibration_curve_decodes": n * sum(h // p for p in PERIODS),
        "calibration_upper_proposals": n * sum(h // p for p in PERIODS),
        "native_episodes": count, "native_steps": count * h, "native_lower_calls": count * h,
        "native_upper_calls": e * modes * sum(h // p for p in PERIODS),
        "native_network_checks": count, "native_pair_checks": len(PERIODS) * e,
        "production_equivalence_checks": len(PERIODS) * e * (2 if preflight else 0),
        "frozen_model_checks": len(PERIODS)}


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "modes": list(MODES), "periods": list(PERIODS),
        "calibration": "all_Stage55_BC_states_joint_reference_error_and_velocity_change_Stage76_fixed_marginal_proposals",
        "alpha": "min(1,sqrt(BC_command_MSE)/original_joint_command_change_RMS)_once_per_root_period_no_reward_selection",
        "curve": "convex_blend_base_and_world_clipped_original_points_float32_same_curve_reference_and_dt_derivative",
        "sampling": "both_actors_sampled_fixed50_100_fresh_paired_seeds_stepwise_lower_noise_identical_upper_proposals",
        "production_equivalence": "preflight_only_zero_original_exact_native_reward_noise_proposal_call_match",
        "statistics": "all12_reward_tracking_contrasts_equal_root_bootstrap_65536_Bonferroni12",
        "decision": "repair_validation_only_no_policy_adoption_Stage67_HOLD_unchanged",
        "artifacts": "compact_JSON_no_native_traces_or_checkpoints_no_optimizer_or_forecaster_fit",
        "limits": "RMS_ratio_is_not_a_nonlinear_command_constraint_teacher_initialized_development_roots_not_joint_HRL_or_frequency_proof"}
