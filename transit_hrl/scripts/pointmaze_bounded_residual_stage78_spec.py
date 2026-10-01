"""Historical nonlinear response constraint, frozen four-curve native test."""

import math
from scripts import pointmaze_calibrated_residual_stage77_spec as source

ROOT, PERIODS, CHUNK_SIZE = source.ROOT, source.PERIODS, source.CHUNK_SIZE
EXPERIMENT_PROTOCOL = "pointmaze_bounded_residual_stage78_v1"
POLICY = "bounded_residual"
RUNNER_SCRIPT = "scripts/run_pointmaze_bounded_residual_stage78.py"
MODES = ("zero", "original", "ratio", "bounded")
METRICS = source.METRICS
CONTRAST_PAIRS = (("original", "zero"), ("ratio", "zero"), ("bounded", "zero"),
    ("ratio", "original"), ("bounded", "original"), ("bounded", "ratio"))
CONTRASTS = tuple(f"{a}_minus_{b}" for a, b in CONTRAST_PAIRS)
ENDPOINTS = tuple(f"{p}/{m}/{c}" for p in PERIODS for m in METRICS for c in CONTRASTS)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 65536, (78, 78078)
options = source.options


def arguments(root, *, preflight):
    args = source.arguments(root, preflight=False)
    if preflight:args.horizon = 300
    return args


def roots(*, preflight):
    return source.roots(preflight=False)[:1] if preflight else source.roots(preflight=False)


def source_preflight(preflight):
    return False


def production_replay_count(preflight):
    return 0


def source_result(root, *, preflight):
    return ROOT / "results" / "pointmaze_calibrated_residual_stage77_full_20261002_r1" / "cells" / f"replicate_{root}" / "result.json"


def seed_roles(root, *, preflight):
    base = 78_090_000 if preflight else 78_100_000 + roots(preflight=False).index(root) * 10000
    n = options(preflight=preflight)["evaluation_episodes"]
    return {"calibration_labels": source.seed_roles(root, preflight=False)["calibration_labels"],
        "native_evaluation": list(range(base + 5001, base + 5001 + n))}


def mode_alphas(c):
    return {"zero": 0., "original": 1., "ratio": c["ratio_alpha"], "bounded": c["alpha"]}


def budget(*, preflight):
    root = roots(preflight=preflight)[0]
    h = arguments(root, preflight=preflight).horizon
    label_h = arguments(root, preflight=False).horizon
    n = len(seed_roles(root, preflight=preflight)["calibration_labels"])
    e = options(preflight=preflight)["evaluation_episodes"]
    count = len(PERIODS) * len(MODES) * e
    return {"source_clone_loads": len(PERIODS), "forecaster_loads": 1, "layout_loads": 1,
        "label_archive_loads": len(PERIODS) * n, "label_state_rows": len(PERIODS) * n * label_h,
        "label_plan_checks": len(PERIODS) * n, "bc_mse_reproductions": len(PERIODS),
        "offline_actor_rows": 3 * len(PERIODS) * n * label_h,
        "offline_actor_forward_batches": 3 * len(PERIODS) * n * math.ceil(label_h / CHUNK_SIZE),
        "calibration_curve_decodes": n * sum(label_h // p for p in PERIODS),
        "calibration_upper_proposals": n * sum(label_h // p for p in PERIODS),
        "ratio_reproductions": len(PERIODS), "constraint_response_evaluations": len(PERIODS),
        "native_episodes": count, "native_steps": count * h, "native_lower_calls": count * h,
        "native_upper_calls": e * len(MODES) * sum(h // p for p in PERIODS),
        "native_network_checks": count, "native_pair_checks": len(PERIODS) * e,
        "production_equivalence_checks": 0, "frozen_model_checks": len(PERIODS)}


def realized_budget(cell, *, preflight):
    expected = budget(preflight=preflight)
    evaluations = sum(len(g["calibration"]["solver_trace"]) for g in cell["groups"].values())
    label_h = arguments(cell["root"], preflight=False).horizon
    n = len(seed_roles(cell["root"], preflight=preflight)["calibration_labels"])
    expected.update(constraint_response_evaluations=evaluations,
        offline_actor_rows=(2 * len(PERIODS) + evaluations) * n * label_h,
        offline_actor_forward_batches=(2 * len(PERIODS) + evaluations) * n * math.ceil(label_h / CHUNK_SIZE))
    return expected


def contract():
    return {"source": source.EXPERIMENT_PROTOCOL, "modes": list(MODES), "periods": list(PERIODS),
        "calibration": "all_full_Stage55_BC_states_and_Stage76_fixed_proposals_exact_Stage77_ratio_reproduction",
        "constraint": "actual_conditional_tanh_command_change_RMS_le_sqrt_BC_command_MSE_joint_reference_and_velocity",
        "solver": "start_at_Stage77_ratio_alpha_halve_until_first_feasible_actual_response_no_monotonicity_or_reward_search",
        "curve": source.contract()["curve"], "sampling": source.contract()["sampling"],
        "preflight": "first_full_root_full_BC_labels_and_trained_clone_four_fresh_native_seeds_H300_no_repeated_production_equivalence",
        "cost": "budget_is_minimum_offline_rows_and_batches_realized_from_complete_solver_traces_no_uncounted_passes",
        "statistics": "all24_reward_tracking_pairwise_contrasts_equal_root_bootstrap_65536_Bonferroni24",
        "decision": "constraint_validation_only_no_policy_adoption_Stage67_HOLD_unchanged",
        "artifacts": "compact_JSON_only_no_optimizer_forecaster_fit_native_traces_or_checkpoints",
        "limits": "aggregate_historical_conditional_command_bound_not_pointwise_or_native_reward_guarantee_not_joint_HRL_or_frequency_proof"}
