"""Frozen reused-path design for MuJoCo v18.5 actor-floor signals."""

from __future__ import annotations

from scripts import mujoco_v18_4_receding_joint_projection_spec as v18_4


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v18_5_actor_floor_signal_v1"
EVIDENCE_ROLE = "reused_path_causal_actor_floor_signal_diagnostic_only"
FROZEN_CORE_REVISION = "e97028fd121693c9c5902f2af61c5833006d887f"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "c37f934d1a5fb528b620e27678434148d65b8a727fde04176af5ce58a24b0d08"
)

REFERENCE_DATASET_RUN = v18_4.REFERENCE_DATASET_RUN
ENVIRONMENTS = v18_4.ENVIRONMENTS
DISTURBANCE_MODES = v18_4.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v18_4.REUSED_SELECTION_SEEDS
EXPECTED_PATH_COUNT = v18_4.EXPECTED_PATH_COUNT
EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT = (
    v18_4.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
)
EXPECTED_ACTOR_FLOOR_PATH_COUNT = v18_4.EXPECTED_ACTOR_FLOOR_PATH_COUNT
EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED = (
    v18_4.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
)
EXPECTED_ACTOR_FLOOR_ENVIRONMENT = "Hopper-v5"
UNRESOLVED_V17_14_PATH = {
    "environment": "Hopper-v5",
    "disturbance_mode": "ood_chirp",
    "evaluation_seed": 294864529,
}

UPPER_RMS_BUDGET = v18_4.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v18_4.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v18_4.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v18_4.LOWER_ACTION_LIMIT
UPPER_WINDOW = v18_4.UPPER_WINDOW
LOWER_WINDOW = v18_4.LOWER_WINDOW
POWER_TOLERANCE = v18_4.POWER_TOLERANCE
VELOCITY_ALPHA = 0.25
VELOCITY_DECAY = 0.75
COORDINATE_SWEEPS = 24
MULTIPLIER_BISECTION_STEPS = 8

CANDIDATES = {
    "actor_floor_h16_hold": {
        "planning_horizon": 16,
        "forecast_mode": "hold",
    },
    "actor_floor_h16_damped_velocity": {
        "planning_horizon": 16,
        "forecast_mode": "damped_velocity",
    },
}

DIAGNOSTIC_CONTRACT = {
    "input": (
        "unchanged v17.8 total-action traces and reference full-horizon "
        "feasibility labels; no v17.12 correction target, reward, observation, "
        "future action, or fresh path"
    ),
    "signals": (
        "causal finite-horizon lower-power excess at the upper-budget floor, "
        "its normalized ratio, forecast joint-infeasibility rate, unavoidable "
        "prefix upper violation, and one-step forecast error"
    ),
    "candidate_set": (
        "H16 hold and H16 damped-velocity forecasts with the already frozen "
        "HPF8/LPF32 budgets, prefix upper projection, velocity parameters, "
        "coordinate sweeps, and bisection count"
    ),
    "assessment": (
        "path-level floor-vs-reference rank AUC both globally and against "
        "Hopper-only reference paths, plus floor recall among the top 7, 14, "
        "and 28 globally ranked paths for each preregistered causal score; the "
        "unresolved Hopper OOD-chirp seed 294864529 rank is reported"
    ),
    "decision_rule": (
        "the diagnostic may motivate a separately frozen FIR debt-feedback "
        "screen only if at least one score has Hopper-conditioned AUC at least "
        "0.75, global top-14 floor recall at least 6/7, and ranks the unresolved "
        "path in the global top 14; this rule does not authorize fresh paths"
    ),
    "claim_boundary": (
        "reused-path signal diagnostic only; no correction policy, reward, "
        "fresh-seed, learned-control, or manuscript claim"
    ),
}
