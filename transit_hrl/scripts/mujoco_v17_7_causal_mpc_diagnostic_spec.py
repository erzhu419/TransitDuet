"""Frozen development design for MuJoCo v17.7 causal responsibility MPC."""

from __future__ import annotations

from scripts import (  # noqa: F401
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_7_causal_mpc_diagnostic_v1"
EVIDENCE_ROLE = "reused_v17_4_path_causal_router_selection_not_confirmatory"
FROZEN_CORE_REVISION = "61b8d6535cadb88f46740af65f8c5e7d0ce84fce"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "e6819fe80ae428755ffd355dbb8c22eece71a7dff9a7da8c750b8029d4b072c7"
)
SOURCE_RUN_NAME = (
    "mujoco_v17_4_streaming_audit_projection_preflight_20260831_r1"
)
ORACLE_RUN_NAME = "mujoco_v17_6_full_horizon_oracle_20260831_r1"
ENVIRONMENTS = v17_4.ENVIRONMENTS
DISTURBANCE_MODES = v17_4.EVALUATION_DISTURBANCE_MODES
EVALUATION_SEEDS = v17_4.EVALUATION_SEEDS
OPTIMIZER_SEED = v17_4.OPTIMIZER_SEEDS[0]
UPPER_RMS_BUDGET = v17_4.UPPER_HF_RMS_BUDGET
LOWER_RMS_BUDGET = v17_4.LOWER_LF_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_4.UPPER_ACTION_SCALE
LOWER_ACTION_LIMIT = v17_4.LOWER_ACTION_SCALE
POWER_TOLERANCE = 1e-8
RECONSTRUCTION_TOLERANCE = 1e-7
BOUND_TOLERANCE = 1e-7
COORDINATE_SWEEPS = 24
MULTIPLIER_BISECTION_STEPS = 8
VELOCITY_ALPHA = 0.25
VELOCITY_DECAY = 0.75
CANDIDATES = {
    "hold_h16": {"forecast_mode": "hold", "planning_horizon": 16},
    "hold_h32": {"forecast_mode": "hold", "planning_horizon": 32},
    "velocity_h16": {
        "forecast_mode": "damped_velocity",
        "planning_horizon": 16,
    },
    "velocity_h32": {
        "forecast_mode": "damped_velocity",
        "planning_horizon": 32,
    },
}
RECOVERY_GATE_TOTAL = 65
RECOVERY_GATE_BY_ENVIRONMENT = {
    "HalfCheetah-v5": 32,
    "Hopper-v5": 24,
    "Walker2d-v5": 6,
}
PRESERVE_BASELINE_FEASIBLE_WALKER_GATE = 30
EXPECTED_PATH_COUNT = (
    len(ENVIRONMENTS) * len(DISTURBANCE_MODES) * len(EVALUATION_SEEDS)
)
