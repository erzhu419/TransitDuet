"""Frozen reused-path design for MuJoCo v18.4 joint MPC projection."""

from __future__ import annotations

from scripts import mujoco_v18_3_causal_joint_projection_spec as v18_3


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v18_4_receding_joint_projection_v1"
EVIDENCE_ROLE = (
    "reused_path_label_free_causal_receding_projection_not_confirmatory"
)
FROZEN_CORE_REVISION = "7f649e23a05f3bf142255fc777bb4322931cc617"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "c7f7f0980508d58ea167d9bd420fc88ecefafb207d8735f791169ea822015dff"
)

REFERENCE_DATASET_RUN = v18_3.REFERENCE_DATASET_RUN
ENVIRONMENTS = v18_3.ENVIRONMENTS
DISTURBANCE_MODES = v18_3.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v18_3.REUSED_SELECTION_SEEDS
EXPECTED_PATH_COUNT = v18_3.EXPECTED_PATH_COUNT
EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT = (
    v18_3.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
)
EXPECTED_ACTOR_FLOOR_PATH_COUNT = v18_3.EXPECTED_ACTOR_FLOOR_PATH_COUNT
EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED = (
    v18_3.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
)

UPPER_RMS_BUDGET = v18_3.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v18_3.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v18_3.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v18_3.LOWER_ACTION_LIMIT
UPPER_WINDOW = v18_3.UPPER_WINDOW
LOWER_WINDOW = v18_3.LOWER_WINDOW
POWER_TOLERANCE = v18_3.POWER_TOLERANCE
EXECUTED_ACTION_LIMIT = v18_3.EXECUTED_ACTION_LIMIT
RECONSTRUCTION_TOLERANCE = v18_3.RECONSTRUCTION_TOLERANCE
BOUND_TOLERANCE = v18_3.BOUND_TOLERANCE
CORRECTION_ABS_MAX_GATE = v18_3.CORRECTION_ABS_MAX_GATE
REFERENCE_CORRECTION_RMS_MAX_GATE = (
    v18_3.REFERENCE_CORRECTION_RMS_MAX_GATE
)
ACTOR_FLOOR_CORRECTION_RMS_MAX_GATE = (
    v18_3.ACTOR_FLOOR_CORRECTION_RMS_MAX_GATE
)
EXECUTED_CORRECTION_RMS_MIN_GATE = (
    v18_3.EXECUTED_CORRECTION_RMS_MIN_GATE
)

VELOCITY_ALPHA = 0.25
VELOCITY_DECAY = 0.75
PROJECTION_TOLERANCE = 1e-9
FEASIBILITY_TOLERANCE = 1e-8
MAXIMUM_PROJECTION_ITERATIONS = 64

CANDIDATES = {
    "joint_mpc_h16_hold": {
        "planning_horizon": 16,
        "forecast_mode": "hold",
    },
    "joint_mpc_h16_damped_velocity": {
        "planning_horizon": 16,
        "forecast_mode": "damped_velocity",
    },
    "joint_mpc_h32_hold": {
        "planning_horizon": 32,
        "forecast_mode": "hold",
    },
    "joint_mpc_h32_damped_velocity": {
        "planning_horizon": 32,
        "forecast_mode": "damped_velocity",
    },
}

SELECTION_CONTRACT = {
    "input": (
        "unchanged v17.8 baseline upper/lower/total action traces; no v17.12 "
        "actor target, observation, reward, or future realized action is "
        "available to the projector"
    ),
    "causality": (
        "at step t the projector uses the current proposed components, its "
        "own projected prefix, and a hold or damped-velocity forecast formed "
        "only from current and previous proposals; it executes one action "
        "pair and replans"
    ),
    "candidate_set": (
        "the Cartesian product of planning horizons 16/32 and fixed causal "
        "forecast modes hold/damped_velocity; velocity alpha 0.25, decay "
        "0.75, filter definitions, budgets, bounds, and solver tolerances are "
        "fixed before reused-path access"
    ),
    "two_stage_audit": (
        "all four candidates receive direct causal endpoint audits on all "
        "120 paths; the lexicographically selected candidate alone is rerun "
        "for an independent full-horizon exact-oracle audit on all 120 paths; "
        "unaudited candidates are not counted as exact-oracle failures"
    ),
    "selection_order": (
        "valid path count, direct joint-feasible count, direct actor-floor "
        "recovery, direct reference preservation, lower maximum reference "
        "correction, lower maximum actor-floor correction, lower global "
        "absolute correction, then candidate id"
    ),
    "advancement_gate": (
        "selected candidate has 120/120 valid and directly joint-feasible "
        "paths, 120/120 independently exact-oracle feasible paths, 113/113 "
        "reference paths preserved, all seven actor-floor paths and both seed "
        "groups recovered, nonzero executed correction on every floor path, "
        "total correction absolute maximum at most 0.05, reference correction "
        "RMS maximum at most 0.01, and actor-floor correction RMS maximum at "
        "most 0.015"
    ),
    "fresh_access_rule": (
        "the reused-path screen never reads a fresh path; a complete pass only "
        "authorizes a separately committed closed-loop validation freeze"
    ),
    "claim_boundary": (
        "label-free causal mechanism selection on reused paths only; no reward, "
        "fresh-seed, learned-policy, no-tradeoff, or manuscript claim"
    ),
}
