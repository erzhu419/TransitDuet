"""Frozen design for MuJoCo v17.12 nearest feasible action targets."""

from __future__ import annotations

from scripts import (  # noqa: F401
    mujoco_v17_8_causal_fir_distillation_spec as v17_8,
)


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_12_nearest_feasible_action_v1"
EVIDENCE_ROLE = (
    "post_router_boundary_reused_path_actor_target_oracle_not_confirmatory"
)
FROZEN_CORE_REVISION = "3ca8ee6d2709e77af7b7c4d022cbeb83886d0a75"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "718a7dd61ea62fa21b52531299a2e0d2e2b22175891f7a7fbdf03b7f58a5410a"
)
SOURCE_DATASET_RUN = "mujoco_v17_8_causal_fir_dataset_20260831_r1"
ENVIRONMENTS = v17_8.ENVIRONMENTS
DISTURBANCE_MODES = v17_8.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v17_8.REUSED_SELECTION_SEEDS
UPPER_RMS_BUDGET = v17_8.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v17_8.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_8.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v17_8.LOWER_ACTION_LIMIT
UPPER_WINDOW = v17_8.UPPER_WINDOW
LOWER_WINDOW = v17_8.LOWER_WINDOW
POWER_TOLERANCE = v17_8.POWER_TOLERANCE
EXPECTED_PATH_COUNT = v17_8.REUSED_EXPECTED_PATH_COUNT
EXPECTED_ORACLE_FEASIBLE_PATH_COUNT = 113
EXPECTED_ACTOR_FLOOR_PATH_COUNT = 7

TOTAL_ACTION_LIMIT = 1.0
CONVERGENCE_TOLERANCE = 1e-10
FEASIBILITY_TOLERANCE = POWER_TOLERANCE
MAX_PROJECTION_ITERATIONS = 10000
PRESERVED_PATH_CORRECTION_TOLERANCE = 1e-12
MAX_ACTOR_FLOOR_TOTAL_CORRECTION_RMS = 0.05
MAX_ACTOR_FLOOR_TOTAL_CORRECTION_ABS = 0.25

SELECTION_CONTRACT = {
    "input": (
        "the unchanged server-only v17.8 reused-path panel and v17.6 oracle "
        "component targets; no fresh path access"
    ),
    "frequency_projection": (
        "Dykstra Euclidean projection of the oracle upper/lower pair onto "
        "component boxes and exact full-horizon HPF8/LPF32 balls"
    ),
    "deployment_diagnostic": (
        "the seven actor-floor paths are also projected with nominal total "
        "action constrained to the environment unit box; this profile does "
        "not select the next target"
    ),
    "advancement_gate": (
        "all 120 frequency-only targets feasible, all 113 previously feasible "
        "paths unchanged, exactly seven nonzero actor targets, actor-floor "
        "total correction RMS at most 0.05 and maximum absolute correction at "
        "most 0.25"
    ),
    "next_step": (
        "a passing result authorizes grouped causal actor-target distillation "
        "on reused paths before any fresh validation"
    ),
    "claim_boundary": (
        "acausal reused-path target construction only; no online policy, "
        "reward, fresh-seed, or manuscript performance claim"
    ),
}
