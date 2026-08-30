"""Frozen design for MuJoCo v17.13 causal actor-target distillation."""

from __future__ import annotations

from scripts import (  # noqa: F401
    mujoco_v17_12_nearest_feasible_action_oracle_spec as v17_12,
)


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_13_causal_actor_adapter_v1"
EVIDENCE_ROLE = (
    "grouped_reused_path_causal_actor_target_distillation_not_confirmatory"
)
FROZEN_CORE_REVISION = "f7b2254d960824ff02b6b6fc69dd6f1202ee2093"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "a4bdf5eb8c436114ee28556bbc978910b4c1cfcd10bf9a29caf747c48cae27d2"
)
SOURCE_DATASET_RUN = v17_12.SOURCE_DATASET_RUN
SOURCE_TARGET_RUN = (
    "mujoco_v17_12_nearest_feasible_action_oracle_20260831_r1"
)
ENVIRONMENTS = v17_12.ENVIRONMENTS
DISTURBANCE_MODES = v17_12.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v17_12.REUSED_SELECTION_SEEDS
UPPER_RMS_BUDGET = v17_12.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v17_12.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_12.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v17_12.LOWER_ACTION_LIMIT
UPPER_WINDOW = v17_12.UPPER_WINDOW
LOWER_WINDOW = v17_12.LOWER_WINDOW
POWER_TOLERANCE = v17_12.POWER_TOLERANCE
EXPECTED_PATH_COUNT = v17_12.EXPECTED_PATH_COUNT
EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT = (
    v17_12.EXPECTED_ORACLE_FEASIBLE_PATH_COUNT
)
EXPECTED_ACTOR_FLOOR_PATH_COUNT = v17_12.EXPECTED_ACTOR_FLOOR_PATH_COUNT
EXPECTED_ACTOR_FLOOR_ENVIRONMENT = "Hopper-v5"
EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED = {
    2802248628: 2,
    294864529: 5,
}

FIR_WINDOWS = (4, 8, 16, 32, 48)
RIDGE_PENALTIES = (1e-6, 1e-4, 1e-2)
ACTOR_FLOOR_PATH_WEIGHTS = (1.0, 4.0, 16.0, 64.0, 256.0)
OUTPUT_GAINS = (0.50, 1.00, 1.50, 2.00)
CORRECTION_ABS_LIMITS = (0.010, 0.025, 0.050)
FEATURE_SCALE_FLOOR = 1e-8
COMPONENT_SUM_LIMIT = UPPER_ACTION_LIMIT + LOWER_ACTION_LIMIT
EXECUTED_ACTION_LIMIT = 1.0

PREFILTER_TOP_PER_RANKING = 16
TARGET_NORMALIZED_MSE_GATE = 0.75
REFERENCE_FEASIBLE_CORRECTION_RMS_MAX_GATE = 0.01
EXECUTED_CORRECTION_RMS_MIN_GATE = 1e-8
CORRECTION_BOUND_TOLERANCE = 1e-10
TARGET_ALIGNMENT_TOLERANCE = 1e-10

SELECTION_CONTRACT = {
    "input": (
        "unchanged server-only v17.8 action paths plus the seven server-only "
        "v17.12 nearest-feasible total-action targets"
    ),
    "learner": (
        "environment-specific path-balanced weighted causal FIR residual "
        "adapters using only current and past upper/lower actor proposals"
    ),
    "grouping": (
        "eight leave-one-evaluation-seed-out folds; all five disturbance "
        "modes for the held seed remain outside fitting"
    ),
    "shared_candidate": (
        "one window, ridge penalty, actor-floor path weight, output gain, and "
        "correction trust limit selected jointly across environments"
    ),
    "prefilter": (
        "union of frozen target-fidelity, preservation, and balanced rankings; "
        "only that fixed-size union receives expensive full-horizon oracles"
    ),
    "advancement_gate": (
        "120/120 valid, 113/113 reference-feasible paths preserved, all seven "
        "actor-floor paths and both actor-floor seed groups recovered, target "
        "normalized MSE at most 0.75, reference-feasible correction RMS at "
        "most 0.01, and nonzero post-clipping target and adapter action change "
        "on every floor path"
    ),
    "fresh_access_rule": (
        "no fresh path is read by this selector; a complete reused-path gate "
        "only authorizes a separately frozen closed-loop fresh validation"
    ),
    "claim_boundary": (
        "reused-path causal action-adapter development only; no reward, online "
        "policy-learning, fresh-seed, or manuscript performance claim"
    ),
}
