"""Frozen grouped screen for the MuJoCo v18.2 state-conditioned actor."""

from __future__ import annotations

from scripts import mujoco_v17_13_causal_actor_adapter_spec as v17_13
from scripts import mujoco_v18_1_state_actor_dataset_spec as v18_1


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v18_2_state_conditioned_actor_v1"
EVIDENCE_ROLE = (
    "grouped_reused_path_state_conditioned_actor_selection_not_confirmatory"
)
FROZEN_CORE_REVISION = "6ebf63c77c5c8ecf2e0784b7361eb90a6d71caf9"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "d5842e0972d59182be55881b52de9b4ac074b5b355aa1b4c929e4b7f0a849099"
)

STATE_DATASET_RUN = "mujoco_v18_1_state_actor_dataset_20260831_r1"
REFERENCE_DATASET_RUN = "mujoco_v17_8_causal_fir_dataset_20260831_r1"
TARGET_RUN = "mujoco_v17_12_nearest_feasible_action_oracle_20260831_r1"
ENVIRONMENTS = v18_1.ENVIRONMENTS
DISTURBANCE_MODES = v18_1.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v18_1.REUSED_SELECTION_SEEDS
EXPECTED_PATH_COUNT = v18_1.EXPECTED_PATH_COUNT

UPPER_RMS_BUDGET = v17_13.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v17_13.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_13.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v17_13.LOWER_ACTION_LIMIT
UPPER_WINDOW = v17_13.UPPER_WINDOW
LOWER_WINDOW = v17_13.LOWER_WINDOW
POWER_TOLERANCE = v17_13.POWER_TOLERANCE
COMPONENT_SUM_LIMIT = v17_13.COMPONENT_SUM_LIMIT
EXECUTED_ACTION_LIMIT = v17_13.EXECUTED_ACTION_LIMIT
TARGET_ALIGNMENT_TOLERANCE = v17_13.TARGET_ALIGNMENT_TOLERANCE
STATE_TRACE_ALIGNMENT_TOLERANCE = 5e-7
CORRECTION_BOUND_TOLERANCE = v17_13.CORRECTION_BOUND_TOLERANCE
EXECUTED_CORRECTION_RMS_MIN_GATE = (
    v17_13.EXECUTED_CORRECTION_RMS_MIN_GATE
)
TARGET_NORMALIZED_MSE_GATE = v17_13.TARGET_NORMALIZED_MSE_GATE
REFERENCE_FEASIBLE_CORRECTION_RMS_MAX_GATE = (
    v17_13.REFERENCE_FEASIBLE_CORRECTION_RMS_MAX_GATE
)
EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT = (
    v17_13.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
)
EXPECTED_ACTOR_FLOOR_PATH_COUNT = v17_13.EXPECTED_ACTOR_FLOOR_PATH_COUNT
EXPECTED_ACTOR_FLOOR_ENVIRONMENT = (
    v17_13.EXPECTED_ACTOR_FLOOR_ENVIRONMENT
)
EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED = (
    v17_13.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
)

PROPOSAL_WINDOWS = (1, 8)
HIDDEN_DIMS = (32, 64)
HIDDEN_LAYERS = 2
ACTOR_FLOOR_PATH_WEIGHTS = (64.0, 256.0)
CORRECTION_ABS_LIMITS = (0.010, 0.025)
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-5
TRAINING_EPOCHS = 120
FEATURE_SCALE_FLOOR = 1e-6
EXPECTED_CANDIDATE_COUNT = (
    len(PROPOSAL_WINDOWS)
    * len(HIDDEN_DIMS)
    * len(ACTOR_FLOOR_PATH_WEIGHTS)
    * len(CORRECTION_ABS_LIMITS)
)

SELECTION_CONTRACT = {
    "input": (
        "unchanged v17.8 total-action references, v18.1 causal lower-policy "
        "states, and seven separate v17.12 nearest-feasible targets"
    ),
    "online_features": (
        "current lower-policy state plus current and past upper/lower action "
        "proposals; actor-floor labels and oracle targets are never features"
    ),
    "grouping": (
        "eight leave-one-seed-out folds; all five disturbance modes of the "
        "held seed remain outside that fold's training data"
    ),
    "candidate_set": (
        "the complete frozen 16-member two-hidden-layer MLP grid; every "
        "candidate receives exact full-horizon responsibility oracles"
    ),
    "selection_order": (
        "joint-feasible path count, actor-floor recovery, reference-feasible "
        "preservation, complete floor seed groups, target fidelity, reference "
        "correction, correction magnitude, then candidate id"
    ),
    "advancement_gate": (
        "120/120 valid, 113/113 reference-feasible paths preserved, all seven "
        "actor-floor paths and both floor seed groups recovered, target "
        "normalized MSE at most 0.75, reference correction RMS at most 0.01, "
        "and nonzero post-clipping correction on all floor paths"
    ),
    "fresh_access_rule": (
        "no fresh path is read; a complete reused-panel gate only authorizes "
        "a separately frozen closed-loop validation"
    ),
    "failure_rule": (
        "if no candidate passes, do not tune this panel further; diagnose the "
        "remaining state/action support before defining another mechanism"
    ),
    "claim_boundary": (
        "grouped reused-path state-conditioned distillation only; no reward, "
        "closed-loop, fresh-seed, online-learning, or manuscript claim"
    ),
}
