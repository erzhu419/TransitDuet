"""Frozen fresh-anchor bank for the v14.29 portfolio confirmation."""

from __future__ import annotations

from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as base


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_29_fresh_anchor_bank_v1"
FROZEN_CORE_PROTOCOL_VERSION = base.FROZEN_CORE_PROTOCOL_VERSION
FROZEN_ALGORITHM_REVISION = "fc7fa8d8c1e55325af9cb32efece3e0cfc2bbd3c"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "02f3ba95376021dff0aa11f30d46dd6159e63b55a1d2678d6011ea350745af39"
)
REQUIRE_EXPLICIT_PROTOCOL_SELECTION = True
PREREGISTRATION_STATUS = (
    "frozen_before_v14_29_fresh_anchor_training_outcome_access"
)
DEVELOPMENT_DISCLOSURE = (
    "Fresh optimizer, training, selection, and evaluation seeds are used only "
    "to construct the source-bound v14.29 anchor bank. Portfolio design and "
    "validation paths are separately frozen and disjoint."
)

ENVIRONMENTS = base.ENVIRONMENTS
TRAINING_DISTURBANCE_MODES = base.TRAINING_DISTURBANCE_MODES
EVALUATION_DISTURBANCE_MODES = base.EVALUATION_DISTURBANCE_MODES
OPTIMIZER_SEEDS = (
    2978317753, 392459795, 2303062597, 422427492,
    348909203, 1155514759, 1693716559, 981850268,
    4097799577, 872933768, 1553837587, 3230217838,
    2518107, 3022864709, 3592646921, 821573137,
)
PRETRAIN_SEEDS = (3446942493, 3269057258, 4078423649, 2350163227)
PRETRAIN_SELECTION_SEEDS = (
    4129680462, 3692230181, 2454832807, 4285569731,
)
DEVELOPMENT_EVALUATION_SEEDS = (
    1547518038, 1200657750, 2291698606, 865878995, 2195645998,
)

ANCHOR_SPEC = dict(base.ANCHOR_SPEC)
ANCHOR_SPEC.update({
    "lower_action_router_strength": 0.5,
    "lower_action_router_training_schedule": "constant",
})
ARMS = base.ARMS
BASE_CONTROL_ARM = base.BASE_CONTROL_ARM
CALIBRATION_ARM = base.CALIBRATION_ARM
MATCHED_COMPARATOR_ARM = base.MATCHED_COMPARATOR_ARM
LEARNED_ARMS = base.LEARNED_ARMS
GROUPWISE_ARMS = base.GROUPWISE_ARMS
PRETRAIN_ITERATIONS = base.PRETRAIN_ITERATIONS
PRETRAIN_CHECKPOINT_MINIMUM_ITERATION = (
    base.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION
)
STEPS = base.STEPS
EPISODE_HORIZON = base.EPISODE_HORIZON
UPPER_PERIOD = base.UPPER_PERIOD
HIDDEN_DIM = base.HIDDEN_DIM
LEARNING_RATE = base.LEARNING_RATE
LOWER_LF_RMS_BUDGET = base.LOWER_LF_RMS_BUDGET
UPPER_HF_RMS_BUDGET = base.UPPER_HF_RMS_BUDGET
UPPER_CONSTRAINT_UPDATE_MODE = base.UPPER_CONSTRAINT_UPDATE_MODE
LOWER_CONSTRAINT_UPDATE_MODE = base.LOWER_CONSTRAINT_UPDATE_MODE
UPPER_ACTION_SCALE = base.UPPER_ACTION_SCALE
LOWER_ACTION_SCALE = base.LOWER_ACTION_SCALE
CHECKPOINT_SELECTION_MODE = base.CHECKPOINT_SELECTION_MODE
CHECKPOINT_SMOOTHING_WINDOW = base.CHECKPOINT_SMOOTHING_WINDOW
CHECKPOINT_MIN_DELTA = base.CHECKPOINT_MIN_DELTA
CHECKPOINT_EVALUATION_INTERVAL = base.CHECKPOINT_EVALUATION_INTERVAL
DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS = (
    base.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS
)


def __getattr__(name: str):
    return getattr(base, name)


def validate() -> None:
    roles = (
        OPTIMIZER_SEEDS,
        PRETRAIN_SEEDS,
        PRETRAIN_SELECTION_SEEDS,
        DEVELOPMENT_EVALUATION_SEEDS,
    )
    flattened = [int(seed) for role in roles for seed in role]
    if len(OPTIMIZER_SEEDS) != 16 or len(ENVIRONMENTS) != 3:
        raise RuntimeError("v14.29 requires sixteen seeds across three environments")
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("v14.29 fresh-anchor seed roles overlap")
    if set(OPTIMIZER_SEEDS) & set(base.OPTIMIZER_SEEDS):
        raise RuntimeError("v14.29 optimizer seeds overlap v14.17 development")
    if ANCHOR_SPEC["lower_action_router_strength"] != 0.5:
        raise RuntimeError("v14.29 anchors must use router strength 0.5")
    if ANCHOR_SPEC["lower_action_router_mode"] != "causal_joint_band_projection":
        raise RuntimeError("v14.29 anchors require the conservative joint router")


validate()
