"""Frozen design for the MuJoCo v14.6 conservative-transfer screen."""

from __future__ import annotations

import math


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_6_conservative_transfer_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_6_conservative_router_transfer"
)
FROZEN_ALGORITHM_REVISION = "a20ddbcb28aa0244e5fb337c7b8261cdf93e2f8a"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "d29b42c228dbdffdfe1f5591c8c641f4491426da85727c2d1273708a04e427a1"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (
    *TRAINING_DISTURBANCE_MODES,
    "ood_chirp",
)


def _base_spec() -> dict[str, object]:
    return {
        "method": "freq_hrl_no_leakage",
        "responsibility_mode": "additive",
        "lower_action_router_alpha": 0.04,
        "lower_action_router_observe_strength": False,
        "leakage_constraint_scope": "joint_behavior",
        "leakage_cost_mode": "power_excess",
        "upper_constraint_mode": "disabled",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 0.0,
        "lower_dual_lr": 0.0,
        "checkpoint_score_mode": "mean_reward",
        "checkpoint_constraint_penalty": 0.0,
    }


ANCHOR_SPEC = {
    **_base_spec(),
    "lower_action_router_mode": "causal_ema_conservative_transfer",
    "lower_action_router_strength": 0.0,
    "lower_action_router_training_schedule": "constant",
    "lower_action_router_warmup_fraction": 0.0,
    "lower_action_router_ramp_fraction": 0.0,
    "upper_actor_anchor_coef": 0.0,
    "lower_actor_anchor_coef": 0.0,
}


def _continuation_arm(
    *,
    router_strength: float,
) -> dict[str, object]:
    return {
        **_base_spec(),
        "lower_action_router_mode": "causal_ema_conservative_transfer",
        "lower_action_router_strength": float(router_strength),
        "lower_action_router_training_schedule": "constant",
        "lower_action_router_warmup_fraction": 0.0,
        "lower_action_router_ramp_fraction": 0.0,
        "upper_actor_anchor_coef": 0.0,
        "lower_actor_anchor_coef": 0.0,
    }


ARMS = {
    "conservative_s000_control": _continuation_arm(
        router_strength=0.0,
    ),
    "conservative_s0025": _continuation_arm(
        router_strength=0.025,
    ),
    "conservative_s0050": _continuation_arm(
        router_strength=0.05,
    ),
    "conservative_s0075": _continuation_arm(
        router_strength=0.075,
    ),
    "conservative_s0100": _continuation_arm(
        router_strength=0.10,
    ),
    "conservative_s0125": _continuation_arm(
        router_strength=0.125,
    ),
    "conservative_s0150": _continuation_arm(
        router_strength=0.15,
    ),
    "conservative_s0200": _continuation_arm(
        router_strength=0.20,
    ),
}
COMPARATOR_ARM = "conservative_s000_control"
CANDIDATE_ARMS = tuple(arm for arm in ARMS if arm != COMPARATOR_ARM)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    3444335686, 793385307, 2838959313, 3604527835,
    4184033156, 1439262987, 1607848361, 2912892372,
    251561761, 3535643925, 362893922, 128515602,
    2996362467, 833649984, 966058201, 2837914129,
)
PRETRAIN_SEEDS = (4035450813, 860639855, 1097806097, 3970896080)
PRETRAIN_SELECTION_SEEDS = (3425095182, 24066168)
CONTINUATION_TRAIN_SEEDS = (4148201941, 2017916074, 2108385976, 2704323268)
CONTINUATION_SELECTION_SEEDS = (529967094, 1589605452)
DEVELOPMENT_EVALUATION_SEEDS = (
    1939152560, 1197067070, 3076081842, 601758856,
    981851985, 4021900705, 3608009246, 2316247433,
)

STEPS = 512
EPISODE_HORIZON = 1000
PRETRAIN_ITERATIONS = 64
CONTINUATION_ITERATIONS = 32
UPPER_PERIOD = 16
HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
LOWER_LF_RMS_BUDGET = 0.05
UPPER_HF_RMS_BUDGET = 0.04
UPPER_HF_REPORTING_GATE = 0.10
UPPER_ACTION_SCALE = 1.0
LOWER_ACTION_SCALE = 1.0
LOWER_CONSTRAINT_UPDATE_MODE = "reward_guarded_adam_projection"
UPPER_CONSTRAINT_UPDATE_MODE = "reward_guarded_adam_projection"
CHECKPOINT_SELECTION_MODE = "crossed_conditions"
CHECKPOINT_SMOOTHING_WINDOW = 4
CHECKPOINT_MIN_DELTA = 1e-3
CHECKPOINT_EVALUATION_INTERVAL = 4
PRETRAIN_CHECKPOINT_MINIMUM_ITERATION = 15
CONTINUATION_CHECKPOINT_MINIMUM_ITERATION = 7

RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.0
MAXIMUM_ABSOLUTE_RETURN_DIFFERENCE = 1e-9
MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION = 0.05
MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION = 0.05
DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER = 0.01
DRIFT_MATERIALITY_FLOOR = (
    DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER
    * LOWER_LF_RMS_BUDGET * LOWER_LF_RMS_BUDGET
)
MINIMUM_EFFECTIVE_LOWER_ACTION_RMS_FRACTION = 0.25
MAXIMUM_ROUTER_CLIP_RATE = 0.05
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
MINIMUM_TRAINED_CHECKPOINT_FRACTION = 1.0
MINIMUM_STRICT_RESPONSIBILITY_IMPROVEMENT_CONDITIONS = 1
MINIMUM_STRICT_RAW_IMPROVEMENT_CONDITIONS = 1
SELECTION_CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 20_000

DEVELOPMENT_DISCLOSURE = (
    "This v14.6 screen was designed after the v14.5 paired high-pass screen "
    "failed because routing changed the executed action before learning. Every "
    "anchor and continuation uses one conservative router state contract. The "
    "zero-strength comparator and seven transfer strengths load the same exact "
    "serialized checkpoint and use identical fresh continuation roots, selection "
    "roots, iteration budgets, and held-out paths. Removed lower-frequency action "
    "is transferred to upper responsibility while the pre-split environment "
    "action is executed. Both phases exclude untrained checkpoints. Exact action, "
    "reward, latent-policy trace hashes, and selected parameter hashes are audited. "
    "The screen is development evidence only."
)


def expected_router_training_strengths(arm: str) -> list[float]:
    arm_spec = ARMS[str(arm)]
    target = float(arm_spec["lower_action_router_strength"])
    schedule = str(arm_spec["lower_action_router_training_schedule"])
    warmup = float(arm_spec["lower_action_router_warmup_fraction"])
    ramp = float(arm_spec["lower_action_router_ramp_fraction"])
    values = []
    for iteration in range(CONTINUATION_ITERATIONS):
        if schedule == "constant" or math.isclose(target, 0.0):
            values.append(target)
            continue
        progress = float(iteration + 1) / float(CONTINUATION_ITERATIONS)
        phase = min(max((progress - warmup) / ramp, 0.0), 1.0)
        values.append(target * phase)
    return values
