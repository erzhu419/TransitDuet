"""Frozen design for the MuJoCo v14.7 joint-learned-projection screen."""

from __future__ import annotations

import math


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_7_joint_learned_projection_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_7_joint_band_projection"
)
FROZEN_ALGORITHM_REVISION = "f31811c662411c0cb58db890f950a75d660470f2"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "a253feb248549b8c4f1a20c858ffc8c2bd3088a7a0cfc9645a1a630cc0691459"
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
        "method": "freq_hrl",
        "responsibility_mode": "additive",
        "lower_action_router_alpha": 0.04,
        "lower_action_router_observe_strength": False,
        "leakage_constraint_scope": "joint_behavior",
        "leakage_cost_mode": "power_excess",
        "upper_constraint_mode": "primal_dual",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 0.0,
        "lower_dual_lr": 0.0,
        "checkpoint_score_mode": "mean_reward",
        "checkpoint_constraint_penalty": 0.0,
    }


ANCHOR_SPEC = {
    **_base_spec(),
    "lower_action_router_mode": "causal_joint_band_projection",
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
    upper_dual_lr: float = 0.0,
    lower_dual_lr: float = 0.0,
    learned: bool = False,
) -> dict[str, object]:
    return {
        **_base_spec(),
        "lower_action_router_mode": "causal_joint_band_projection",
        "lower_action_router_strength": float(router_strength),
        "lower_action_router_training_schedule": "constant",
        "lower_action_router_warmup_fraction": 0.0,
        "lower_action_router_ramp_fraction": 0.0,
        "upper_actor_anchor_coef": 0.0,
        "lower_actor_anchor_coef": 0.0,
        "upper_dual_lr": float(upper_dual_lr),
        "lower_dual_lr": float(lower_dual_lr),
        "checkpoint_score_mode": (
            "behavior_robust" if learned else "mean_reward"
        ),
        "checkpoint_constraint_penalty": 10.0 if learned else 0.0,
        "learned_frequency_objective": bool(learned),
    }


ARMS = {
    "joint_s000_control": _continuation_arm(
        router_strength=0.0,
    ),
    "joint_s050_calibration": _continuation_arm(
        router_strength=0.50,
    ),
    "joint_s025_pd_u001_l001": _continuation_arm(
        router_strength=0.25,
        upper_dual_lr=0.01,
        lower_dual_lr=0.01,
        learned=True,
    ),
    "joint_s050_pd_u001_l001": _continuation_arm(
        router_strength=0.50,
        upper_dual_lr=0.01,
        lower_dual_lr=0.01,
        learned=True,
    ),
    "joint_s075_pd_u001_l001": _continuation_arm(
        router_strength=0.75,
        upper_dual_lr=0.01,
        lower_dual_lr=0.01,
        learned=True,
    ),
    "joint_s050_pd_u003_l003": _continuation_arm(
        router_strength=0.50,
        upper_dual_lr=0.03,
        lower_dual_lr=0.03,
        learned=True,
    ),
    "joint_s050_pd_u010_l010": _continuation_arm(
        router_strength=0.50,
        upper_dual_lr=0.10,
        lower_dual_lr=0.10,
        learned=True,
    ),
}
COMPARATOR_ARM = "joint_s000_control"
CALIBRATION_ARM = "joint_s050_calibration"
CANDIDATE_ARMS = tuple(arm for arm in ARMS if arm != COMPARATOR_ARM)
LEARNED_ARMS = tuple(
    arm for arm, arm_spec in ARMS.items()
    if bool(arm_spec["learned_frequency_objective"])
)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    2341827529, 955030098, 581891137, 2499414328,
    179014918, 1377812162, 792752490, 3590851901,
    1241198292, 2049445494, 1137791003, 927106486,
    3333355612, 3555320995, 3943633411, 1373569846,
)
PRETRAIN_SEEDS = (2741812134, 1125683721, 2116528915, 2689615359)
PRETRAIN_SELECTION_SEEDS = (3912102474, 2838915186)
CONTINUATION_TRAIN_SEEDS = (1436129519, 4233141023, 2180024276, 3674070659)
CONTINUATION_SELECTION_SEEDS = (1410488695, 1155757563)
DEVELOPMENT_EVALUATION_SEEDS = (
    3642591014, 723970234, 1520018692, 2308656598,
    186166092, 2602553181, 1772262377, 2605203890,
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

RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
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
MINIMUM_LEARNED_PARAMETER_RMS = 1e-6
MINIMUM_CHANGED_ACTION_TRACE_CONDITIONS = 1
MINIMUM_CHANGED_ACTION_TRACE_ENVIRONMENTS = len(ENVIRONMENTS)
MINIMUM_STRICT_REWARD_IMPROVEMENT_CONDITIONS = 1
SELECTION_CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 20_000

DEVELOPMENT_DISCLOSURE = (
    "This v14.7 screen was designed after v14.6 showed that one-sided "
    "function-preserving transfer was a responsibility-coordinate diagnostic, "
    "not learned control, and violated Hopper's upper-HF budget. Every arm uses "
    "a causal joint projection whose transfer is LPF32(lower)-HPF8(upper) and "
    "whose two responsibilities exactly reconstruct the pre-split action. The "
    "projection-only calibration arm must remain pathwise identical to control. "
    "Learned arms activate both upper and lower primal-dual objectives, must "
    "produce non-identical trained parameters and action traces, and must pass "
    "reward, lower-LF, upper-HF, activity, and reconstruction gates. All arms "
    "load the same matched checkpoint and use identical fresh roots. The screen "
    "is development evidence only."
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
