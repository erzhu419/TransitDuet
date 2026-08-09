"""Frozen design for the MuJoCo v14.9 asymmetric feasibility screen."""

from __future__ import annotations

import math


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v14_9_asymmetric_feasibility_screen_v1"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_9_state_aligned_feasibility"
)
FROZEN_ALGORITHM_REVISION = "40e86c2f99f7f22d72a89d7ba4dc4799094fd380"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "4be2354c2383778f6c28507684e719d5a83e841aab652623444ffc0f0e3db5ca"
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
        "lower_action_router_mode": "causal_joint_band_projection",
        "lower_action_router_alpha": 0.04,
        "lower_action_router_observe_strength": False,
        "lower_action_router_training_schedule": "constant",
        "lower_action_router_warmup_fraction": 0.0,
        "lower_action_router_ramp_fraction": 0.0,
        "leakage_constraint_scope": "joint_behavior_latent",
        "leakage_cost_mode": "power_excess",
        "upper_constraint_mode": "primal_dual",
        "upper_hf_penalty_coef": 0.0,
        "upper_actor_anchor_coef": 0.0,
        "lower_actor_anchor_coef": 0.0,
    }


ANCHOR_SPEC = {
    **_base_spec(),
    "lower_action_router_strength": 0.0,
    "upper_dual_lr": 0.0,
    "lower_dual_lr": 0.0,
    "checkpoint_score_mode": "mean_reward",
    "checkpoint_constraint_penalty": 0.0,
    "arm_role": "shared_anchor",
}


def _continuation_arm(
    *,
    role: str,
    router_strength: float,
    upper_dual_lr: float,
    lower_dual_lr: float,
    checkpoint_score_mode: str,
    checkpoint_constraint_penalty: float,
) -> dict[str, object]:
    return {
        **_base_spec(),
        "lower_action_router_strength": float(router_strength),
        "upper_dual_lr": float(upper_dual_lr),
        "lower_dual_lr": float(lower_dual_lr),
        "checkpoint_score_mode": str(checkpoint_score_mode),
        "checkpoint_constraint_penalty": float(
            checkpoint_constraint_penalty
        ),
        "arm_role": str(role),
        "learned_frequency_objective": role == "learned",
    }


ARMS = {
    "mean_s000_control": _continuation_arm(
        role="mean_control",
        router_strength=0.0,
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "mean_s050_projection_calibration": _continuation_arm(
        role="projection_calibration",
        router_strength=0.50,
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "feasible_s050_u000_l000_control": _continuation_arm(
        role="matched_control",
        router_strength=0.50,
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "feasible_s050_u000_l030": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_dual_lr=0.0,
        lower_dual_lr=0.30,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "feasible_s050_u000_l100": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_dual_lr=0.0,
        lower_dual_lr=1.00,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "feasible_s050_u000_l300": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_dual_lr=0.0,
        lower_dual_lr=3.00,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "feasible_s050_u010_l100": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_dual_lr=0.10,
        lower_dual_lr=1.00,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "feasible_s050_u020_l100": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_dual_lr=0.20,
        lower_dual_lr=1.00,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "feasible_s050_u020_l300": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_dual_lr=0.20,
        lower_dual_lr=3.00,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "feasible_s050_u030_l100": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_dual_lr=0.30,
        lower_dual_lr=1.00,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "feasible_s050_u030_l300": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_dual_lr=0.30,
        lower_dual_lr=3.00,
        checkpoint_score_mode="latent_behavior_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
}
BASE_CONTROL_ARM = "mean_s000_control"
CALIBRATION_ARM = "mean_s050_projection_calibration"
MATCHED_COMPARATOR_ARM = "feasible_s050_u000_l000_control"
COMPARATOR_ARM = MATCHED_COMPARATOR_ARM
LEARNED_ARMS = tuple(
    arm for arm, arm_spec in ARMS.items()
    if arm_spec["arm_role"] == "learned"
)
CANDIDATE_ARMS = LEARNED_ARMS
EVALUATED_ARMS = (CALIBRATION_ARM, *LEARNED_ARMS)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    1726536284, 2104737544, 3898074391, 93692392,
    1954982810, 1320726304, 886311402, 2902165799,
    2811805952, 156451558, 1597244749, 3750689819,
    3148463739, 3337963620, 2137792248, 1295949384,
)
PRETRAIN_SEEDS = (582327704, 2969175022, 152062172, 3358463371)
PRETRAIN_SELECTION_SEEDS = (4246214572, 3135050262)
CONTINUATION_TRAIN_SEEDS = (2057674552, 3682263298, 539838608, 47921148)
CONTINUATION_SELECTION_SEEDS = (1957419202, 3166770062)
DEVELOPMENT_EVALUATION_SEEDS = (
    1844123307, 1554313725, 1004629616, 1173662313,
    1514166732, 2707403350, 677950690, 1675287618,
)

STEPS = 512
EPISODE_HORIZON = 1000
PRETRAIN_ITERATIONS = 64
CONTINUATION_ITERATIONS = 32
UPPER_PERIOD = 16
HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
LOWER_LF_RMS_BUDGET = 0.0475
UPPER_HF_RMS_BUDGET = 0.075
UPPER_HF_REPORTING_GATE = 0.10
LATENT_UPPER_HF_REPORTING_GATE = 0.10
UPPER_ACTION_SCALE = 1.0
LOWER_ACTION_SCALE = 1.0
LOWER_CONSTRAINT_UPDATE_MODE = "reward_guarded_adam_projection"
UPPER_CONSTRAINT_UPDATE_MODE = "reward_guarded_adam_projection"
CHECKPOINT_SELECTION_MODE = "crossed_conditions"
CHECKPOINT_SMOOTHING_WINDOW = 1
CHECKPOINT_MIN_DELTA = 0.0
CHECKPOINT_EVALUATION_INTERVAL = 4
PRETRAIN_CHECKPOINT_MINIMUM_ITERATION = 15
CONTINUATION_CHECKPOINT_MINIMUM_ITERATION = 7

RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
MAXIMUM_ABSOLUTE_RETURN_DIFFERENCE = 1e-9
MAXIMUM_CALIBRATION_ACTOR_RMS = 1e-12
MINIMUM_PROJECTION_LOWER_REDUCTION_FRACTION = 0.05
MINIMUM_PROJECTION_UPPER_REDUCTION_FRACTION = 0.05
MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION = 0.05
MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION = 0.05
MINIMUM_LATENT_LOWER_DRIFT_REDUCTION_FRACTION = 0.05
MINIMUM_LATENT_UPPER_HF_REDUCTION_FRACTION = 0.05
DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER = 0.01
LOWER_DRIFT_MATERIALITY_FLOOR = (
    DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER
    * LOWER_LF_RMS_BUDGET * LOWER_LF_RMS_BUDGET
)
UPPER_HF_MATERIALITY_FLOOR = (
    DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER
    * UPPER_HF_RMS_BUDGET * UPPER_HF_RMS_BUDGET
)
MINIMUM_EFFECTIVE_LOWER_ACTION_RMS_FRACTION = 0.25
MAXIMUM_ROUTER_CLIP_RATE = 0.05
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
MINIMUM_TRAINED_CHECKPOINT_FRACTION = 1.0
MINIMUM_LEARNED_PARAMETER_RMS = 1e-6
MINIMUM_CHANGED_ACTION_TRACE_CONDITIONS = 1
MINIMUM_CHANGED_ACTION_TRACE_ENVIRONMENTS = len(ENVIRONMENTS)
MINIMUM_STRICT_REWARD_IMPROVEMENT_CONDITIONS = 1
SELECTION_CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 20_000

DEVELOPMENT_DISCLOSURE = (
    "This v14.9 screen was designed after the v14.8 matched preflight showed "
    "that equal upper/lower dual rates operated on incompatible cost scales and "
    "that a trailing checkpoint score could select a state whose own endpoint "
    "diagnostics were poor. The projection calibration is compared "
    "only with a mean-reward zero-strength continuation and must preserve actor "
    "tensors, action, reward, and latent-policy traces exactly. Every learned "
    "arm is compared with a projection- and feasibility-selector-matched zero-"
    "dual control. Upper and lower dual rates are varied independently, and "
    "the upper training budget is aligned with the registered reporting scale. "
    "Selection requires non-identical actors/actions, reward safety, "
    "and reductions in routed and pre-projection lower/upper leakage. All arms "
    "load the same matched checkpoint and use fresh roots. This is development "
    "evidence only."
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


def validate_frozen_design() -> None:
    roles = (
        OPTIMIZER_SEEDS,
        PRETRAIN_SEEDS,
        PRETRAIN_SELECTION_SEEDS,
        CONTINUATION_TRAIN_SEEDS,
        CONTINUATION_SELECTION_SEEDS,
        DEVELOPMENT_EVALUATION_SEEDS,
    )
    flattened = [int(seed) for role in roles for seed in role]
    if len(flattened) != len(set(flattened)):
        raise ValueError("v14.9 seed roles overlap")
    if len(OPTIMIZER_SEEDS) != 16:
        raise ValueError("v14.9 requires 16 optimizer replicates")
    if ARMS[CALIBRATION_ARM]["checkpoint_score_mode"] != "mean_reward":
        raise ValueError("v14.9 calibration must use mean-reward selection")
    matched = ARMS[MATCHED_COMPARATOR_ARM]
    if (
        matched["checkpoint_score_mode"]
        != "latent_behavior_feasibility_first"
        or float(matched["upper_dual_lr"]) != 0.0
        or float(matched["lower_dual_lr"]) != 0.0
    ):
        raise ValueError("v14.9 matched comparator contract drifted")
    for arm in LEARNED_ARMS:
        arm_spec = ARMS[arm]
        if (
            arm_spec["checkpoint_score_mode"]
            != matched["checkpoint_score_mode"]
            or arm_spec["lower_action_router_strength"]
            != matched["lower_action_router_strength"]
            or (
                float(arm_spec["upper_dual_lr"]) <= 0.0
                and float(arm_spec["lower_dual_lr"]) <= 0.0
            )
        ):
            raise ValueError(f"v14.9 learned arm is not matched: {arm}")


validate_frozen_design()
