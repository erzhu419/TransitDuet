"""Frozen design for the MuJoCo v14.10 deployment-aligned screen."""

from __future__ import annotations

import math


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v14_10_deployment_aligned_screen_v1"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_10_deployment_aligned_constraints"
)
FROZEN_ALGORITHM_REVISION = "1704ce3f5f867ab493899d424f1398557cc4a625"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "dbc8cc11399a4baad5cb7c1231860c3911293398f553cd2426670fd08636ea80"
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
        "upper_constraint_mode": "disabled",
        "upper_hf_penalty_coef": 0.0,
        "upper_actor_anchor_coef": 0.0,
        "lower_actor_anchor_coef": 0.0,
        "upper_deployment_frequency_rms_budget": 0.001,
        "lower_deployment_frequency_rms_budget": 0.001,
    }


ANCHOR_SPEC = {
    **_base_spec(),
    "lower_action_router_strength": 0.0,
    "upper_dual_lr": 0.0,
    "lower_dual_lr": 0.0,
    "upper_deployment_frequency_dual_lr": 0.0,
    "lower_deployment_frequency_dual_lr": 0.0,
    "upper_deployment_frequency_lambda_init": 0.0,
    "lower_deployment_frequency_lambda_init": 0.0,
    "upper_deployment_frequency_step_scale": 1.0,
    "lower_deployment_frequency_step_scale": 1.0,
    "upper_deployment_frequency_reference_reduction_fraction": 0.0,
    "lower_deployment_frequency_reference_reduction_fraction": 0.0,
    "checkpoint_score_mode": "mean_reward",
    "checkpoint_constraint_penalty": 0.0,
    "arm_role": "shared_anchor",
}


def _continuation_arm(
    *,
    role: str,
    router_strength: float,
    upper_deployment_dual_lr: float,
    lower_deployment_dual_lr: float,
    upper_step_scale: float,
    lower_step_scale: float,
    upper_reduction_fraction: float,
    lower_reduction_fraction: float,
    checkpoint_score_mode: str,
    checkpoint_constraint_penalty: float,
) -> dict[str, object]:
    return {
        **_base_spec(),
        "lower_action_router_strength": float(router_strength),
        "upper_dual_lr": 0.0,
        "lower_dual_lr": 0.0,
        "upper_deployment_frequency_dual_lr": float(
            upper_deployment_dual_lr
        ),
        "lower_deployment_frequency_dual_lr": float(
            lower_deployment_dual_lr
        ),
        "upper_deployment_frequency_lambda_init": (
            0.1 if float(upper_deployment_dual_lr) > 0.0 else 0.0
        ),
        "lower_deployment_frequency_lambda_init": (
            0.1 if float(lower_deployment_dual_lr) > 0.0 else 0.0
        ),
        "upper_deployment_frequency_step_scale": float(upper_step_scale),
        "lower_deployment_frequency_step_scale": float(lower_step_scale),
        "upper_deployment_frequency_reference_reduction_fraction": float(
            upper_reduction_fraction
        ),
        "lower_deployment_frequency_reference_reduction_fraction": float(
            lower_reduction_fraction
        ),
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
        upper_deployment_dual_lr=0.0,
        lower_deployment_dual_lr=0.0,
        upper_step_scale=1.0,
        lower_step_scale=1.0,
        upper_reduction_fraction=0.0,
        lower_reduction_fraction=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "mean_s050_projection_calibration": _continuation_arm(
        role="projection_calibration",
        router_strength=0.50,
        upper_deployment_dual_lr=0.0,
        lower_deployment_dual_lr=0.0,
        upper_step_scale=1.0,
        lower_step_scale=1.0,
        upper_reduction_fraction=0.0,
        lower_reduction_fraction=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "paired_s050_d000_control": _continuation_arm(
        role="matched_control",
        router_strength=0.50,
        upper_deployment_dual_lr=0.0,
        lower_deployment_dual_lr=0.0,
        upper_step_scale=1.0,
        lower_step_scale=1.0,
        upper_reduction_fraction=0.0,
        lower_reduction_fraction=0.0,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "relative_s050_j001_s1_r05": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_deployment_dual_lr=0.01,
        lower_deployment_dual_lr=0.01,
        upper_step_scale=1.0,
        lower_step_scale=1.0,
        upper_reduction_fraction=0.05,
        lower_reduction_fraction=0.05,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "relative_s050_j005_s1_r05": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_deployment_dual_lr=0.05,
        lower_deployment_dual_lr=0.05,
        upper_step_scale=1.0,
        lower_step_scale=1.0,
        upper_reduction_fraction=0.05,
        lower_reduction_fraction=0.05,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "relative_s050_j005_s3_r05": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_deployment_dual_lr=0.05,
        lower_deployment_dual_lr=0.05,
        upper_step_scale=3.0,
        lower_step_scale=3.0,
        upper_reduction_fraction=0.05,
        lower_reduction_fraction=0.05,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "relative_s050_asym_u003_l008_s13_r05": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_deployment_dual_lr=0.03,
        lower_deployment_dual_lr=0.08,
        upper_step_scale=1.0,
        lower_step_scale=3.0,
        upper_reduction_fraction=0.05,
        lower_reduction_fraction=0.05,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "relative_s050_upper_u005_s1_r05": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_deployment_dual_lr=0.05,
        lower_deployment_dual_lr=0.0,
        upper_step_scale=1.0,
        lower_step_scale=1.0,
        upper_reduction_fraction=0.05,
        lower_reduction_fraction=0.0,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "relative_s050_lower_l005_s1_r05": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_deployment_dual_lr=0.0,
        lower_deployment_dual_lr=0.05,
        upper_step_scale=1.0,
        lower_step_scale=1.0,
        upper_reduction_fraction=0.0,
        lower_reduction_fraction=0.05,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
    "relative_s050_j005_s3_r10": _continuation_arm(
        role="learned",
        router_strength=0.50,
        upper_deployment_dual_lr=0.05,
        lower_deployment_dual_lr=0.05,
        upper_step_scale=3.0,
        lower_step_scale=3.0,
        upper_reduction_fraction=0.10,
        lower_reduction_fraction=0.10,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
        checkpoint_constraint_penalty=10.0,
    ),
}
BASE_CONTROL_ARM = "mean_s000_control"
CALIBRATION_ARM = "mean_s050_projection_calibration"
MATCHED_COMPARATOR_ARM = "paired_s050_d000_control"
COMPARATOR_ARM = MATCHED_COMPARATOR_ARM
LEARNED_ARMS = tuple(
    arm for arm, arm_spec in ARMS.items()
    if arm_spec["arm_role"] == "learned"
)
CANDIDATE_ARMS = LEARNED_ARMS
EVALUATED_ARMS = (CALIBRATION_ARM, *LEARNED_ARMS)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    2706173366, 929158302, 3897052084, 1118998580,
    3375040175, 3394131150, 1720146203, 1989774263,
    3217438138, 4172654258, 3444271738, 4004248010,
    1112764540, 578718003, 674689624, 1660434559,
)
PRETRAIN_SEEDS = (1547581733, 2651787860, 3795183683, 1429177133)
PRETRAIN_SELECTION_SEEDS = (1191857505, 1804014777)
CONTINUATION_TRAIN_SEEDS = (1509443718, 1458455199, 977947963, 314274618)
CONTINUATION_SELECTION_SEEDS = (4094477409, 2011443768)
DEVELOPMENT_EVALUATION_SEEDS = (
    20796979, 1625837976, 3119366621, 1368940167,
    11428093, 741726722, 2603094458, 2843193718,
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
CONTINUATION_CHECKPOINT_MINIMUM_ITERATION = -1
ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION = 7

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
    "This v14.10 screen was designed after the v14.9 preflight showed that "
    "sampled Gaussian rollout-action constraints did not reliably improve the "
    "deterministic actor-mean deployment traces used by evaluation. The "
    "projection calibration is compared "
    "only with a mean-reward zero-strength continuation and must preserve actor "
    "tensors, action, reward, and latent-policy traces exactly. Every learned "
    "arm is compared with a projection- and paired-relative-selector-matched "
    "zero-dual control. Its projected update constrains deterministic actor-mean "
    "frequency power on the same states and relative to the frozen paired "
    "checkpoint. Upper and lower deployment dual rates, step scales, and target "
    "reductions are varied independently. "
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


def checkpoint_minimum_iteration(arm: str) -> int:
    """Permit anchor fallback only for paired-relative selection arms."""

    return (
        CONTINUATION_CHECKPOINT_MINIMUM_ITERATION
        if ARMS[str(arm)]["checkpoint_score_mode"]
        == "paired_relative_frequency_feasibility_first"
        else ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION
    )


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
        raise ValueError("v14.10 seed roles overlap")
    if len(OPTIMIZER_SEEDS) != 16:
        raise ValueError("v14.10 requires 16 optimizer replicates")
    if ARMS[CALIBRATION_ARM]["checkpoint_score_mode"] != "mean_reward":
        raise ValueError("v14.10 calibration must use mean-reward selection")
    if len(ARMS) != 10:
        raise ValueError("v14.10 requires exactly 10 continuation arms")
    matched = ARMS[MATCHED_COMPARATOR_ARM]
    if (
        matched["checkpoint_score_mode"]
        != "paired_relative_frequency_feasibility_first"
        or float(matched["upper_deployment_frequency_dual_lr"]) != 0.0
        or float(matched["lower_deployment_frequency_dual_lr"]) != 0.0
    ):
        raise ValueError("v14.10 matched comparator contract drifted")
    for arm in LEARNED_ARMS:
        arm_spec = ARMS[arm]
        if (
            arm_spec["checkpoint_score_mode"]
            != matched["checkpoint_score_mode"]
            or arm_spec["lower_action_router_strength"]
            != matched["lower_action_router_strength"]
            or (
                float(arm_spec["upper_deployment_frequency_dual_lr"]) <= 0.0
                and float(arm_spec["lower_deployment_frequency_dual_lr"]) <= 0.0
            )
        ):
            raise ValueError(f"v14.10 learned arm is not matched: {arm}")
        for level in ("upper", "lower"):
            dual = float(arm_spec[f"{level}_deployment_frequency_dual_lr"])
            reduction = float(
                arm_spec[
                    f"{level}_deployment_frequency_reference_reduction_fraction"
                ]
            )
            if (dual > 0.0) != (reduction > 0.0):
                raise ValueError(
                    f"v14.10 {level} deployment target is not active-matched: {arm}"
                )


validate_frozen_design()
