"""Frozen design for the MuJoCo v14.14 closed-loop actor-guard screen."""

from __future__ import annotations

import math


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v14_14_closed_loop_actor_guard_screen_v1"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_14_closed_loop_actor_guard"
)
FROZEN_ALGORITHM_REVISION = "94ad4a1d16d763c36a80c2b58f9c841ede3fb4dd"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "0649cfbdd481df8b004efa9e03642ec81f232501f24093ae28c7e3c1372c61f7"
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
        "upper_deployment_frequency_max_projection_steps": 1,
        "lower_deployment_frequency_max_projection_steps": 1,
        "upper_deployment_frequency_reward_tolerance": 1e-8,
        "lower_deployment_frequency_reward_tolerance": 1e-8,
        "upper_deployment_frequency_target_tolerance": 0.0,
        "lower_deployment_frequency_target_tolerance": 0.0,
        "deployment_frequency_groupwise_robust": False,
        "deployment_frequency_anchor_state_replay": False,
        "deployment_frequency_ppo_trust_region": False,
        "deployment_frequency_ppo_trust_region_backtracks": 8,
        "deployment_frequency_closed_loop_trust_region": False,
        "deployment_frequency_closed_loop_trust_region_backtracks": 8,
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
    max_projection_steps: int = 1,
    reward_tolerance: float = 1e-8,
    target_tolerance: float = 0.0,
    groupwise_robust: bool = False,
    anchor_state_replay: bool = False,
    ppo_trust_region: bool = False,
    trust_region_backtracks: int = 8,
    closed_loop_trust_region: bool = False,
    closed_loop_trust_region_backtracks: int = 8,
    actor_anchor_coef: float = 0.0,
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
        "upper_deployment_frequency_max_projection_steps": int(
            max_projection_steps
        ),
        "lower_deployment_frequency_max_projection_steps": int(
            max_projection_steps
        ),
        "upper_deployment_frequency_reward_tolerance": float(
            reward_tolerance
        ),
        "lower_deployment_frequency_reward_tolerance": float(
            reward_tolerance
        ),
        "upper_deployment_frequency_target_tolerance": float(
            target_tolerance
        ),
        "lower_deployment_frequency_target_tolerance": float(
            target_tolerance
        ),
        "deployment_frequency_groupwise_robust": bool(groupwise_robust),
        "deployment_frequency_anchor_state_replay": bool(
            anchor_state_replay
        ),
        "deployment_frequency_ppo_trust_region": bool(ppo_trust_region),
        "deployment_frequency_ppo_trust_region_backtracks": int(
            trust_region_backtracks
        ),
        "deployment_frequency_closed_loop_trust_region": bool(
            closed_loop_trust_region
        ),
        "deployment_frequency_closed_loop_trust_region_backtracks": int(
            closed_loop_trust_region_backtracks
        ),
        "upper_actor_anchor_coef": float(actor_anchor_coef),
        "lower_actor_anchor_coef": float(actor_anchor_coef),
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
    **{
        arm: _continuation_arm(
            role="learned",
            router_strength=0.50,
            upper_deployment_dual_lr=0.03,
            lower_deployment_dual_lr=0.08,
            upper_step_scale=3.0,
            lower_step_scale=10.0,
            upper_reduction_fraction=0.05,
            lower_reduction_fraction=0.05,
            checkpoint_score_mode=(
                "paired_relative_frequency_feasibility_first"
            ),
            checkpoint_constraint_penalty=10.0,
            max_projection_steps=projection_steps,
            reward_tolerance=reward_tolerance,
            groupwise_robust=True,
            anchor_state_replay=anchor_replay,
            ppo_trust_region=ppo_trust,
            closed_loop_trust_region=closed_loop_trust,
            closed_loop_trust_region_backtracks=closed_loop_backtracks,
        )
        for (
            arm,
            anchor_replay,
            ppo_trust,
            closed_loop_trust,
            reward_tolerance,
            projection_steps,
            closed_loop_backtracks,
        ) in (
            (
                "group_replay1_trust1_outer0_eps1e3_k8_control",
                True, True, False, 1e-3, 8, 8,
            ),
            (
                "group_replay1_trust1_outer0_eps5e3_k8_control",
                True, True, False, 5e-3, 8, 8,
            ),
            (
                "group_replay0_trust0_outer1_eps1e3_bt8",
                False, False, True, 1e-3, 8, 8,
            ),
            (
                "group_replay1_trust1_outer1_eps1e3_bt4",
                True, True, True, 1e-3, 8, 4,
            ),
            (
                "group_replay1_trust1_outer1_eps1e3_bt8",
                True, True, True, 1e-3, 8, 8,
            ),
            (
                "group_replay1_trust1_outer1_eps5e3_bt8",
                True, True, True, 5e-3, 8, 8,
            ),
        )
    },
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
GROUPWISE_ARMS = tuple(
    arm for arm in LEARNED_ARMS
    if bool(ARMS[arm]["deployment_frequency_groupwise_robust"])
)
REPLAY_ARMS = tuple(
    arm for arm in LEARNED_ARMS
    if bool(ARMS[arm]["deployment_frequency_anchor_state_replay"])
)
TRUST_ARMS = tuple(
    arm for arm in LEARNED_ARMS
    if bool(ARMS[arm]["deployment_frequency_ppo_trust_region"])
)
JOINT_ARMS = tuple(
    arm for arm in LEARNED_ARMS if arm in REPLAY_ARMS and arm in TRUST_ARMS
)
CLOSED_LOOP_ARMS = tuple(
    arm for arm in LEARNED_ARMS
    if bool(ARMS[arm]["deployment_frequency_closed_loop_trust_region"])
)
AUTHORIZING_ARMS = tuple(
    arm for arm in JOINT_ARMS if arm in CLOSED_LOOP_ARMS
)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    1195064860, 3244804604, 4287559378, 1066423950,
    490062525, 2777261578, 428645894, 782755504,
    2426251303, 3265362540, 4168590025, 568542280,
    3275807154, 3466334645, 634847213, 993691266,
)
PRETRAIN_SEEDS = (452064954, 1267877354, 3819999315, 1494131082)
PRETRAIN_SELECTION_SEEDS = (124180470, 2636535476)
CONTINUATION_TRAIN_SEEDS = (2041430480, 2154572447, 2498287455, 3922141145)
CONTINUATION_SELECTION_SEEDS = (3900430227, 4166056516)
DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS = (3750550362, 97662145)
DEVELOPMENT_EVALUATION_SEEDS = (
    1437756380, 3592951798, 1546206487, 2219765512,
    357963746, 1719577914, 3682091719, 1165220600,
)

STEPS = 512
EPISODE_HORIZON = 1000
PRETRAIN_ITERATIONS = 64
CONTINUATION_ITERATIONS = 24
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
MINIMUM_GROUPWISE_GROUP_COUNT = len(TRAINING_DISTURBANCE_MODES)
MINIMUM_REPLAY_GROUP_COUNT = 2 * len(TRAINING_DISTURBANCE_MODES)
EXPECTED_REWARD_GUARD_GROUP_COUNT = len(TRAINING_DISTURBANCE_MODES)
EXPECTED_ANCHOR_REPLAY_PATH_COUNT = len(TRAINING_DISTURBANCE_MODES)
EXPECTED_CLOSED_LOOP_GUARD_PATH_COUNT = (
    len(DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS)
    * len(TRAINING_DISTURBANCE_MODES)
)
EXPECTED_CLOSED_LOOP_GUARD_CONSTRAINT_COUNT = (
    6 * len(TRAINING_DISTURBANCE_MODES)
)
MINIMUM_CLOSED_LOOP_GUARD_EVALUATIONS = CONTINUATION_ITERATIONS + 2
MINIMUM_CLOSED_LOOP_EFFECTIVE_UPDATES = 1
MAXIMUM_CLOSED_LOOP_REWARD_VIOLATIONS = 0
MAXIMUM_CLOSED_LOOP_FREQUENCY_VIOLATIONS = 0
MINIMUM_TRUST_ACCEPTED_STEPS = 1
MINIMUM_GROUPWISE_ACCEPTED_STEPS = 1
MINIMUM_GROUPWISE_TARGET_REACHED_GROUPS = 1
MAXIMUM_PROJECTION_REWARD_BUDGET_VIOLATIONS = 0
MAXIMUM_GROUP_REWARD_BUDGET_VIOLATIONS = 0
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
    "This v14.14 screen was designed only after the frozen v14.13 preflight "
    "finished with no eligible learned checkpoint. v14.13 showed that anchor "
    "state replay, PPO-level trust regions, and iterative projection could meet "
    "same-batch constraints but did not transfer to unseen deterministic "
    "closed-loop trajectories. v14.14 therefore evaluates each joint upper and "
    "lower actor update on two independent roots crossed with all four training "
    "disturbances. It backtracks or rejects the actor transaction whenever the "
    "actual 2% reward floor is violated or the frozen-anchor five-endpoint "
    "frequency rank worsens. Critics and duals retain their on-policy update. "
    "Guard roots are training constraints and are disjoint from training, "
    "checkpoint-selection, and held-out roots. Outer-only and matched v14.13 "
    "inner-mechanism controls isolate causality. Only joint outer-guard arms with "
    "a nonzero accepted actor update, a trained selected checkpoint, and zero "
    "selection and held-out paired violations may authorize expansion. This is "
    "development evidence only."
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


def deployment_constraint_contract(arm: str) -> str:
    arm_spec = ARMS[str(arm)]
    requested = bool(
        float(arm_spec["upper_deployment_frequency_dual_lr"]) > 0.0
        or float(arm_spec["lower_deployment_frequency_dual_lr"]) > 0.0
    )
    if not requested:
        return "disabled"
    base = (
        "episode_reset_groupwise_worst_differentiable_actor_mean_tanh_upper_"
        "hold_hpf8_lower_lpf32_per_group_anchor_relative_target_with_"
        "absolute_floor"
    )
    replay = bool(
        arm_spec["deployment_frequency_anchor_state_replay"]
    )
    trust = bool(arm_spec["deployment_frequency_ppo_trust_region"])
    closed_loop = bool(
        arm_spec["deployment_frequency_closed_loop_trust_region"]
    )
    if closed_loop:
        inner_mechanisms = ""
        if replay:
            inner_mechanisms += "frozen_anchor_state_replay_"
        if trust:
            inner_mechanisms += "ppo_trust_region_"
        return (
            base
            + "_"
            + inner_mechanisms
            + "independent_crossed_closed_loop_reward_floor_and_five_"
            "frequency_endpoint_joint_actor_backtracking_v7"
        )
    if replay and trust:
        return (
            "episode_reset_candidate_and_frozen_anchor_state_replay_"
            "groupwise_worst_differentiable_actor_mean_tanh_upper_hold_hpf8_"
            "lower_lpf32_per_group_anchor_relative_target_with_absolute_"
            "floor_ppo_trust_region_and_iterative_per_group_cumulative_"
            "reward_budget_projection_v6"
        )
    if replay:
        return (
            base
            + "_frozen_anchor_state_replay_iterative_per_group_cumulative_"
            "reward_budget_projection_v6"
        )
    if trust:
        return (
            base
            + "_ppo_trust_region_and_iterative_per_group_cumulative_reward_"
            "budget_projection_v6"
        )
    return (
        base
        + "_iterative_per_group_cumulative_reward_budget_projection_v5"
    )


def validate_frozen_design() -> None:
    roles = (
        OPTIMIZER_SEEDS,
        PRETRAIN_SEEDS,
        PRETRAIN_SELECTION_SEEDS,
        CONTINUATION_TRAIN_SEEDS,
        CONTINUATION_SELECTION_SEEDS,
        DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS,
        DEVELOPMENT_EVALUATION_SEEDS,
    )
    flattened = [int(seed) for role in roles for seed in role]
    if len(flattened) != len(set(flattened)):
        raise ValueError("v14.14 seed roles overlap")
    if len(OPTIMIZER_SEEDS) != 16:
        raise ValueError("v14.14 requires 16 optimizer replicates")
    if ARMS[CALIBRATION_ARM]["checkpoint_score_mode"] != "mean_reward":
        raise ValueError("v14.14 calibration must use mean-reward selection")
    if len(ARMS) != 9:
        raise ValueError("v14.14 requires exactly 9 continuation arms")
    matched = ARMS[MATCHED_COMPARATOR_ARM]
    if (
        matched["checkpoint_score_mode"]
        != "paired_relative_frequency_feasibility_first"
        or float(matched["upper_deployment_frequency_dual_lr"]) != 0.0
        or float(matched["lower_deployment_frequency_dual_lr"]) != 0.0
    ):
        raise ValueError("v14.14 matched comparator contract drifted")
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
            raise ValueError(f"v14.14 learned arm is not matched: {arm}")
        for level in ("upper", "lower"):
            dual = float(arm_spec[f"{level}_deployment_frequency_dual_lr"])
            reduction = float(
                arm_spec[
                    f"{level}_deployment_frequency_reference_reduction_fraction"
                ]
            )
            if (dual > 0.0) != (reduction > 0.0):
                raise ValueError(
                    f"v14.14 {level} deployment target is not active-matched: {arm}"
                )
        projection_steps = int(
            arm_spec["upper_deployment_frequency_max_projection_steps"]
        )
        if projection_steps != int(
            arm_spec["lower_deployment_frequency_max_projection_steps"]
        ):
            raise ValueError(f"v14.14 projection-step levels differ: {arm}")
        tolerances = {
            float(arm_spec[
                f"{level}_deployment_frequency_reward_tolerance"
            ])
            for level in ("upper", "lower")
        }
        if len(tolerances) != 1 or tolerances.pop() not in {
            1e-8, 1e-3, 5e-3, 1e-2, 2e-2,
        }:
            raise ValueError(f"v14.14 reward budget drifted: {arm}")
        if bool(arm_spec[
            "deployment_frequency_anchor_state_replay"
        ]) != (arm in REPLAY_ARMS):
            raise ValueError(f"v14.14 replay arm contract drifted: {arm}")
        if bool(arm_spec[
            "deployment_frequency_ppo_trust_region"
        ]) != (arm in TRUST_ARMS):
            raise ValueError(f"v14.14 trust arm contract drifted: {arm}")
        if bool(arm_spec[
            "deployment_frequency_closed_loop_trust_region"
        ]) != (arm in CLOSED_LOOP_ARMS):
            raise ValueError(f"v14.14 closed-loop arm contract drifted: {arm}")
    if not GROUPWISE_ARMS or any(
        not bool(ARMS[arm]["deployment_frequency_groupwise_robust"])
        for arm in GROUPWISE_ARMS
    ):
        raise ValueError("v14.14 groupwise arm contract drifted")
    if (
        len(JOINT_ARMS) != 5
        or len(CLOSED_LOOP_ARMS) != 4
        or len(AUTHORIZING_ARMS) != 3
    ):
        raise ValueError("v14.14 joint authorization contract drifted")
    if any(
        float(ARMS[arm][
            "upper_deployment_frequency_reward_tolerance"
        ]) > 1e-2
        for arm in AUTHORIZING_ARMS
    ):
        raise ValueError("v14.14 authorizing reward budget is too large")


validate_frozen_design()
