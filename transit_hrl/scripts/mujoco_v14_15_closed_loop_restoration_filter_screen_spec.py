"""Frozen design for the MuJoCo v14.15 closed-loop restoration-filter screen."""

from __future__ import annotations

import math


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v14_15_closed_loop_restoration_filter_screen_v1"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_15_closed_loop_restoration_filter"
)
FROZEN_ALGORITHM_REVISION = "8cc7ecd537167f05c04d9df9792cb5de88c5ce52"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "f52ffdacf29a0e90567f86ab7cfa4221aa422b890808c0ae1289707e06f207d5"
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
        "deployment_frequency_closed_loop_restoration_filter": False,
        "deployment_frequency_closed_loop_restoration_min_reduction": 1e-4,
        "deployment_frequency_closed_loop_restoration_funnel_multiplier": 3.0,
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
    closed_loop_restoration_filter: bool = False,
    closed_loop_restoration_min_reduction: float = 1e-4,
    closed_loop_restoration_funnel_multiplier: float = 3.0,
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
        "deployment_frequency_closed_loop_restoration_filter": bool(
            closed_loop_restoration_filter
        ),
        "deployment_frequency_closed_loop_restoration_min_reduction": float(
            closed_loop_restoration_min_reduction
        ),
        "deployment_frequency_closed_loop_restoration_funnel_multiplier": float(
            closed_loop_restoration_funnel_multiplier
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
            max_projection_steps=8,
            reward_tolerance=reward_tolerance,
            groupwise_robust=True,
            anchor_state_replay=True,
            ppo_trust_region=True,
            closed_loop_trust_region=True,
            closed_loop_trust_region_backtracks=closed_loop_backtracks,
            closed_loop_restoration_filter=restoration_filter,
            closed_loop_restoration_min_reduction=1e-4,
            closed_loop_restoration_funnel_multiplier=funnel_multiplier,
        )
        for (
            arm,
            reward_tolerance,
            closed_loop_backtracks,
            restoration_filter,
            funnel_multiplier,
        ) in (
            (
                "group_replay1_trust1_outer1_eps1e3_bt4_strict_control",
                1e-3, 4, False, 3.0,
            ),
            (
                "group_replay1_trust1_outer1_restore1_eps1e3_bt4_f2",
                1e-3, 4, True, 2.0,
            ),
            (
                "group_replay1_trust1_outer1_restore1_eps1e3_bt4_f3",
                1e-3, 4, True, 3.0,
            ),
            (
                "group_replay1_trust1_outer1_restore1_eps5e3_bt4_f2",
                5e-3, 4, True, 2.0,
            ),
            (
                "group_replay1_trust1_outer1_restore1_eps5e3_bt4_f3",
                5e-3, 4, True, 3.0,
            ),
            (
                "group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3",
                5e-3, 8, True, 3.0,
            ),
        )
    },
}
BASE_CONTROL_ARM = "mean_s000_control"
CALIBRATION_ARM = "mean_s050_projection_calibration"
MATCHED_COMPARATOR_ARM = "paired_s050_d000_control"
COMPARATOR_ARM = MATCHED_COMPARATOR_ARM
STRICT_CLOSED_LOOP_CONTROL_ARM = (
    "group_replay1_trust1_outer1_eps1e3_bt4_strict_control"
)
LEARNED_ARMS = tuple(
    arm for arm, arm_spec in ARMS.items()
    if arm_spec["arm_role"] == "learned"
)
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
RESTORATION_ARMS = tuple(
    arm for arm in LEARNED_ARMS
    if bool(ARMS[arm][
        "deployment_frequency_closed_loop_restoration_filter"
    ])
)
CANDIDATE_ARMS = RESTORATION_ARMS
AUTHORIZING_ARMS = tuple(
    arm for arm in RESTORATION_ARMS
    if arm in JOINT_ARMS and arm in CLOSED_LOOP_ARMS
)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    1361331598, 2182280623, 4016970853, 2829355830,
    2069740444, 1282433932, 975280645, 2798338935,
    23627913, 3508173532, 3799486943, 4004573542,
    1167085633, 1775539633, 3810522536, 1442147348,
)
PRETRAIN_SEEDS = (41449075, 1224782572, 1077763750, 4036728464)
PRETRAIN_SELECTION_SEEDS = (3855120584, 2989066164)
CONTINUATION_TRAIN_SEEDS = (1337173826, 2280134531, 2693568730, 2287416316)
CONTINUATION_SELECTION_SEEDS = (2620430295, 1321833778)
DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS = (3509242572, 3118340037)
DEVELOPMENT_EVALUATION_SEEDS = (
    4285151767, 828212288, 1026664497, 3463847558,
    3812827077, 357535940, 1763058948, 1203729697,
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
RESTORATION_MIN_REDUCTION = 1e-4
RESTORATION_FUNNEL_MULTIPLIERS = (2.0, 3.0)
EXPECTED_CLOSED_LOOP_GUARD_CONTRACT = (
    "paired_frozen_anchor_actual_closed_loop_reward_floor_and_five_"
    "frequency_endpoints_with_restoration_merit_v2"
)
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
    "This v14.15 screen was designed only after the frozen v14.14 preflight "
    "finished with no accepted closed-loop actor update. The v14.14 full steps "
    "often preserved the reward floor and reduced continuous frequency "
    "violation mass, but its lexicographic discrete-count gate rejected every "
    "backtrack whenever the worst remaining endpoint increased. v14.15 keeps "
    "the same independent closed-loop guard roots, reward floor, five frequency "
    "endpoints, anchor replay, PPO trust region, and training budget. Before "
    "feasibility, it accepts only reward-safe transactions that reduce the "
    "continuous squared violation merit by the pre-registered fraction-scaled "
    "minimum while staying inside a fixed initial-worst funnel. After all "
    "frequency endpoints become feasible, it switches to strict maintenance "
    "and forbids any violation. The discrete violation count controls only this "
    "phase switch. Critics and duals retain their on-policy update. Guard roots "
    "remain disjoint from training, checkpoint-selection, and held-out roots. "
    "The strict v14.14 guard is retained as a causal control. Only restoration "
    "arms with a nonzero accepted actor update, a trained selected checkpoint, "
    "and zero selection and held-out paired violations may authorize expansion. "
    "This is single-optimizer-seed development evidence only."
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
        closed_loop_contract = (
            base
            + "_"
            + inner_mechanisms
            + "independent_crossed_closed_loop_reward_floor_and_five_"
            "frequency_endpoint_joint_actor_backtracking_v7"
        )
        if bool(arm_spec[
            "deployment_frequency_closed_loop_restoration_filter"
        ]):
            return (
                closed_loop_contract
                + "_two_phase_continuous_violation_merit_restoration_and_"
                "strict_feasibility_maintenance_v8"
            )
        return closed_loop_contract
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
        raise ValueError("v14.15 seed roles overlap")
    if len(OPTIMIZER_SEEDS) != 16:
        raise ValueError("v14.15 requires 16 optimizer replicates")
    if ARMS[CALIBRATION_ARM]["checkpoint_score_mode"] != "mean_reward":
        raise ValueError("v14.15 calibration must use mean-reward selection")
    if len(ARMS) != 9:
        raise ValueError("v14.15 requires exactly 9 continuation arms")
    matched = ARMS[MATCHED_COMPARATOR_ARM]
    if (
        matched["checkpoint_score_mode"]
        != "paired_relative_frequency_feasibility_first"
        or float(matched["upper_deployment_frequency_dual_lr"]) != 0.0
        or float(matched["lower_deployment_frequency_dual_lr"]) != 0.0
    ):
        raise ValueError("v14.15 matched comparator contract drifted")
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
            raise ValueError(f"v14.15 learned arm is not matched: {arm}")
        for level in ("upper", "lower"):
            dual = float(arm_spec[f"{level}_deployment_frequency_dual_lr"])
            reduction = float(
                arm_spec[
                    f"{level}_deployment_frequency_reference_reduction_fraction"
                ]
            )
            if (dual > 0.0) != (reduction > 0.0):
                raise ValueError(
                    f"v14.15 {level} deployment target is not active-matched: {arm}"
                )
        projection_steps = int(
            arm_spec["upper_deployment_frequency_max_projection_steps"]
        )
        if projection_steps != int(
            arm_spec["lower_deployment_frequency_max_projection_steps"]
        ):
            raise ValueError(f"v14.15 projection-step levels differ: {arm}")
        tolerances = {
            float(arm_spec[
                f"{level}_deployment_frequency_reward_tolerance"
            ])
            for level in ("upper", "lower")
        }
        if len(tolerances) != 1 or tolerances.pop() not in {
            1e-8, 1e-3, 5e-3, 1e-2, 2e-2,
        }:
            raise ValueError(f"v14.15 reward budget drifted: {arm}")
        if bool(arm_spec[
            "deployment_frequency_anchor_state_replay"
        ]) != (arm in REPLAY_ARMS):
            raise ValueError(f"v14.15 replay arm contract drifted: {arm}")
        if bool(arm_spec[
            "deployment_frequency_ppo_trust_region"
        ]) != (arm in TRUST_ARMS):
            raise ValueError(f"v14.15 trust arm contract drifted: {arm}")
        if bool(arm_spec[
            "deployment_frequency_closed_loop_trust_region"
        ]) != (arm in CLOSED_LOOP_ARMS):
            raise ValueError(f"v14.15 closed-loop arm contract drifted: {arm}")
        restoration = bool(arm_spec[
            "deployment_frequency_closed_loop_restoration_filter"
        ])
        if restoration != (arm in RESTORATION_ARMS):
            raise ValueError(f"v14.15 restoration arm contract drifted: {arm}")
        if restoration and (
            not bool(arm_spec["deployment_frequency_anchor_state_replay"])
            or not bool(arm_spec["deployment_frequency_ppo_trust_region"])
            or not bool(arm_spec[
                "deployment_frequency_closed_loop_trust_region"
            ])
            or not math.isclose(
                float(arm_spec[
                    "deployment_frequency_closed_loop_restoration_min_reduction"
                ]),
                RESTORATION_MIN_REDUCTION,
            )
            or float(arm_spec[
                "deployment_frequency_closed_loop_restoration_funnel_multiplier"
            ]) not in RESTORATION_FUNNEL_MULTIPLIERS
        ):
            raise ValueError(
                f"v14.15 restoration filter is not frozen: {arm}"
            )
    if not GROUPWISE_ARMS or any(
        not bool(ARMS[arm]["deployment_frequency_groupwise_robust"])
        for arm in GROUPWISE_ARMS
    ):
        raise ValueError("v14.15 groupwise arm contract drifted")
    if (
        len(JOINT_ARMS) != 6
        or len(CLOSED_LOOP_ARMS) != 6
        or len(RESTORATION_ARMS) != 5
        or len(AUTHORIZING_ARMS) != 5
        or set(AUTHORIZING_ARMS) != set(RESTORATION_ARMS)
    ):
        raise ValueError("v14.15 joint authorization contract drifted")
    strict = ARMS[STRICT_CLOSED_LOOP_CONTROL_ARM]
    if (
        bool(strict["deployment_frequency_closed_loop_restoration_filter"])
        or int(strict[
            "deployment_frequency_closed_loop_trust_region_backtracks"
        ]) != 4
        or not math.isclose(
            float(strict["upper_deployment_frequency_reward_tolerance"]),
            1e-3,
        )
    ):
        raise ValueError("v14.15 strict closed-loop control drifted")
    if any(
        float(ARMS[arm][
            "upper_deployment_frequency_reward_tolerance"
        ]) > 1e-2
        for arm in AUTHORIZING_ARMS
    ):
        raise ValueError("v14.15 authorizing reward budget is too large")


validate_frozen_design()
