"""Frozen MuJoCo v14.17 native primal-dual/CVaR mechanism screen."""

from __future__ import annotations

import math

from scripts import (
    mujoco_v14_16_crossed_restoration_mechanism_screen_spec as predecessor,
)


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_17_native_pd_cvar_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_17_native_pd_cvar"
)
FROZEN_ALGORITHM_REVISION = (
    "fd6bfc316b0beafe4edc17044a02361f21175e5b"
)
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "d5fdb86a80847f2f4bc8c2a667f64dea25fa0de1621e399f69b4277dcf1af12e"
)
REQUIRE_EXPLICIT_PROTOCOL_SELECTION = True
PREREGISTRATION_STATUS = (
    "frozen_before_v14_17_native_pd_cvar_outcome_access"
)

ENVIRONMENTS = predecessor.ENVIRONMENTS
TRAINING_DISTURBANCE_MODES = predecessor.TRAINING_DISTURBANCE_MODES
EVALUATION_DISTURBANCE_MODES = predecessor.EVALUATION_DISTURBANCE_MODES

PROJECTION_CVAR_ALPHA = 0.5
CLOSED_LOOP_CVAR_ALPHA = 0.5
DUAL_SCALE_EMA_BETA = 0.95
DUAL_SCALE_FLOOR = 1e-6
NATIVE_DUAL_LR = 0.03


def _upgrade(
    specification: dict[str, object],
    *,
    risk_mode: str = "mode_mean",
) -> dict[str, object]:
    return {
        **specification,
        # All arms retain the upper cost critic so paired checkpoint hashes
        # remain architecture-matched; a zero dual LR disables its actor cost.
        "upper_constraint_mode": "primal_dual",
        "deployment_frequency_projection_cvar_alpha": PROJECTION_CVAR_ALPHA,
        "deployment_frequency_closed_loop_risk_mode": str(risk_mode),
        "deployment_frequency_closed_loop_cvar_alpha": (
            CLOSED_LOOP_CVAR_ALPHA
        ),
        "constraint_dual_normalization": "none",
        "constraint_dual_scale_ema_beta": DUAL_SCALE_EMA_BETA,
        "constraint_dual_scale_floor": DUAL_SCALE_FLOOR,
        "deployment_frequency_restoration_freeze_reward_actor": False,
        "deployment_frequency_anchor_state_replay_seed_roots": (),
    }


ANCHOR_SPEC = _upgrade(dict(predecessor.ANCHOR_SPEC))
ANCHOR_SPEC.update({
    "arm_role": "shared_v14_17_capacity_matched_anchor",
    "upper_dual_lr": 0.0,
    "lower_dual_lr": 0.0,
})

BASE_CONTROL_ARM = predecessor.BASE_CONTROL_ARM
CALIBRATION_ARM = predecessor.CALIBRATION_ARM
MATCHED_COMPARATOR_ARM = predecessor.MATCHED_COMPARATOR_ARM
V14_16_COMPARATOR_ARM = "l2_path_v1416_comparator"
NATIVE_PD_ARM = "native_pd_cvar_select"
CVAR_PROJECTION_ARM = "cvar_projection"
HYBRID_ARM = "native_pd_cvar_projection"
PRIMARY_CANDIDATE_ARM = HYBRID_ARM
COMPARATOR_ARM = MATCHED_COMPARATOR_ARM


def _capacity_matched_control(
    source: dict[str, object], *, role: str, risk_mode: str
) -> dict[str, object]:
    arm = _upgrade(dict(source), risk_mode=risk_mode)
    arm.update({
        "arm_role": str(role),
        "upper_dual_lr": 0.0,
        "lower_dual_lr": 0.0,
        "constraint_dual_normalization": "none",
    })
    return arm


def _native_pd(source: dict[str, object]) -> dict[str, object]:
    arm = _capacity_matched_control(
        source, role="native_primal_dual", risk_mode="mode_cvar"
    )
    arm.update({
        "upper_dual_lr": NATIVE_DUAL_LR,
        "lower_dual_lr": NATIVE_DUAL_LR,
        "constraint_dual_normalization": "ema_abs",
        "learned_frequency_objective": True,
    })
    return arm


def _cvar_projection(*, native_pd: bool) -> dict[str, object]:
    arm = _upgrade(
        dict(predecessor.ARMS[predecessor.WORST_MODE_TRAIN_REPLAY_ARM]),
        risk_mode="mode_cvar",
    )
    arm.update({
        "arm_role": (
            "hybrid_native_primal_dual_cvar"
            if native_pd else "cvar_projection_only"
        ),
        "deployment_frequency_projection_objective": "violation_cvar",
        "deployment_frequency_pathwise_robust": False,
        "upper_dual_lr": NATIVE_DUAL_LR if native_pd else 0.0,
        "lower_dual_lr": NATIVE_DUAL_LR if native_pd else 0.0,
        "constraint_dual_normalization": (
            "ema_abs" if native_pd else "none"
        ),
    })
    return arm


ARMS = {
    BASE_CONTROL_ARM: _capacity_matched_control(
        predecessor.ARMS[BASE_CONTROL_ARM],
        role="mean_control",
        risk_mode="mode_mean",
    ),
    CALIBRATION_ARM: _capacity_matched_control(
        predecessor.ARMS[CALIBRATION_ARM],
        role="projection_calibration",
        risk_mode="mode_mean",
    ),
    MATCHED_COMPARATOR_ARM: _capacity_matched_control(
        predecessor.ARMS[MATCHED_COMPARATOR_ARM],
        role="matched_cvar_selection_control",
        risk_mode="mode_cvar",
    ),
    V14_16_COMPARATOR_ARM: _capacity_matched_control(
        predecessor.ARMS[predecessor.L2_PATH_TRAIN_REPLAY_ARM],
        role="v14_16_best_nonfreeze_comparator",
        risk_mode="legacy",
    ),
    NATIVE_PD_ARM: _native_pd(predecessor.ARMS[MATCHED_COMPARATOR_ARM]),
    CVAR_PROJECTION_ARM: _cvar_projection(native_pd=False),
    HYBRID_ARM: _cvar_projection(native_pd=True),
}

LEARNED_ARMS = (
    V14_16_COMPARATOR_ARM,
    NATIVE_PD_ARM,
    CVAR_PROJECTION_ARM,
    HYBRID_ARM,
)
EVALUATED_ARMS = (CALIBRATION_ARM, *LEARNED_ARMS)
NATIVE_PD_ARMS = (NATIVE_PD_ARM, HYBRID_ARM)
CVAR_ARMS = (CVAR_PROJECTION_ARM, HYBRID_ARM)
GROUPWISE_ARMS = (
    V14_16_COMPARATOR_ARM,
    CVAR_PROJECTION_ARM,
    HYBRID_ARM,
)
REPLAY_ARMS = GROUPWISE_ARMS
TRUST_ARMS = GROUPWISE_ARMS
JOINT_ARMS = GROUPWISE_ARMS
CLOSED_LOOP_ARMS = GROUPWISE_ARMS
RESTORATION_ARMS = GROUPWISE_ARMS
CANDIDATE_ARMS = LEARNED_ARMS
AUTHORIZING_ARMS = LEARNED_ARMS
STRICT_CLOSED_LOOP_CONTROL_ARM = V14_16_COMPARATOR_ARM

OPTIMIZER_SEEDS = (4196455150, 3082324697, 1915709332)
PRETRAIN_SEEDS = (820919588, 2161015708, 3439212220, 3885597299)
PRETRAIN_SELECTION_SEEDS = (2463701431, 1322604469)
CONTINUATION_TRAIN_SEEDS = (246200137, 892444410, 1969253750, 3189922377)
CONTINUATION_SELECTION_SEEDS = (
    58627415,
    626487109,
    3491585429,
    3821343377,
)
DEPLOYMENT_FREQUENCY_ANCHOR_STATE_REPLAY_SEEDS = (
    1247834558,
    2258703547,
    3832946328,
    2465385296,
)
DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS = (
    2233157472,
    4009222515,
    612912378,
    3777442308,
)
DEVELOPMENT_EVALUATION_SEEDS = (
    3212130827,
    3228904836,
    1067832843,
    1080430413,
    2918603560,
    878049687,
    1285613380,
    605531257,
)

STEPS = predecessor.STEPS
EPISODE_HORIZON = predecessor.EPISODE_HORIZON
PRETRAIN_ITERATIONS = predecessor.PRETRAIN_ITERATIONS
CONTINUATION_ITERATIONS = 32
UPPER_PERIOD = predecessor.UPPER_PERIOD
HIDDEN_DIM = predecessor.HIDDEN_DIM
LEARNING_RATE = predecessor.LEARNING_RATE
LOWER_LF_RMS_BUDGET = predecessor.LOWER_LF_RMS_BUDGET
UPPER_HF_RMS_BUDGET = predecessor.UPPER_HF_RMS_BUDGET
UPPER_ACTION_SCALE = predecessor.UPPER_ACTION_SCALE
LOWER_ACTION_SCALE = predecessor.LOWER_ACTION_SCALE
LOWER_CONSTRAINT_UPDATE_MODE = predecessor.LOWER_CONSTRAINT_UPDATE_MODE
UPPER_CONSTRAINT_UPDATE_MODE = predecessor.UPPER_CONSTRAINT_UPDATE_MODE
CHECKPOINT_SELECTION_MODE = predecessor.CHECKPOINT_SELECTION_MODE
CHECKPOINT_SMOOTHING_WINDOW = predecessor.CHECKPOINT_SMOOTHING_WINDOW
CHECKPOINT_MIN_DELTA = predecessor.CHECKPOINT_MIN_DELTA
CHECKPOINT_EVALUATION_INTERVAL = predecessor.CHECKPOINT_EVALUATION_INTERVAL
PRETRAIN_CHECKPOINT_MINIMUM_ITERATION = (
    predecessor.PRETRAIN_CHECKPOINT_MINIMUM_ITERATION
)
CONTINUATION_CHECKPOINT_MINIMUM_ITERATION = -1
ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION = 7

EXPECTED_REWARD_GUARD_GROUP_COUNT = len(TRAINING_DISTURBANCE_MODES)
EXPECTED_ANCHOR_REPLAY_PATH_COUNT = len(CONTINUATION_TRAIN_SEEDS)
EXPECTED_CLOSED_LOOP_GUARD_PATH_COUNT = (
    len(DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS)
    * len(TRAINING_DISTURBANCE_MODES)
)
MINIMUM_REPLAY_GROUP_COUNT = 2 * len(TRAINING_DISTURBANCE_MODES)
MINIMUM_GROUPWISE_GROUP_COUNT = len(TRAINING_DISTURBANCE_MODES)
MINIMUM_CLOSED_LOOP_GUARD_EVALUATIONS = CONTINUATION_ITERATIONS + 2
MINIMUM_CLOSED_LOOP_EFFECTIVE_UPDATES = 1
MAXIMUM_CLOSED_LOOP_REWARD_VIOLATIONS = 0
MAXIMUM_CLOSED_LOOP_FREQUENCY_VIOLATIONS = 0
RESTORATION_MIN_REDUCTION = predecessor.RESTORATION_MIN_REDUCTION
RESTORATION_FUNNEL_MULTIPLIERS = (3.0,)

DEVELOPMENT_DISCLOSURE = (
    "This v14.17 development screen was frozen after the complete v14.16 "
    "mechanism screen rejected all-path hard feasibility and reward-actor "
    "freezing. The causal diagnosis was fixed before v14.17 outcomes: native "
    "trajectory leakage multipliers were disabled in v14.16; raw upper and "
    "lower costs had incompatible scales; and individual-path feasibility "
    "created a brittle 96-constraint checkpoint gate. v14.17 therefore tests "
    "EMA-normalized native primal-dual updates, signed upper-tail CVaR "
    "projection and selection, and their combination without reward-actor "
    "freezing. A matched CVaR-selection control and the best v14.16 nonfreeze "
    "arm separate selector, projection, and native-cost effects. All arms use "
    "one source-bound capacity-matched anchor, identical budgets, four "
    "disturbance modes, and fresh disjoint seed roles. Optimizer seed is the "
    "replication unit. This is adaptive development evidence, not a "
    "confirmation claim."
)


def expected_anchor_replay_path_count(arm: str) -> int:
    roots = tuple(ARMS[str(arm)].get(
        "deployment_frequency_anchor_state_replay_seed_roots", ()
    ))
    return (
        len(roots) * len(TRAINING_DISTURBANCE_MODES)
        if roots else len(CONTINUATION_TRAIN_SEEDS)
    )


def expected_closed_loop_guard_constraint_count(arm: str) -> int:
    arm_spec = ARMS[str(arm)]
    if not arm_spec["deployment_frequency_closed_loop_trust_region"]:
        return 0
    if arm_spec.get("deployment_frequency_pathwise_robust", False):
        return 6 * EXPECTED_CLOSED_LOOP_GUARD_PATH_COUNT
    return 6 * len(TRAINING_DISTURBANCE_MODES)


def expected_closed_loop_guard_contract(arm: str) -> str:
    arm_spec = ARMS[str(arm)]
    if arm_spec.get("deployment_frequency_pathwise_robust", False):
        return (
            "paired_frozen_anchor_actual_closed_loop_pathwise_reward_floor_"
            "and_five_frequency_endpoints_with_restoration_merit_v3"
        )
    if arm_spec["deployment_frequency_closed_loop_risk_mode"] == "mode_cvar":
        return (
            "paired_frozen_anchor_actual_closed_loop_mode_cvar_reward_floor_"
            "and_five_frequency_endpoints_with_restoration_merit_alpha_"
            f"{CLOSED_LOOP_CVAR_ALPHA:.6g}_v4"
        )
    return predecessor.EXPECTED_CLOSED_LOOP_GUARD_CONTRACT


def expected_router_training_strengths(arm: str) -> list[float]:
    target = float(ARMS[str(arm)]["lower_action_router_strength"])
    schedule = str(ARMS[str(arm)]["lower_action_router_training_schedule"])
    warmup = float(ARMS[str(arm)]["lower_action_router_warmup_fraction"])
    ramp = float(ARMS[str(arm)]["lower_action_router_ramp_fraction"])
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
    return (
        CONTINUATION_CHECKPOINT_MINIMUM_ITERATION
        if ARMS[str(arm)]["checkpoint_score_mode"]
        == "paired_relative_frequency_feasibility_first"
        else ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION
    )


def deployment_constraint_contract(arm: str) -> str:
    from freq_hrl.experiments.mujoco.control_validation import (
        deployment_frequency_constraint_contract,
    )

    arm_spec = ARMS[str(arm)]
    requested = bool(
        float(arm_spec["upper_deployment_frequency_dual_lr"]) > 0.0
        or float(arm_spec["lower_deployment_frequency_dual_lr"]) > 0.0
    )
    return deployment_frequency_constraint_contract(
        requested=requested,
        groupwise=bool(arm_spec["deployment_frequency_groupwise_robust"]),
        anchor_state_replay=bool(
            arm_spec["deployment_frequency_anchor_state_replay"]
        ),
        ppo_trust_region=bool(
            arm_spec["deployment_frequency_ppo_trust_region"]
        ),
        closed_loop_trust_region=bool(
            arm_spec["deployment_frequency_closed_loop_trust_region"]
        ),
        closed_loop_restoration_filter=bool(arm_spec[
            "deployment_frequency_closed_loop_restoration_filter"
        ]),
        projection_objective=str(arm_spec[
            "deployment_frequency_projection_objective"
        ]),
        projection_cvar_alpha=float(arm_spec[
            "deployment_frequency_projection_cvar_alpha"
        ]),
        restoration_freeze_reward_actor=bool(arm_spec[
            "deployment_frequency_restoration_freeze_reward_actor"
        ]),
        pathwise_robust=bool(arm_spec[
            "deployment_frequency_pathwise_robust"
        ]),
        closed_loop_risk_mode=str(arm_spec[
            "deployment_frequency_closed_loop_risk_mode"
        ]),
        closed_loop_cvar_alpha=float(arm_spec[
            "deployment_frequency_closed_loop_cvar_alpha"
        ]),
    )


def validate_frozen_design() -> None:
    roles = (
        OPTIMIZER_SEEDS,
        PRETRAIN_SEEDS,
        PRETRAIN_SELECTION_SEEDS,
        CONTINUATION_TRAIN_SEEDS,
        CONTINUATION_SELECTION_SEEDS,
        DEPLOYMENT_FREQUENCY_ANCHOR_STATE_REPLAY_SEEDS,
        DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS,
        DEVELOPMENT_EVALUATION_SEEDS,
    )
    flattened = [int(seed) for role in roles for seed in role]
    if len(flattened) != len(set(flattened)):
        raise ValueError("v14.17 seed roles overlap")
    predecessor_seeds = {
        int(seed)
        for name in (
            "OPTIMIZER_SEEDS",
            "PRETRAIN_SEEDS",
            "PRETRAIN_SELECTION_SEEDS",
            "CONTINUATION_TRAIN_SEEDS",
            "CONTINUATION_SELECTION_SEEDS",
            "DEPLOYMENT_FREQUENCY_ANCHOR_STATE_REPLAY_SEEDS",
            "DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS",
            "DEVELOPMENT_EVALUATION_SEEDS",
            "RETIRED_ENGINEERING_SEEDS",
        )
        for seed in getattr(predecessor, name, ())
    }
    if predecessor_seeds & set(flattened):
        raise ValueError("v14.17 reuses a v14.16/retired seed")
    if len(OPTIMIZER_SEEDS) != 3 or len(ARMS) != 7:
        raise ValueError("v14.17 mechanism matrix is incomplete")
    if any(
        ARMS[arm]["upper_constraint_mode"] != "primal_dual"
        for arm in ARMS
    ) or ANCHOR_SPEC["upper_constraint_mode"] != "primal_dual":
        raise ValueError("v14.17 upper cost-critic capacity is not matched")
    if any(
        ARMS[arm]["constraint_dual_normalization"] != "ema_abs"
        or float(ARMS[arm]["upper_dual_lr"]) != NATIVE_DUAL_LR
        or float(ARMS[arm]["lower_dual_lr"]) != NATIVE_DUAL_LR
        for arm in NATIVE_PD_ARMS
    ):
        raise ValueError("v14.17 native primal-dual contract drifted")
    if any(
        ARMS[arm]["deployment_frequency_projection_objective"]
        != "violation_cvar"
        or ARMS[arm]["deployment_frequency_closed_loop_risk_mode"]
        != "mode_cvar"
        or ARMS[arm]["deployment_frequency_pathwise_robust"]
        for arm in CVAR_ARMS
    ):
        raise ValueError("v14.17 CVaR contract drifted")
    if any(
        ARMS[arm]["deployment_frequency_restoration_freeze_reward_actor"]
        for arm in ARMS
    ):
        raise ValueError("v14.17 must not freeze the reward actor")
    if len(FROZEN_ALGORITHM_REVISION) != 40:
        raise ValueError("v14.17 algorithm revision is not frozen")
    if len(FROZEN_SOURCE_MANIFEST_SHA256) != 64:
        raise ValueError("v14.17 source manifest is not frozen")


def __getattr__(name: str):
    return getattr(predecessor, name)


validate_frozen_design()
