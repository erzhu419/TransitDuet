"""Frozen MuJoCo v14.16 crossed-restoration mechanism screen."""

from __future__ import annotations

import math

from scripts import (
    mujoco_v14_15_closed_loop_restoration_filter_screen_spec as predecessor,
)


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v14_16_crossed_restoration_mechanism_screen_v2"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_16_crossed_pathwise_restoration"
)
FROZEN_ALGORITHM_REVISION = (
    "c9823e22b91d0260ec37ab20457d4c8c411aafd5"
)
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "df99be84451c01feeb05a68206ba91819db3ea488df13d471f6cb013574d2ea6"
)
REQUIRE_EXPLICIT_PROTOCOL_SELECTION = True
PREREGISTRATION_STATUS = (
    "frozen_before_v14_16_v2_crossed_restoration_outcome_access"
)

ENVIRONMENTS = predecessor.ENVIRONMENTS
TRAINING_DISTURBANCE_MODES = predecessor.TRAINING_DISTURBANCE_MODES
EVALUATION_DISTURBANCE_MODES = predecessor.EVALUATION_DISTURBANCE_MODES


def _upgrade(specification: dict[str, object]) -> dict[str, object]:
    return {
        **specification,
        "deployment_frequency_projection_objective": "worst_group",
        "deployment_frequency_restoration_freeze_reward_actor": False,
        "deployment_frequency_pathwise_robust": False,
        "deployment_frequency_anchor_state_replay_seed_roots": (),
    }


ANCHOR_SPEC = _upgrade(dict(predecessor.ANCHOR_SPEC))
ANCHOR_SPEC["arm_role"] = "shared_v14_16_anchor"

BASE_CONTROL_ARM = predecessor.BASE_CONTROL_ARM
CALIBRATION_ARM = predecessor.CALIBRATION_ARM
MATCHED_COMPARATOR_ARM = predecessor.MATCHED_COMPARATOR_ARM
COMPARATOR_ARM = MATCHED_COMPARATOR_ARM

WORST_MODE_TRAIN_REPLAY_ARM = "worst_mode_trainreplay"
L2_MODE_TRAIN_REPLAY_ARM = "l2_mode_trainreplay"
L2_PATH_TRAIN_REPLAY_ARM = "l2_path_trainreplay"
L2_PATH_FREEZE_TRAIN_REPLAY_ARM = "l2_path_freeze_trainreplay"
L2_PATH_FREEZE_CROSSED_REPLAY_ARM = "l2_path_freeze_crossreplay"

RETIRED_ENGINEERING_SEEDS = (
    66030672, 1747289615, 2261102359,
    2427260739, 1976957629, 3818387904, 2873153857,
    2411939979, 3140363664,
    3129729082, 3425257929, 1586248794, 3437281405,
    3619375873, 3879066966, 2778113525, 96038920,
    2509648319, 1086070922, 2035058809, 114106095,
    1501727607, 713873139, 1384337882, 118404040,
    3095107190, 285836482, 3578577806, 2097452820,
    2450247925, 910994775, 1975100367, 4017253568,
)

OPTIMIZER_SEEDS = (3493980176, 4020259488, 583213049)
PRETRAIN_SEEDS = (4195437019, 1509618766, 3695692096, 2694076119)
PRETRAIN_SELECTION_SEEDS = (1816480686, 2948470013)
CONTINUATION_TRAIN_SEEDS = (
    1318590799,
    378906977,
    874061244,
    2920971131,
)
CONTINUATION_SELECTION_SEEDS = (
    754438151,
    4033819361,
    3680814507,
    2350542369,
)
DEPLOYMENT_FREQUENCY_ANCHOR_STATE_REPLAY_SEEDS = (
    1504882304,
    1528351262,
    3968189266,
    1765074794,
)
DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS = (
    3171922821,
    3848210681,
    3133178319,
    939796718,
)
DEVELOPMENT_EVALUATION_SEEDS = (
    324724199,
    98480853,
    3308987476,
    4218259887,
    4010771860,
    1767642859,
    2994879384,
    1045477827,
)


def _learned_arm(
    *,
    objective: str,
    pathwise: bool,
    freeze_reward_actor: bool,
    crossed_replay: bool,
) -> dict[str, object]:
    source = dict(predecessor.ARMS[
        "group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3"
    ])
    return {
        **_upgrade(source),
        "arm_role": "learned",
        "deployment_frequency_projection_objective": str(objective),
        "deployment_frequency_pathwise_robust": bool(pathwise),
        "deployment_frequency_restoration_freeze_reward_actor": bool(
            freeze_reward_actor
        ),
        "deployment_frequency_anchor_state_replay_seed_roots": (
            DEPLOYMENT_FREQUENCY_ANCHOR_STATE_REPLAY_SEEDS
            if crossed_replay else ()
        ),
    }


ARMS = {
    BASE_CONTROL_ARM: _upgrade(dict(predecessor.ARMS[BASE_CONTROL_ARM])),
    CALIBRATION_ARM: _upgrade(dict(predecessor.ARMS[CALIBRATION_ARM])),
    MATCHED_COMPARATOR_ARM: _upgrade(dict(
        predecessor.ARMS[MATCHED_COMPARATOR_ARM]
    )),
    WORST_MODE_TRAIN_REPLAY_ARM: _learned_arm(
        objective="worst_group",
        pathwise=False,
        freeze_reward_actor=False,
        crossed_replay=False,
    ),
    L2_MODE_TRAIN_REPLAY_ARM: _learned_arm(
        objective="violation_l2",
        pathwise=False,
        freeze_reward_actor=False,
        crossed_replay=False,
    ),
    L2_PATH_TRAIN_REPLAY_ARM: _learned_arm(
        objective="violation_l2",
        pathwise=True,
        freeze_reward_actor=False,
        crossed_replay=False,
    ),
    L2_PATH_FREEZE_TRAIN_REPLAY_ARM: _learned_arm(
        objective="violation_l2",
        pathwise=True,
        freeze_reward_actor=True,
        crossed_replay=False,
    ),
    L2_PATH_FREEZE_CROSSED_REPLAY_ARM: _learned_arm(
        objective="violation_l2",
        pathwise=True,
        freeze_reward_actor=True,
        crossed_replay=True,
    ),
}

LEARNED_ARMS = (
    WORST_MODE_TRAIN_REPLAY_ARM,
    L2_MODE_TRAIN_REPLAY_ARM,
    L2_PATH_TRAIN_REPLAY_ARM,
    L2_PATH_FREEZE_TRAIN_REPLAY_ARM,
    L2_PATH_FREEZE_CROSSED_REPLAY_ARM,
)
EVALUATED_ARMS = (CALIBRATION_ARM, *LEARNED_ARMS)
GROUPWISE_ARMS = LEARNED_ARMS
REPLAY_ARMS = LEARNED_ARMS
TRUST_ARMS = LEARNED_ARMS
JOINT_ARMS = LEARNED_ARMS
CLOSED_LOOP_ARMS = LEARNED_ARMS
RESTORATION_ARMS = LEARNED_ARMS
CANDIDATE_ARMS = LEARNED_ARMS
AUTHORIZING_ARMS = LEARNED_ARMS
STRICT_CLOSED_LOOP_CONTROL_ARM = WORST_MODE_TRAIN_REPLAY_ARM

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
EXPECTED_CLOSED_LOOP_GUARD_CONSTRAINT_COUNT = (
    6 * len(TRAINING_DISTURBANCE_MODES)
)
EXPECTED_CLOSED_LOOP_GUARD_CONTRACT = (
    "paired_frozen_anchor_actual_closed_loop_reward_floor_and_five_"
    "frequency_endpoints_with_restoration_merit_v2"
)
MINIMUM_REPLAY_GROUP_COUNT = 2 * len(TRAINING_DISTURBANCE_MODES)
MINIMUM_GROUPWISE_GROUP_COUNT = len(TRAINING_DISTURBANCE_MODES)
MINIMUM_CLOSED_LOOP_GUARD_EVALUATIONS = CONTINUATION_ITERATIONS + 2
RESTORATION_MIN_REDUCTION = predecessor.RESTORATION_MIN_REDUCTION
RESTORATION_FUNNEL_MULTIPLIERS = (3.0,)

DEVELOPMENT_DISCLOSURE = (
    "This v14.16 development screen was designed after the complete v14.15 "
    "multiseed screen rejected its simultaneous primary family. The causal "
    "diagnosis was frozen before v14.16 outcomes: v14.15 optimized only the "
    "worst differentiable group, reused iteration-zero training paths for "
    "constraint-state replay, averaged independent guard roots within each "
    "disturbance mode, and continued reward-PPO actor updates while restoring "
    "frequency feasibility. Five cumulative arms isolate normalized all-active "
    "L2 projection, individual-path guard/selection constraints, restoration-"
    "phase reward-actor freezing, and crossed independent frozen-state replay. "
    "The first v14.16 scheduler attempt was invalidated before analysis when "
    "training-path replay arms exposed an input-validation regression; all "
    "active tasks were cancelled and its seed namespace was retired. All v2 "
    "arms share one explicit v14.16 checkpoint protocol, source-bound "
    "anchors, budgets, disturbance registry, optimization budget, and held-out "
    "paths. Optimizer seed is the replication unit. This screen is adaptive "
    "development evidence and cannot support a confirmation claim."
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
    return 6 * (
        EXPECTED_CLOSED_LOOP_GUARD_PATH_COUNT
        if ARMS[str(arm)].get(
            "deployment_frequency_pathwise_robust", False
        ) else len(TRAINING_DISTURBANCE_MODES)
    )


def expected_closed_loop_guard_contract(arm: str) -> str:
    if ARMS[str(arm)].get("deployment_frequency_pathwise_robust", False):
        return (
            "paired_frozen_anchor_actual_closed_loop_pathwise_reward_floor_"
            "and_five_frequency_endpoints_with_restoration_merit_v3"
        )
    return EXPECTED_CLOSED_LOOP_GUARD_CONTRACT


def expected_router_training_strengths(arm: str) -> list[float]:
    target = float(ARMS[str(arm)]["lower_action_router_strength"])
    schedule = str(ARMS[str(arm)][
        "lower_action_router_training_schedule"
    ])
    warmup = float(ARMS[str(arm)][
        "lower_action_router_warmup_fraction"
    ])
    ramp = float(ARMS[str(arm)][
        "lower_action_router_ramp_fraction"
    ])
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
        restoration_freeze_reward_actor=bool(arm_spec[
            "deployment_frequency_restoration_freeze_reward_actor"
        ]),
        pathwise_robust=bool(arm_spec[
            "deployment_frequency_pathwise_robust"
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
        raise ValueError("v14.16 seed roles overlap")
    predecessor_roles = (
        predecessor.OPTIMIZER_SEEDS,
        predecessor.PRETRAIN_SEEDS,
        predecessor.PRETRAIN_SELECTION_SEEDS,
        predecessor.CONTINUATION_TRAIN_SEEDS,
        predecessor.CONTINUATION_SELECTION_SEEDS,
        predecessor.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS,
        predecessor.DEVELOPMENT_EVALUATION_SEEDS,
    )
    predecessor_seeds = {
        int(seed) for role in predecessor_roles for seed in role
    }
    if predecessor_seeds & set(flattened):
        raise ValueError("v14.16 reuses a v14.15 seed")
    if set(RETIRED_ENGINEERING_SEEDS) & set(flattened):
        raise ValueError("v14.16 v2 reuses a retired engineering seed")
    if len(OPTIMIZER_SEEDS) != 3 or len(ARMS) != 8:
        raise ValueError("v14.16 mechanism matrix is incomplete")
    expected_chain = (
        ("worst_group", False, False, False),
        ("violation_l2", False, False, False),
        ("violation_l2", True, False, False),
        ("violation_l2", True, True, False),
        ("violation_l2", True, True, True),
    )
    observed_chain = tuple(
        (
            str(ARMS[arm]["deployment_frequency_projection_objective"]),
            bool(ARMS[arm]["deployment_frequency_pathwise_robust"]),
            bool(ARMS[arm][
                "deployment_frequency_restoration_freeze_reward_actor"
            ]),
            bool(ARMS[arm][
                "deployment_frequency_anchor_state_replay_seed_roots"
            ]),
        )
        for arm in LEARNED_ARMS
    )
    if observed_chain != expected_chain:
        raise ValueError("v14.16 cumulative ablation chain drifted")
    if any(
        not bool(ARMS[arm]["deployment_frequency_anchor_state_replay"])
        or not bool(ARMS[arm]["deployment_frequency_ppo_trust_region"])
        or not bool(ARMS[arm][
            "deployment_frequency_closed_loop_restoration_filter"
        ])
        for arm in LEARNED_ARMS
    ):
        raise ValueError("v14.16 learned restoration core drifted")
    if len(FROZEN_ALGORITHM_REVISION) != 40:
        raise ValueError("v14.16 algorithm revision is not frozen")
    if len(FROZEN_SOURCE_MANIFEST_SHA256) != 64:
        raise ValueError("v14.16 source manifest is not frozen")


def __getattr__(name: str):
    return getattr(predecessor, name)


validate_frozen_design()
