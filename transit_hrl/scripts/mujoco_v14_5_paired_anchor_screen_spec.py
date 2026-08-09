"""Frozen development design for the MuJoCo v14.5 paired-anchor screen."""

from __future__ import annotations

import math


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_5_paired_anchor_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_5_paired_anchor_continuation"
)
FROZEN_ALGORITHM_REVISION = "d3201ad9b12a558fbcec10887d3c9217c6c2585c"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "dfa329af4eb2aa9dc6206cd2e2ae712936e835c7731ef8b6e6fa958629831fa0"
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
        "responsibility_mode": "causal_lf_transfer",
        "lower_action_router_alpha": 0.04,
        "lower_action_router_observe_strength": True,
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
    "lower_action_router_mode": "direct",
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
    upper_anchor_coef: float,
    lower_anchor_coef: float,
) -> dict[str, object]:
    direct = math.isclose(float(router_strength), 0.0)
    return {
        **_base_spec(),
        "lower_action_router_mode": (
            "direct" if direct else "causal_ema_high_pass"
        ),
        "lower_action_router_strength": float(router_strength),
        "lower_action_router_training_schedule": (
            "constant" if direct else "delayed_linear"
        ),
        "lower_action_router_warmup_fraction": 0.0,
        "lower_action_router_ramp_fraction": 0.0 if direct else 0.25,
        "upper_actor_anchor_coef": float(upper_anchor_coef),
        "lower_actor_anchor_coef": float(lower_anchor_coef),
    }


ARMS = {
    "direct_continuation_control": _continuation_arm(
        router_strength=0.0,
        upper_anchor_coef=0.0,
        lower_anchor_coef=0.0,
    ),
    "router_s010_ua000_la000": _continuation_arm(
        router_strength=0.10,
        upper_anchor_coef=0.0,
        lower_anchor_coef=0.0,
    ),
    "router_s010_ua000_la001": _continuation_arm(
        router_strength=0.10,
        upper_anchor_coef=0.0,
        lower_anchor_coef=0.01,
    ),
    "router_s010_ua000_la010": _continuation_arm(
        router_strength=0.10,
        upper_anchor_coef=0.0,
        lower_anchor_coef=0.10,
    ),
    "router_s010_ua000_la100": _continuation_arm(
        router_strength=0.10,
        upper_anchor_coef=0.0,
        lower_anchor_coef=1.00,
    ),
    "router_s010_ua005_la010": _continuation_arm(
        router_strength=0.10,
        upper_anchor_coef=0.05,
        lower_anchor_coef=0.10,
    ),
    "router_s010_ua020_la100": _continuation_arm(
        router_strength=0.10,
        upper_anchor_coef=0.20,
        lower_anchor_coef=1.00,
    ),
    "router_s015_ua005_la010": _continuation_arm(
        router_strength=0.15,
        upper_anchor_coef=0.05,
        lower_anchor_coef=0.10,
    ),
}
COMPARATOR_ARM = "direct_continuation_control"
CANDIDATE_ARMS = tuple(arm for arm in ARMS if arm != COMPARATOR_ARM)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    434562922, 3096243666, 4060990905, 3231325794,
    4000048231, 754736605, 1967037826, 2061821212,
    369827789, 3742451693, 4163578452, 1286048439,
    417233433, 1675004327, 210989381, 2380411461,
)
PRETRAIN_SEEDS = (2524163744, 3673336867, 613664291, 3743163679)
PRETRAIN_SELECTION_SEEDS = (2786781678, 784441900)
CONTINUATION_TRAIN_SEEDS = (1028082993, 874305515, 4216868753, 10676213)
CONTINUATION_SELECTION_SEEDS = (512633833, 1278031587)
DEVELOPMENT_EVALUATION_SEEDS = (
    1518503308, 2649873907, 2067816079, 3092442458,
    448740820, 819446978, 1287281967, 3033156082,
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

RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION = 0.0
MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION = 0.10
DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER = 0.01
DRIFT_MATERIALITY_FLOOR = (
    DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER
    * LOWER_LF_RMS_BUDGET * LOWER_LF_RMS_BUDGET
)
MINIMUM_EFFECTIVE_LOWER_ACTION_RMS_FRACTION = 0.25
MAXIMUM_ROUTER_CLIP_RATE = 0.05
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
MINIMUM_TRAINED_CHECKPOINT_FRACTION = 0.75
MINIMUM_STRICT_RESPONSIBILITY_IMPROVEMENT_CONDITIONS = 0
MINIMUM_STRICT_RAW_IMPROVEMENT_CONDITIONS = 1
SELECTION_CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 20_000

DEVELOPMENT_DISCLOSURE = (
    "This v14.5 screen was designed after inspecting the failed v14.4 router-"
    "homotopy development screen. Each environment-by-optimizer anchor is "
    "trained once with a direct router. The compute-matched direct comparator "
    "and all routed candidates then load that exact serialized checkpoint, "
    "including optimizer state, and use the same fresh continuation roots, "
    "selection roots, iteration budget, and held-out paths. Candidate actor "
    "anchors compare the current Gaussian policy with the frozen direct policy "
    "evaluated at zero router context. The screen is development evidence only."
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
