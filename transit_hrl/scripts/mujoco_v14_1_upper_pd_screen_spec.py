"""Frozen development design for the MuJoCo v14.1 algorithm screen."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_1_upper_pd_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_1_crossed_behavior_selection"
)
FROZEN_ALGORITHM_REVISION = "4c9e2cee31c03d7dcdaa8f1a1a4bc763f356606f"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "e93d59638a36a72cb7715bfe02f1728cfc0aaf2e08fb055815930e6bcf2a9d7c"
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

ARMS = {
    "additive_baseline": {
        "method": "freq_hrl_no_leakage",
        "responsibility_mode": "additive",
        "leakage_constraint_scope": "responsibility",
        "upper_constraint_mode": "disabled",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 0.0,
    },
    "crossed_joint_beta0": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_constraint_mode": "disabled",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 0.0,
    },
    "crossed_joint_beta0p5": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_constraint_mode": "static_reward_penalty",
        "upper_hf_penalty_coef": 0.5,
        "upper_dual_lr": 0.0,
    },
    "crossed_joint_pd0p5": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_constraint_mode": "primal_dual",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 0.5,
    },
    "crossed_joint_pd2": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_constraint_mode": "primal_dual",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 2.0,
    },
    "crossed_joint_pd8": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_constraint_mode": "primal_dual",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 8.0,
    },
}
BASELINE_ARM = "additive_baseline"
CANDIDATE_ARMS = tuple(arm for arm in ARMS if arm != BASELINE_ARM)

# Fresh development-only namespace, checked against the v12-v14 registries
# before this design was frozen. Any v15 confirmation must use new values.
OPTIMIZER_SEEDS = (
    2384341569,
    2171910770,
    909935778,
    2373555520,
    769629071,
    2603186137,
    806881874,
    1476542116,
)
TRAIN_SEEDS = (1306335389, 2269532698, 352405062, 70448635)
CHECKPOINT_SELECTION_SEEDS = (1518484598, 2637913576)
DEVELOPMENT_EVALUATION_SEEDS = (
    2696100381,
    4165531049,
    900204231,
    3310622046,
    67063853,
    2501987370,
    4270294130,
    2613965647,
)

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 64
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
CHECKPOINT_SCORE_MODE = "behavior_robust"
CHECKPOINT_CONSTRAINT_PENALTY = 10.0
CHECKPOINT_SMOOTHING_WINDOW = 8
CHECKPOINT_MIN_DELTA = 1e-3
CHECKPOINT_EVALUATION_INTERVAL = 4

RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION = 0.10
MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION = 0.10
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
SELECTION_CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 20_000

DEVELOPMENT_DISCLOSURE = (
    "This v14.1 screen was designed after inspecting the failed v14 fixed-"
    "coefficient screen. It evaluates crossed-condition checkpoint selection "
    "and upper primal-dual rates using fresh development seeds. Its outcomes "
    "cannot support a confirmatory claim; any selected arm must be frozen "
    "before a fresh v15 seed namespace is generated."
)


def validate_frozen_design() -> None:
    sequences = {
        "optimizer": OPTIMIZER_SEEDS,
        "training": TRAIN_SEEDS,
        "checkpoint_selection": CHECKPOINT_SELECTION_SEEDS,
        "development_evaluation": DEVELOPMENT_EVALUATION_SEEDS,
    }
    for name, values in sequences.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"invalid v14.1 development seed registry: {name}")
    roles = {name: set(values) for name, values in sequences.items()}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if roles[left] & roles[right]:
                raise ValueError(
                    f"v14.1 development seed roles overlap: {left}/{right}"
                )
    if len(OPTIMIZER_SEEDS) != 8:
        raise ValueError("v14.1 screen requires eight optimizer replicates")
    if len(CHECKPOINT_SELECTION_SEEDS) != 2:
        raise ValueError("v14.1 screen requires two crossed selection roots")
    if set(ARMS) != {
        "additive_baseline",
        "crossed_joint_beta0",
        "crossed_joint_beta0p5",
        "crossed_joint_pd0p5",
        "crossed_joint_pd2",
        "crossed_joint_pd8",
    }:
        raise ValueError("v14.1 algorithm-screen arm registry drifted")
    if any(
        arm["upper_constraint_mode"] == "primal_dual"
        and float(arm["upper_dual_lr"]) <= 0.0
        for arm in ARMS.values()
    ):
        raise ValueError("v14.1 primal-dual arms require a positive dual rate")


validate_frozen_design()
