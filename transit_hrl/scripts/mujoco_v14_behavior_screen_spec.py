"""Frozen development design for the MuJoCo v14 behavior screen."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_behavior_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_endpoint_aligned_training"
)
FROZEN_ALGORITHM_REVISION = "ae5f0c46078e97d6aa3ea291fe5d011b7b879d7f"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "2beec62e6234c00c1b95c8dd50e5e6744851c8118f11353ca34bdfef5b042659"
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
        "upper_hf_penalty_coef": 0.0,
    },
    "joint_beta0": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_hf_penalty_coef": 0.0,
    },
    "joint_beta0p5": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_hf_penalty_coef": 0.5,
    },
    "joint_beta2": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_hf_penalty_coef": 2.0,
    },
    "joint_beta8": {
        "method": "freq_hrl",
        "responsibility_mode": "causal_lf_transfer",
        "leakage_constraint_scope": "joint_behavior",
        "upper_hf_penalty_coef": 8.0,
    },
}
BASELINE_ARM = "additive_baseline"
CANDIDATE_ARMS = tuple(arm for arm in ARMS if arm != BASELINE_ARM)

# Fresh development-only namespaces. These values do not occur in the frozen
# v12 or v13 role registries. A future v15 confirmation must use new values.
OPTIMIZER_SEEDS = (
    3958240865,
    2639133191,
    3742222834,
    1194749114,
    251693988,
    3880796643,
    1411779146,
    1326282916,
)
TRAIN_SEEDS = (271587099, 1474837618, 2763631309, 756460463)
CHECKPOINT_SELECTION_SEEDS = (
    2025695210,
    2110393562,
    1725686394,
    3844889581,
)
SAFETY_SELECTION_SEEDS = (
    3756030826,
    3286536093,
    2742901426,
    2807500593,
    660363503,
    123092653,
    1187900661,
    4045830263,
    2005093473,
    556766420,
    2634618632,
    1415718076,
)
DEVELOPMENT_EVALUATION_SEEDS = (
    3909254008,
    944081776,
    3708634948,
    3952410597,
    3084109406,
    3091896499,
    1226425527,
    1800600194,
)

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 64
UPPER_PERIOD = 16
HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
LOWER_LF_RMS_BUDGET = 0.05
UPPER_HF_RMS_BUDGET = 0.10
UPPER_ACTION_SCALE = 1.0
LOWER_ACTION_SCALE = 1.0
LOWER_CONSTRAINT_UPDATE_MODE = "reward_guarded_adam_projection"
CHECKPOINT_SMOOTHING_WINDOW = 8
CHECKPOINT_MIN_DELTA = 1e-3
CHECKPOINT_EVALUATION_INTERVAL = 4

RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION = 0.10
MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION = 0.10
MAXIMUM_UPPER_HF_RMS = 0.10
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
SELECTION_CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 20_000

DEVELOPMENT_DISCLOSURE = (
    "This screen was designed after inspecting the failed MuJoCo v13 global "
    "behavioral gate. Its seeds and outcomes are development data only. The "
    "selected coefficient, if any, must be frozen before a fresh v15 seed "
    "namespace and confirmatory decision rule are committed."
)


def validate_frozen_design() -> None:
    sequences = {
        "optimizer": OPTIMIZER_SEEDS,
        "training": TRAIN_SEEDS,
        "checkpoint_selection": CHECKPOINT_SELECTION_SEEDS,
        "safety_selection": SAFETY_SELECTION_SEEDS,
        "development_evaluation": DEVELOPMENT_EVALUATION_SEEDS,
    }
    for name, values in sequences.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"invalid v14 development seed registry: {name}")
    roles = {name: set(values) for name, values in sequences.items()}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if roles[left] & roles[right]:
                raise ValueError(
                    f"v14 development seed roles overlap: {left}/{right}"
                )
    if set(ARMS) != {
        "additive_baseline",
        "joint_beta0",
        "joint_beta0p5",
        "joint_beta2",
        "joint_beta8",
    }:
        raise ValueError("v14 behavior-screen arm registry drifted")
    if len(OPTIMIZER_SEEDS) != 8:
        raise ValueError("v14 screen requires eight optimizer replicates")


validate_frozen_design()
