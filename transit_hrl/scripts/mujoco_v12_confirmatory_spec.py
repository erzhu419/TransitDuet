"""Frozen design constants for the MuJoCo v12 confirmatory experiment."""

from __future__ import annotations


CONFIRMATORY_PROTOCOL_VERSION = "mujoco_v12_full_method_confirmatory_v1"
RUNTIME_ADAPTER_VERSION = "mujoco_v12_source_preserving_runtime_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v11_canonical_policy_state"
)
FROZEN_ALGORITHM_REVISION = "8e47614f1005d8a064a3d6691a0ca6e5bb311ee4"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "002878a554049947768f7c1b654d92bc58ca332a272ba422bacd0764336bf5f7"
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
    },
    "transfer_full_method": {
        "method": "freq_hrl_safe_selector",
        "responsibility_mode": "causal_lf_transfer",
    },
}

# These seed namespaces were generated and committed before confirmatory access.
OPTIMIZER_SEEDS = (
    3760743984, 3694979909, 389202265, 4051790486,
    1653209591, 2850251129, 914719826, 223202612,
    4262194313, 2726429378, 3703719678, 3019160601,
    2250225306, 2768496775, 844758473, 681257194,
    1978663101, 3049442218, 661891628, 1380702675,
    3803331401, 640779130, 2570816311, 4025270303,
)
TRAIN_SEEDS = (1475780604, 922098860, 2150922656, 3112338055)
CHECKPOINT_SELECTION_SEEDS = (
    3805350496, 4287501742, 1863857547, 1385694196,
)
SAFETY_SELECTION_SEEDS = (
    1810690907, 1535441834, 2185447748, 2287802716,
    110329099, 1038981365, 623018687, 766912702,
    2378359440, 2550850667, 1013929129, 922096000,
)
HELDOUT_EVALUATION_SEEDS = (
    2207946371, 3863714629, 2841153822, 990979066,
    1224803170, 474729935, 3597166873, 2209601210,
)

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 64
UPPER_PERIOD = 16
HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
LOWER_LF_RMS_BUDGET = 0.05
UPPER_ACTION_SCALE = 1.0
LOWER_ACTION_SCALE = 1.0
LOWER_CONSTRAINT_UPDATE_MODE = "reward_guarded_adam_projection"
CHECKPOINT_SMOOTHING_WINDOW = 8
CHECKPOINT_MIN_DELTA = 1e-3
CHECKPOINT_EVALUATION_INTERVAL = 4

RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
MINIMUM_DRIFT_REDUCTION_FRACTION = 0.10
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
FAMILY_WISE_ALPHA = 0.05
PRIMARY_GATE_COUNT = len(ENVIRONMENTS) * 2
PER_GATE_ONE_SIDED_CONFIDENCE = 1.0 - FAMILY_WISE_ALPHA / PRIMARY_GATE_COUNT
BOOTSTRAP_DRAWS = 50_000


def validate_frozen_design() -> None:
    """Reject accidental seed overlap or an incomplete confirmatory registry."""

    sequences = {
        "optimizer": OPTIMIZER_SEEDS,
        "training": TRAIN_SEEDS,
        "checkpoint_selection": CHECKPOINT_SELECTION_SEEDS,
        "safety_selection": SAFETY_SELECTION_SEEDS,
        "heldout_evaluation": HELDOUT_EVALUATION_SEEDS,
    }
    for name, values in sequences.items():
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate seeds in frozen {name} role")
    roles = {name: set(values) for name, values in sequences.items()}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = roles[left] & roles[right]
            if overlap:
                raise ValueError(
                    f"frozen confirmatory seed roles overlap: {left}/{right}"
                )
    if set(ARMS) != {"additive_baseline", "transfer_full_method"}:
        raise ValueError("frozen confirmatory arm registry drifted")
    if len(OPTIMIZER_SEEDS) != 24:
        raise ValueError("confirmatory design requires 24 optimizer replicates")


validate_frozen_design()
