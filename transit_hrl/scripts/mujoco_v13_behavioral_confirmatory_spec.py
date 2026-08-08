"""Frozen design constants for the MuJoCo v13 behavioral confirmatory experiment."""

from __future__ import annotations


CONFIRMATORY_PROTOCOL_VERSION = "mujoco_v13_behavioral_confirmatory_v1"
RUNTIME_ADAPTER_VERSION = "mujoco_v13_behavioral_source_preserving_runtime_v1"
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

# These seed namespaces are fresh relative to v12 and are committed before v13
# held-out access. The v13 endpoints were developed after inspecting v12.
OPTIMIZER_SEEDS = (
    3596220588, 707075335, 2102554650, 729733352,
    2677106215, 2294726814, 180002411, 2597703301,
    4120591705, 2971934405, 3557572450, 514480913,
    3888877169, 3466954944, 2796825593, 1083410249,
    3094989646, 402019680, 3276696591, 3507872935,
    2829590401, 655399053, 4222339063, 1548381879,
)
TRAIN_SEEDS = (4264032191, 2100677060, 4239844877, 1144300253)
CHECKPOINT_SELECTION_SEEDS = (
    2866619922, 4075269154, 2027354516, 2490162525,
)
SAFETY_SELECTION_SEEDS = (
    3790153287, 4213534078, 1672261794, 352246897,
    1939359332, 1422921, 801293948, 3997314877,
    1713598018, 533171056, 1333501423, 1531967902,
)
HELDOUT_EVALUATION_SEEDS = (
    1121714046, 1262290433, 722017948, 636775586,
    4031344500, 2096612910, 2973127221, 459425468,
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
MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION = 0.10
MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION = 0.10
MAXIMUM_UPPER_HF_RMS = 0.10
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
FAMILY_WISE_ALPHA = 0.05
PRIMARY_GATE_COUNT = len(ENVIRONMENTS) * 4
PER_GATE_ONE_SIDED_CONFIDENCE = 1.0 - FAMILY_WISE_ALPHA / PRIMARY_GATE_COUNT
BOOTSTRAP_DRAWS = 50_000

DEVELOPMENT_DISCLOSURE = (
    "The behavioral endpoints and numerical thresholds were developed after "
    "exploratory inspection of MuJoCo v12. All optimizer, training, checkpoint, "
    "safety-selection, and held-out evaluation seeds are fresh for v13, and the "
    "v13 decision rule is frozen before access to any v13 held-out result."
)


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
    if PRIMARY_GATE_COUNT != 12:
        raise ValueError("behavioral confirmatory design requires 12 statistical gates")


validate_frozen_design()
