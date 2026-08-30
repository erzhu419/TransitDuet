"""Frozen specification for the v14.26 robust paired-FD preflight."""

from __future__ import annotations

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as v14_21
from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as v14_22
from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as v14_23
from scripts import mujoco_v14_24_paired_intervention_critic_preflight_spec as v14_24
from scripts import mujoco_v14_25_bias_matched_critic_preflight_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_26_robust_paired_fd_preflight_v1"
EVIDENCE_ROLE = "post_v14_25_robust_paired_fd_actor_preflight_not_confirmatory"
PROBE_VERSION = "mujoco_action_cost_critic_restoration_probe_v5"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
CRITIC_TRAIN_ROOTS = (
    2541714852,
    3830334948,
    2901952511,
    1877922706,
    749194464,
    4087995062,
    1654168131,
    2045354363,
)
CRITIC_HOLDOUT_ROOTS = (
    3977343434,
    2707398341,
    3700120474,
    3747428517,
    817987063,
    382230945,
    1098435176,
    2211779587,
)
DESIGN_ROOTS = (
    2866239657,
    4161094575,
    4033735344,
    1662612561,
    1576919891,
    1770205521,
    1279217467,
    3707314250,
    2095263217,
    2298735595,
    3455777110,
    100767080,
    797763680,
    1490513946,
    2351071295,
    3614932939,
)
VALIDATION_ROOTS = (
    1558133677,
    96888691,
    1645388073,
    3488787739,
    3487742878,
    861932601,
    591676869,
    714467944,
    1277609673,
    3135056609,
    1693928031,
    3864732639,
    3129813184,
    142496086,
    3476378228,
    205833719,
)
CRITIC_ENSEMBLE_SEEDS = predecessor.CRITIC_ENSEMBLE_SEEDS
CRITIC_HIDDEN_DIM = predecessor.CRITIC_HIDDEN_DIM
CRITIC_EPOCHS = predecessor.CRITIC_EPOCHS
CRITIC_MINIBATCH_SIZE = predecessor.CRITIC_MINIBATCH_SIZE
CRITIC_LEARNING_RATE = predecessor.CRITIC_LEARNING_RATE
CRITIC_MINIMUM_HOLDOUT_R2 = predecessor.CRITIC_MINIMUM_HOLDOUT_R2
CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE = (
    predecessor.CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE
)
MINIMUM_GRADIENT_MEDIAN_COSINE = predecessor.MINIMUM_GRADIENT_MEDIAN_COSINE
CRITIC_COLLECTION_MODE = predecessor.CRITIC_COLLECTION_MODE
CRITIC_INTERVENTION_BIAS_RMS = predecessor.CRITIC_INTERVENTION_BIAS_RMS
CRITIC_INTERVENTION_VARIANTS = predecessor.CRITIC_INTERVENTION_VARIANTS
UPPER_COST_RETURN_HORIZON_DECISIONS = (
    predecessor.UPPER_COST_RETURN_HORIZON_DECISIONS
)
LOWER_COST_RETURN_HORIZON_DECISIONS = (
    predecessor.LOWER_COST_RETURN_HORIZON_DECISIONS
)
ACTOR_UPDATE_SCOPE = "output_bias"
ACTOR_DIRECTION_SOURCE = "paired_finite_difference"
MINIMUM_PAIRED_HOLDOUT_COSINE = 0.0
ACTOR_STATE_LIMIT_PER_LEVEL = predecessor.ACTOR_STATE_LIMIT_PER_LEVEL
ACTOR_STEP_RMS_VALUES = predecessor.ACTOR_STEP_RMS_VALUES
RISK_MODE = predecessor.RISK_MODE
CVAR_ALPHA = predecessor.CVAR_ALPHA
MINIMUM_REDUCTION = predecessor.MINIMUM_REDUCTION
FUNNEL_MULTIPLIER = predecessor.FUNNEL_MULTIPLIER
EPISODE_HORIZON = predecessor.EPISODE_HORIZON
LEAKAGE_COST_MODE = predecessor.LEAKAGE_COST_MODE
WORKERS = predecessor.WORKERS
CPU_PER_TASK = predecessor.CPU_PER_TASK
RAM_MB_PER_TASK = predecessor.RAM_MB_PER_TASK
DISTURBANCE_MODE_COUNT = 4
EXPECTED_CRITIC_TRAIN_BASE_PATH_COUNT = (
    DISTURBANCE_MODE_COUNT * len(CRITIC_TRAIN_ROOTS)
)
EXPECTED_CRITIC_HOLDOUT_BASE_PATH_COUNT = (
    DISTURBANCE_MODE_COUNT * len(CRITIC_HOLDOUT_ROOTS)
)
EXPECTED_CRITIC_TRAIN_PATH_COUNT = (
    len(CRITIC_INTERVENTION_VARIANTS) * EXPECTED_CRITIC_TRAIN_BASE_PATH_COUNT
)
EXPECTED_CRITIC_HOLDOUT_PATH_COUNT = (
    len(CRITIC_INTERVENTION_VARIANTS) * EXPECTED_CRITIC_HOLDOUT_BASE_PATH_COUNT
)
EXPECTED_DESIGN_PATH_COUNT = DISTURBANCE_MODE_COUNT * len(DESIGN_ROOTS)
EXPECTED_VALIDATION_PATH_COUNT = DISTURBANCE_MODE_COUNT * len(VALIDATION_ROOTS)
EXPECTED_PAIRED_TRAIN_DIRECTION_COUNT = EXPECTED_CRITIC_TRAIN_BASE_PATH_COUNT
EXPECTED_PAIRED_HOLDOUT_DIRECTION_COUNT = EXPECTED_CRITIC_HOLDOUT_BASE_PATH_COUNT
EXPECTED_CANDIDATE_COUNT = len(ACTOR_STEP_RMS_VALUES)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "critic_data": predecessor.SELECTION_CONTRACT["critic_data"],
    "intervention": predecessor.SELECTION_CONTRACT["intervention"],
    "target": predecessor.SELECTION_CONTRACT["target"],
    "critic": predecessor.SELECTION_CONTRACT["critic"],
    "critic_gate": (
        "upper_and_lower_holdout_r2_and_action_permutation_gain_above_zero_"
        "plus_positive_train_holdout_paired_direction_cosine_overall_and_"
        "within_every_disturbance_mode"
    ),
    "actor_direction": (
        "coordinate_median_of_pathwise_antithetic_output_bias_finite_"
        "differences_with_equal_rms_upper_lower_blocks"
    ),
    "design_paths": "sixteen_fresh_unintervened_roots_crossed_with_four_modes",
    "validation_paths": (
        "sixteen_independent_unintervened_roots_crossed_with_four_modes"
    ),
    "eligibility": predecessor.SELECTION_CONTRACT["eligibility"],
    "preflight_gate": "validation_supported_in_all_three_environments",
}


def validate() -> None:
    roles = (
        CRITIC_TRAIN_ROOTS,
        CRITIC_HOLDOUT_ROOTS,
        DESIGN_ROOTS,
        VALIDATION_ROOTS,
    )
    flattened = [int(root) for role in roles for root in role]
    if len(flattened) != 48 or len(flattened) != len(set(flattened)):
        raise RuntimeError("v14.26 requires 48 unique seed-role roots")
    previous = set(
        v14_20.DESIGN_ROOTS
        + v14_20.VALIDATION_ROOTS
        + v14_21.DESIGN_ROOTS
        + v14_21.VALIDATION_ROOTS
    )
    for source in (v14_22, v14_23, v14_24, predecessor):
        previous.update(
            source.CRITIC_TRAIN_ROOTS
            + source.CRITIC_HOLDOUT_ROOTS
            + source.DESIGN_ROOTS
            + source.VALIDATION_ROOTS
        )
    if previous & set(flattened):
        raise RuntimeError("v14.26 roots must be fresh relative to v14.20-v14.25")
    if ACTOR_UPDATE_SCOPE != "output_bias":
        raise RuntimeError("v14.26 must update only actor output biases")
    if ACTOR_DIRECTION_SOURCE != "paired_finite_difference":
        raise RuntimeError("v14.26 must use paired finite-difference directions")
    if MINIMUM_PAIRED_HOLDOUT_COSINE != 0.0:
        raise RuntimeError("v14.26 direction agreement threshold drifted")
    if EXPECTED_CRITIC_TRAIN_PATH_COUNT != 160:
        raise RuntimeError("v14.26 requires 160 intervention train paths")
    if EXPECTED_CRITIC_HOLDOUT_PATH_COUNT != 160:
        raise RuntimeError("v14.26 requires 160 intervention holdout paths")
    if EXPECTED_CELL_COUNT != 3 or WORKERS != CPU_PER_TASK:
        raise RuntimeError("v14.26 cell or scheduler contract drifted")


validate()
