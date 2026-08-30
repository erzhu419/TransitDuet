"""Frozen specification for the v14.25 bias-matched critic preflight."""

from __future__ import annotations

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as v14_21
from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as v14_22
from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as v14_23
from scripts import mujoco_v14_24_paired_intervention_critic_preflight_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_25_bias_matched_critic_preflight_v1"
EVIDENCE_ROLE = "post_v14_24_bias_matched_actor_update_preflight_not_confirmatory"
PROBE_VERSION = "mujoco_action_cost_critic_restoration_probe_v4"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
CRITIC_TRAIN_ROOTS = (
    463108515,
    2353291356,
    1264983659,
    3414767292,
    4280113228,
    59188085,
    1560150663,
    1828934118,
)
CRITIC_HOLDOUT_ROOTS = (
    371680227,
    2176047177,
    737671614,
    4162047907,
)
DESIGN_ROOTS = (
    1079377154,
    3664811967,
    4030572148,
    1867246542,
    3660996657,
    684879,
    1041872867,
    116551589,
    2420590201,
    527332615,
    2474663098,
    1269883029,
    3927231003,
    494977800,
    2246035766,
    2662853794,
)
VALIDATION_ROOTS = (
    1015224266,
    3451284626,
    1370140012,
    916302353,
    1265040662,
    4282145080,
    4046390775,
    363991780,
    2900859644,
    1956794319,
    2864471245,
    1735278240,
    1127361720,
    3989817577,
    1556045787,
    1062515032,
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
ACTOR_STATE_LIMIT_PER_LEVEL = predecessor.ACTOR_STATE_LIMIT_PER_LEVEL
ACTOR_STEP_RMS_VALUES = (1e-7, 1e-6, 1e-5, 3e-5, 1e-4)
RISK_MODE = predecessor.RISK_MODE
CVAR_ALPHA = predecessor.CVAR_ALPHA
MINIMUM_REDUCTION = predecessor.MINIMUM_REDUCTION
FUNNEL_MULTIPLIER = predecessor.FUNNEL_MULTIPLIER
EPISODE_HORIZON = predecessor.EPISODE_HORIZON
LEAKAGE_COST_MODE = predecessor.LEAKAGE_COST_MODE
WORKERS = predecessor.WORKERS
CPU_PER_TASK = predecessor.CPU_PER_TASK
RAM_MB_PER_TASK = predecessor.RAM_MB_PER_TASK
EXPECTED_CRITIC_TRAIN_BASE_PATH_COUNT = 4 * len(CRITIC_TRAIN_ROOTS)
EXPECTED_CRITIC_HOLDOUT_BASE_PATH_COUNT = 4 * len(CRITIC_HOLDOUT_ROOTS)
EXPECTED_CRITIC_TRAIN_PATH_COUNT = (
    len(CRITIC_INTERVENTION_VARIANTS) * EXPECTED_CRITIC_TRAIN_BASE_PATH_COUNT
)
EXPECTED_CRITIC_HOLDOUT_PATH_COUNT = (
    len(CRITIC_INTERVENTION_VARIANTS) * EXPECTED_CRITIC_HOLDOUT_BASE_PATH_COUNT
)
EXPECTED_DESIGN_PATH_COUNT = 4 * len(DESIGN_ROOTS)
EXPECTED_VALIDATION_PATH_COUNT = 4 * len(VALIDATION_ROOTS)
EXPECTED_CANDIDATE_COUNT = len(ACTOR_STEP_RMS_VALUES)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "critic_data": predecessor.SELECTION_CONTRACT["critic_data"],
    "intervention": predecessor.SELECTION_CONTRACT["intervention"],
    "target": predecessor.SELECTION_CONTRACT["target"],
    "critic": predecessor.SELECTION_CONTRACT["critic"],
    "critic_gate": predecessor.SELECTION_CONTRACT["critic_gate"],
    "actor_direction": (
        "upper_and_lower_output_bias_only_action_cost_gradient_matching_"
        "the_intervention_estimand"
    ),
    "design_paths": "sixteen_fresh_unintervened_roots_crossed_with_four_modes",
    "validation_paths": "sixteen_independent_unintervened_roots_crossed_with_four_modes",
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
    if len(flattened) != 44 or len(flattened) != len(set(flattened)):
        raise RuntimeError("v14.25 requires 44 unique seed-role roots")
    previous = set(
        v14_20.DESIGN_ROOTS
        + v14_20.VALIDATION_ROOTS
        + v14_21.DESIGN_ROOTS
        + v14_21.VALIDATION_ROOTS
    )
    for source in (v14_22, v14_23, predecessor):
        previous.update(
            source.CRITIC_TRAIN_ROOTS
            + source.CRITIC_HOLDOUT_ROOTS
            + source.DESIGN_ROOTS
            + source.VALIDATION_ROOTS
        )
    if previous & set(flattened):
        raise RuntimeError("v14.25 roots must be fresh relative to v14.20-v14.24")
    if ACTOR_UPDATE_SCOPE != "output_bias":
        raise RuntimeError("v14.25 must match the intervention output-bias scope")
    if ACTOR_STEP_RMS_VALUES != (1e-7, 1e-6, 1e-5, 3e-5, 1e-4):
        raise RuntimeError("v14.25 output-bias step registry drifted")
    if EXPECTED_CRITIC_TRAIN_PATH_COUNT != 160:
        raise RuntimeError("v14.25 requires 160 intervention train paths")
    if EXPECTED_CRITIC_HOLDOUT_PATH_COUNT != 80:
        raise RuntimeError("v14.25 requires 80 intervention holdout paths")
    if EXPECTED_CELL_COUNT != 3 or WORKERS != CPU_PER_TASK:
        raise RuntimeError("v14.25 cell or scheduler contract drifted")


validate()
