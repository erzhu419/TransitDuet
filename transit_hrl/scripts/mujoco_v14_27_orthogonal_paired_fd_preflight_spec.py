"""Frozen specification for the v14.27 orthogonal paired-FD preflight."""

from __future__ import annotations

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as v14_21
from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as v14_22
from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as v14_23
from scripts import mujoco_v14_24_paired_intervention_critic_preflight_spec as v14_24
from scripts import mujoco_v14_25_bias_matched_critic_preflight_spec as v14_25
from scripts import mujoco_v14_26_robust_paired_fd_preflight_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_27_orthogonal_paired_fd_preflight_v1"
EVIDENCE_ROLE = "post_v14_26_orthogonal_paired_fd_actor_preflight_not_confirmatory"
PROBE_VERSION = "mujoco_action_cost_critic_restoration_probe_v6"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
CRITIC_TRAIN_ROOTS = (
    2341539053, 1593737913, 2442527320, 2351593385,
    3945474337, 2218529720, 2326468774, 3861810763,
    3255064811, 1812841007, 3320361562, 2975804388,
    3473116437, 2890071929, 641699132, 1983316418,
)
CRITIC_HOLDOUT_ROOTS = (
    1512772745, 1930267031, 112600516, 2561536851,
    725277648, 1229988405, 1489028792, 192357863,
    1971339632, 1853797387, 2214575789, 3670123983,
    4148630563, 3997291849, 3358117125, 3341200952,
)
DESIGN_ROOTS = (
    1695273745, 599873195, 1484650783, 3156649517,
    4270003400, 2563863455, 3383764914, 2113885735,
    3471526238, 3629988227, 2368214362, 3559429585,
    2159857915, 1328585154, 1838651495, 2099019506,
)
VALIDATION_ROOTS = (
    4215758363, 3266198277, 992016681, 2785724813,
    3710482441, 3193767103, 1051985344, 524567585,
    730118100, 2689600890, 2182542049, 1073224181,
    3269094836, 1663570222, 1696662002, 3432926941,
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
CRITIC_INTERVENTION_DIRECTION_SCHEME = "balanced_hadamard"
CRITIC_INTERVENTION_HADAMARD_ORDER = 8
PAIRED_DIRECTION_ESTIMATOR = "orthogonal_least_squares"
UPPER_COST_RETURN_HORIZON_DECISIONS = (
    predecessor.UPPER_COST_RETURN_HORIZON_DECISIONS
)
LOWER_COST_RETURN_HORIZON_DECISIONS = (
    predecessor.LOWER_COST_RETURN_HORIZON_DECISIONS
)
ACTOR_UPDATE_SCOPE = predecessor.ACTOR_UPDATE_SCOPE
ACTOR_DIRECTION_SOURCE = predecessor.ACTOR_DIRECTION_SOURCE
MINIMUM_PAIRED_HOLDOUT_COSINE = predecessor.MINIMUM_PAIRED_HOLDOUT_COSINE
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
DISTURBANCE_MODE_COUNT = predecessor.DISTURBANCE_MODE_COUNT
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
    "critic_data": (
        "deterministic_control_and_level_isolated_antithetic_bias_paths_"
        "with_two_replicates_of_each_hadamard_row_per_mode"
    ),
    "intervention": (
        "balanced_order_eight_hadamard_raw_actor_output_bias_at_rms_0p25"
    ),
    "target": predecessor.SELECTION_CONTRACT["target"],
    "critic": predecessor.SELECTION_CONTRACT["critic"],
    "critic_gate": predecessor.SELECTION_CONTRACT["critic_gate"],
    "actor_direction": (
        "full_rank_least_squares_solution_of_antithetic_directional_cost_"
        "derivatives_with_equal_rms_upper_lower_blocks"
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
    if len(flattened) != 64 or len(flattened) != len(set(flattened)):
        raise RuntimeError("v14.27 requires 64 unique seed-role roots")
    previous = set(
        v14_20.DESIGN_ROOTS
        + v14_20.VALIDATION_ROOTS
        + v14_21.DESIGN_ROOTS
        + v14_21.VALIDATION_ROOTS
    )
    for source in (v14_22, v14_23, v14_24, v14_25, predecessor):
        previous.update(
            source.CRITIC_TRAIN_ROOTS
            + source.CRITIC_HOLDOUT_ROOTS
            + source.DESIGN_ROOTS
            + source.VALIDATION_ROOTS
        )
    if previous & set(flattened):
        raise RuntimeError("v14.27 roots must be fresh relative to v14.20-v14.26")
    if CRITIC_INTERVENTION_DIRECTION_SCHEME != "balanced_hadamard":
        raise RuntimeError("v14.27 must use balanced Hadamard interventions")
    if CRITIC_INTERVENTION_HADAMARD_ORDER != 8:
        raise RuntimeError("v14.27 requires an order-eight Hadamard design")
    if PAIRED_DIRECTION_ESTIMATOR != "orthogonal_least_squares":
        raise RuntimeError("v14.27 must solve paired directional derivatives")
    if len(CRITIC_TRAIN_ROOTS) % CRITIC_INTERVENTION_HADAMARD_ORDER:
        raise RuntimeError("v14.27 train roots must balance Hadamard rows")
    if len(CRITIC_HOLDOUT_ROOTS) % CRITIC_INTERVENTION_HADAMARD_ORDER:
        raise RuntimeError("v14.27 holdout roots must balance Hadamard rows")
    if EXPECTED_CRITIC_TRAIN_PATH_COUNT != 320:
        raise RuntimeError("v14.27 requires 320 intervention train paths")
    if EXPECTED_CRITIC_HOLDOUT_PATH_COUNT != 320:
        raise RuntimeError("v14.27 requires 320 intervention holdout paths")
    if EXPECTED_CELL_COUNT != 3 or WORKERS != CPU_PER_TASK:
        raise RuntimeError("v14.27 cell or scheduler contract drifted")


validate()
