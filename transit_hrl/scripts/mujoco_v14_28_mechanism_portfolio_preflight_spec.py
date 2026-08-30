"""Frozen specification for the v14.28 restoration portfolio preflight."""

from __future__ import annotations

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as v14_21
from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as v14_22
from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as v14_23
from scripts import mujoco_v14_24_paired_intervention_critic_preflight_spec as v14_24
from scripts import mujoco_v14_25_bias_matched_critic_preflight_spec as v14_25
from scripts import mujoco_v14_26_robust_paired_fd_preflight_spec as v14_26
from scripts import mujoco_v14_27_orthogonal_paired_fd_preflight_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_28_mechanism_portfolio_preflight_v1"
EVIDENCE_ROLE = "post_v14_27_domain_general_restoration_portfolio_not_confirmatory"
PROBE_VERSION = "mujoco_action_cost_critic_restoration_probe_v7"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
CRITIC_TRAIN_ROOTS = (
    2635685950, 4275142973, 2934225320, 2667329081,
    1359715721, 3800259984, 2505057269, 3921045193,
    2699589421, 2794424016, 472203987, 1753300390,
    689620208, 544695928, 2203328758, 3376965464,
)
CRITIC_HOLDOUT_ROOTS = (
    3799391715, 3993572440, 974497084, 163787043,
    3637068610, 2267061199, 1246897756, 2051960726,
    3482589468, 2099108563, 1442716059, 4231363777,
    3476370538, 1728093522, 2102399453, 3172252133,
)
DESIGN_ROOTS = (
    3132599364, 1364749090, 1064211356, 3579465298,
    1741140119, 1320788946, 2596373894, 1506916250,
    1376078713, 4073369538, 1207271460, 1114649551,
    555516549, 2752728282, 3513363009, 46274092,
    2059567448, 2343320941, 1541392517, 3771074943,
    597340748, 3139054340, 2609735942, 4062931090,
    2854316005, 1940371314, 2097761885, 3309130602,
    2798382068, 4117922366, 2710884694, 1594266442,
)
VALIDATION_ROOTS = (
    490036223, 2626204884, 817877776, 3473150139,
    3755566892, 4149552094, 232500480, 1983047458,
    2929904834, 1524972511, 2335733896, 1572931009,
    3593693541, 2140036629, 2495691324, 4152162674,
    1853816039, 2419224946, 3120120981, 1698471534,
    2067474276, 4016003115, 628000576, 1500328319,
    3701043817, 168382851, 1549186156, 1819693125,
    3983755012, 4139404871, 2314498953, 753812111,
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
CRITIC_INTERVENTION_DIRECTION_SCHEME = (
    predecessor.CRITIC_INTERVENTION_DIRECTION_SCHEME
)
CRITIC_INTERVENTION_HADAMARD_ORDER = (
    predecessor.CRITIC_INTERVENTION_HADAMARD_ORDER
)
PAIRED_DIRECTION_ESTIMATOR = predecessor.PAIRED_DIRECTION_ESTIMATOR
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
BASELINE_ROUTER_STRENGTH = 0.5
ROUTER_STRENGTH_VALUES = (
    0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0,
)
DESIGN_FOLD_COUNT = 2
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
EXPECTED_DESIGN_FOLD_PATH_COUNT = EXPECTED_DESIGN_PATH_COUNT // DESIGN_FOLD_COUNT
EXPECTED_PAIRED_TRAIN_DIRECTION_COUNT = EXPECTED_CRITIC_TRAIN_BASE_PATH_COUNT
EXPECTED_PAIRED_HOLDOUT_DIRECTION_COUNT = EXPECTED_CRITIC_HOLDOUT_BASE_PATH_COUNT
EXPECTED_ACTOR_CANDIDATE_COUNT = len(ACTOR_STEP_RMS_VALUES)
EXPECTED_ROUTER_CANDIDATE_COUNT = len(ROUTER_STRENGTH_VALUES)
EXPECTED_CANDIDATE_COUNT = (
    EXPECTED_ACTOR_CANDIDATE_COUNT + EXPECTED_ROUTER_CANDIDATE_COUNT
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "mechanism_portfolio": (
        "same_environment_agnostic_registry_of_orthogonal_actor_steps_and_"
        "function_preserving_router_strengths_in_every_cell"
    ),
    "actor_direction": predecessor.SELECTION_CONTRACT["actor_direction"],
    "router_candidates": (
        "ten_nonbaseline_strengths_from_zero_to_one_at_0p1_resolution"
    ),
    "design_paths": (
        "thirty_two_fresh_roots_crossed_with_four_modes_and_split_into_two_"
        "predeclared_sixteen_root_folds"
    ),
    "fold_gate": (
        "pooled_and_both_design_folds_must_independently_pass_the_same_"
        "reward_frequency_eligibility_rule"
    ),
    "validation_paths": (
        "thirty_two_independent_unintervened_roots_crossed_with_four_modes"
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
    if len(flattened) != 96 or len(flattened) != len(set(flattened)):
        raise RuntimeError("v14.28 requires 96 unique seed-role roots")
    previous = set(
        v14_20.DESIGN_ROOTS
        + v14_20.VALIDATION_ROOTS
        + v14_21.DESIGN_ROOTS
        + v14_21.VALIDATION_ROOTS
    )
    for source in (
        v14_22, v14_23, v14_24, v14_25, v14_26, predecessor,
    ):
        previous.update(
            source.CRITIC_TRAIN_ROOTS
            + source.CRITIC_HOLDOUT_ROOTS
            + source.DESIGN_ROOTS
            + source.VALIDATION_ROOTS
        )
    if previous & set(flattened):
        raise RuntimeError("v14.28 roots must be fresh relative to v14.20-v14.27")
    if BASELINE_ROUTER_STRENGTH in ROUTER_STRENGTH_VALUES:
        raise RuntimeError("v14.28 router registry must exclude the baseline")
    if ROUTER_STRENGTH_VALUES != tuple(value / 10 for value in range(11) if value != 5):
        raise RuntimeError("v14.28 router registry drifted")
    if len(DESIGN_ROOTS) % DESIGN_FOLD_COUNT:
        raise RuntimeError("v14.28 design folds must contain equal root counts")
    if EXPECTED_DESIGN_FOLD_PATH_COUNT != 64:
        raise RuntimeError("v14.28 requires 64 paths per design fold")
    if EXPECTED_CRITIC_TRAIN_PATH_COUNT != 320:
        raise RuntimeError("v14.28 requires 320 intervention train paths")
    if EXPECTED_CRITIC_HOLDOUT_PATH_COUNT != 320:
        raise RuntimeError("v14.28 requires 320 intervention holdout paths")
    if EXPECTED_DESIGN_PATH_COUNT != 128 or EXPECTED_VALIDATION_PATH_COUNT != 128:
        raise RuntimeError("v14.28 design/validation path counts drifted")
    if EXPECTED_CANDIDATE_COUNT != 15:
        raise RuntimeError("v14.28 requires fifteen portfolio candidates")
    if EXPECTED_CELL_COUNT != 3 or WORKERS != CPU_PER_TASK:
        raise RuntimeError("v14.28 cell or scheduler contract drifted")


validate()
