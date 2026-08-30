"""Frozen specification for the v14.23 frequency-horizon critic preflight."""

from __future__ import annotations

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as v14_21
from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_23_frequency_horizon_critic_preflight_v1"
EVIDENCE_ROLE = "post_v14_22_frequency_horizon_action_cost_preflight_not_confirmatory"
PROBE_VERSION = "mujoco_action_cost_critic_restoration_probe_v2"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
CRITIC_TRAIN_ROOTS = (
    3619826213,
    2313128229,
    1593139171,
    3888234081,
    3139057849,
    3504254337,
    989587494,
    3103217913,
    1354268184,
    805670046,
    357650441,
    3555814958,
)
CRITIC_HOLDOUT_ROOTS = (
    2914010057,
    1696852596,
    2009467421,
    458117594,
)
DESIGN_ROOTS = (
    3313214588,
    860409329,
    3134492136,
    276095621,
    286787529,
    2197176711,
    595980720,
    2986238767,
    3336654733,
    934753863,
    3834356617,
    3224158205,
    2532697319,
    123040806,
    2498332680,
    876927083,
)
VALIDATION_ROOTS = (
    3855229228,
    4193760088,
    3676052749,
    1024603506,
    503912165,
    3392112073,
    1011003024,
    1701550806,
    215445838,
    2859058148,
    2435984734,
    3885123317,
    1976185965,
    735329059,
    284076891,
    3081620131,
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
UPPER_COST_RETURN_HORIZON_DECISIONS = 8
LOWER_COST_RETURN_HORIZON_DECISIONS = 32
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
EXPECTED_CRITIC_TRAIN_PATH_COUNT = 4 * len(CRITIC_TRAIN_ROOTS)
EXPECTED_CRITIC_HOLDOUT_PATH_COUNT = 4 * len(CRITIC_HOLDOUT_ROOTS)
EXPECTED_DESIGN_PATH_COUNT = 4 * len(DESIGN_ROOTS)
EXPECTED_VALIDATION_PATH_COUNT = 4 * len(VALIDATION_ROOTS)
EXPECTED_CANDIDATE_COUNT = len(ACTOR_STEP_RMS_VALUES)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "target_change": (
        "upper_eight_decision_and_lower_thirty_two_decision_duration_"
        "discounted_native_cost_returns_aligned_to_frequency_windows"
    ),
    "critic_data": (
        "sampled_current_policy_paths_with_twelve_train_roots_and_four_"
        "independent_holdout_roots_crossed_with_four_modes"
    ),
    "critic": "four_bootstrap_action_conditioned_cost_critics_per_policy_level",
    "critic_gate": (
        "upper_and_lower_holdout_r2_above_zero_true_action_prediction_"
        "better_than_fixed_permuted_action_prediction_and_median_actor_"
        "gradient_cosine_above_zero"
    ),
    "actor_direction": (
        "equal_block_rms_upper_lower_deterministic_action_cost_gradient_on_"
        "current_closed_loop_design_occupancy"
    ),
    "design_paths": "sixteen_fresh_roots_crossed_with_four_modes",
    "validation_paths": "sixteen_independent_fresh_roots_crossed_with_four_modes",
    "eligibility": (
        "zero_reward_violations_merit_reduction_at_least_1e-4_and_"
        "worst_violation_within_three_times_baseline"
    ),
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
        raise RuntimeError("v14.23 requires 48 unique seed-role roots")
    previous = set(
        v14_20.DESIGN_ROOTS
        + v14_20.VALIDATION_ROOTS
        + v14_21.DESIGN_ROOTS
        + v14_21.VALIDATION_ROOTS
        + predecessor.CRITIC_TRAIN_ROOTS
        + predecessor.CRITIC_HOLDOUT_ROOTS
        + predecessor.DESIGN_ROOTS
        + predecessor.VALIDATION_ROOTS
    )
    if previous & set(flattened):
        raise RuntimeError("v14.23 roots must be fresh relative to v14.20-v14.22")
    if EXPECTED_CELL_COUNT != 3:
        raise RuntimeError("v14.23 preflight requires one cell per environment")
    if UPPER_COST_RETURN_HORIZON_DECISIONS != 8:
        raise RuntimeError("v14.23 upper target must span eight decisions")
    if LOWER_COST_RETURN_HORIZON_DECISIONS != 32:
        raise RuntimeError("v14.23 lower target must span thirty-two decisions")
    if WORKERS != CPU_PER_TASK:
        raise RuntimeError("v14.23 workers must match scheduler CPU declaration")


validate()
