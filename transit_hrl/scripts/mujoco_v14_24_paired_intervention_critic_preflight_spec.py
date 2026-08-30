"""Frozen specification for the v14.24 paired-intervention critic preflight."""

from __future__ import annotations

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as v14_21
from scripts import mujoco_v14_22_action_cost_critic_preflight_spec as v14_22
from scripts import mujoco_v14_23_frequency_horizon_critic_preflight_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_24_paired_intervention_critic_preflight_v1"
EVIDENCE_ROLE = "post_v14_23_paired_action_intervention_preflight_not_confirmatory"
PROBE_VERSION = "mujoco_action_cost_critic_restoration_probe_v3"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
CRITIC_TRAIN_ROOTS = (
    2986607454,
    604484376,
    4161364804,
    849461190,
    1021036399,
    1933461708,
    1628763506,
    4265225819,
)
CRITIC_HOLDOUT_ROOTS = (
    1819186136,
    1437410567,
    2653257839,
    1917551321,
)
DESIGN_ROOTS = (
    1805898593,
    941539579,
    2654352000,
    3075430224,
    2512389827,
    2666962646,
    3689545477,
    1151153810,
    3716503300,
    180912837,
    88071634,
    845938369,
    691442754,
    766245497,
    1765759876,
    2974658318,
)
VALIDATION_ROOTS = (
    852526433,
    988088971,
    1937175766,
    744553142,
    7770843,
    1734851965,
    1248870913,
    876471757,
    523270021,
    247322838,
    4176967874,
    2704195646,
    1402444409,
    1394818676,
    2106546975,
    691491428,
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
CRITIC_COLLECTION_MODE = "paired_output_bias"
CRITIC_INTERVENTION_BIAS_RMS = 0.25
CRITIC_INTERVENTION_VARIANTS = (
    "control",
    "lower_minus",
    "lower_plus",
    "upper_minus",
    "upper_plus",
)
UPPER_COST_RETURN_HORIZON_DECISIONS = (
    predecessor.UPPER_COST_RETURN_HORIZON_DECISIONS
)
LOWER_COST_RETURN_HORIZON_DECISIONS = (
    predecessor.LOWER_COST_RETURN_HORIZON_DECISIONS
)
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
    "critic_data": (
        "deterministic_control_upper_plus_minus_and_lower_plus_minus_output_"
        "bias_interventions_paired_by_environment_path"
    ),
    "intervention": (
        "level_isolated_antithetic_raw_actor_mean_bias_with_rms_0p25"
    ),
    "target": (
        "upper_eight_decision_and_lower_thirty_two_decision_duration_"
        "discounted_native_cost_returns"
    ),
    "critic": "four_bootstrap_action_conditioned_cost_critics_per_policy_level",
    "critic_gate": (
        "upper_and_lower_holdout_r2_above_zero_true_action_prediction_"
        "better_than_fixed_permuted_action_prediction_and_median_actor_"
        "gradient_cosine_above_zero"
    ),
    "actor_direction": (
        "equal_block_rms_upper_lower_deterministic_action_cost_gradient_on_"
        "unintervened_closed_loop_design_occupancy"
    ),
    "design_paths": "sixteen_fresh_unintervened_roots_crossed_with_four_modes",
    "validation_paths": "sixteen_independent_unintervened_roots_crossed_with_four_modes",
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
    if len(flattened) != 44 or len(flattened) != len(set(flattened)):
        raise RuntimeError("v14.24 requires 44 unique seed-role roots")
    previous = set(
        v14_20.DESIGN_ROOTS
        + v14_20.VALIDATION_ROOTS
        + v14_21.DESIGN_ROOTS
        + v14_21.VALIDATION_ROOTS
        + v14_22.CRITIC_TRAIN_ROOTS
        + v14_22.CRITIC_HOLDOUT_ROOTS
        + v14_22.DESIGN_ROOTS
        + v14_22.VALIDATION_ROOTS
        + predecessor.CRITIC_TRAIN_ROOTS
        + predecessor.CRITIC_HOLDOUT_ROOTS
        + predecessor.DESIGN_ROOTS
        + predecessor.VALIDATION_ROOTS
    )
    if previous & set(flattened):
        raise RuntimeError("v14.24 roots must be fresh relative to v14.20-v14.23")
    if EXPECTED_CRITIC_TRAIN_PATH_COUNT != 160:
        raise RuntimeError("v14.24 requires 160 intervention train paths")
    if EXPECTED_CRITIC_HOLDOUT_PATH_COUNT != 80:
        raise RuntimeError("v14.24 requires 80 intervention holdout paths")
    if EXPECTED_CELL_COUNT != 3 or WORKERS != CPU_PER_TASK:
        raise RuntimeError("v14.24 cell or scheduler contract drifted")


validate()
