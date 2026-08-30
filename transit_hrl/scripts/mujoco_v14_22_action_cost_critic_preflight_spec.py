"""Frozen specification for the v14.22 action-cost critic preflight."""

from __future__ import annotations

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as v14_20
from scripts import mujoco_v14_21_distributional_actor_preflight_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_22_action_cost_critic_preflight_v1"
EVIDENCE_ROLE = "post_v14_21_occupancy_aware_action_cost_preflight_not_confirmatory"
PROBE_VERSION = "mujoco_action_cost_critic_restoration_probe_v1"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
CRITIC_TRAIN_ROOTS = (
    1383356771,
    2746904315,
    2274109117,
    1414036876,
    4112159287,
    245790411,
    189838443,
    2883597223,
    4238960056,
    1987545767,
    1882581177,
    4008950701,
)
CRITIC_HOLDOUT_ROOTS = (
    2446424600,
    357381402,
    3206987278,
    3023885263,
)
DESIGN_ROOTS = (
    587025278,
    2303685907,
    2317600820,
    1949918335,
    346669469,
    32380445,
    4244590366,
    1520423861,
    589269703,
    1569453812,
    66085,
    1130199418,
    1376814453,
    752213601,
    4126543259,
    3225070337,
)
VALIDATION_ROOTS = (
    1355340483,
    2114749684,
    2401645556,
    1335607378,
    781787872,
    662570279,
    2220177340,
    1440692782,
    3875960831,
    3499266629,
    3634653546,
    2323096859,
    788489380,
    3152455465,
    3445563004,
    982174536,
)
CRITIC_ENSEMBLE_SEEDS = (
    174059501,
    3501542227,
    2143749919,
    424079015,
)
CRITIC_HIDDEN_DIM = 64
CRITIC_EPOCHS = 40
CRITIC_MINIBATCH_SIZE = 1024
CRITIC_LEARNING_RATE = 1e-3
CRITIC_MINIMUM_HOLDOUT_R2 = 0.0
CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE = 0.0
MINIMUM_GRADIENT_MEDIAN_COSINE = 0.0
ACTOR_STATE_LIMIT_PER_LEVEL = 32768
ACTOR_STEP_RMS_VALUES = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6)
RISK_MODE = "mode_mean"
CVAR_ALPHA = 0.5
MINIMUM_REDUCTION = predecessor.MINIMUM_REDUCTION
FUNNEL_MULTIPLIER = predecessor.FUNNEL_MULTIPLIER
EPISODE_HORIZON = predecessor.EPISODE_HORIZON
LEAKAGE_COST_MODE = predecessor.LEAKAGE_COST_MODE
WORKERS = 24
CPU_PER_TASK = 24
RAM_MB_PER_TASK = 16384
EXPECTED_CRITIC_TRAIN_PATH_COUNT = 4 * len(CRITIC_TRAIN_ROOTS)
EXPECTED_CRITIC_HOLDOUT_PATH_COUNT = 4 * len(CRITIC_HOLDOUT_ROOTS)
EXPECTED_DESIGN_PATH_COUNT = 4 * len(DESIGN_ROOTS)
EXPECTED_VALIDATION_PATH_COUNT = 4 * len(VALIDATION_ROOTS)
EXPECTED_CANDIDATE_COUNT = len(ACTOR_STEP_RMS_VALUES)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "critic_data": (
        "sampled_current_policy_paths_with_twelve_train_roots_and_four_"
        "independent_holdout_roots_crossed_with_four_modes"
    ),
    "critic": (
        "four_bootstrap_action_conditioned_smdp_cost_critics_per_policy_level"
    ),
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
    "validation_paths": (
        "sixteen_independent_fresh_roots_crossed_with_four_modes"
    ),
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
        raise RuntimeError("v14.22 requires 48 unique seed-role roots")
    predecessor_roots = set(
        predecessor.DESIGN_ROOTS
        + predecessor.VALIDATION_ROOTS
        + v14_20.DESIGN_ROOTS
        + v14_20.VALIDATION_ROOTS
    )
    if predecessor_roots & set(flattened):
        raise RuntimeError("v14.22 roots must be fresh relative to v14.20-v14.21")
    if len(set(CRITIC_ENSEMBLE_SEEDS)) != 4:
        raise RuntimeError("v14.22 requires four unique critic seeds")
    if EXPECTED_CELL_COUNT != 3:
        raise RuntimeError("v14.22 preflight requires one cell per environment")
    if WORKERS != CPU_PER_TASK:
        raise RuntimeError("v14.22 workers must match scheduler CPU declaration")
    if RISK_MODE != "mode_mean":
        raise RuntimeError("v14.22 freezes mode-mean closed-loop risk")
    if ACTOR_STEP_RMS_VALUES != (1e-8, 3e-8, 1e-7, 3e-7, 1e-6):
        raise RuntimeError("v14.22 actor step registry drifted")


validate()
