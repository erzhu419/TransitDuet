"""Frozen specification for the v14.21 distributional actor preflight."""

from __future__ import annotations

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_21_distributional_actor_preflight_v1"
EVIDENCE_ROLE = (
    "post_v14_20_distributional_actor_mechanism_preflight_not_confirmatory"
)
PROBE_VERSION = "mujoco_zeroth_order_actor_restoration_probe_v2"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
DIRECTION_COUNT = predecessor.DIRECTION_COUNT
DIRECTION_SEED = predecessor.DIRECTION_SEED
PERTURB_RMS = predecessor.PERTURB_RMS
STEP_RMS_VALUES = predecessor.STEP_RMS_VALUES
DESIGN_ROOTS = (
    711725209,
    631495597,
    572315226,
    4229196870,
    1365493165,
    3330743771,
    2153978727,
    4101484206,
    3910977303,
    2981862315,
    351468744,
    489947276,
    1592711096,
    3483280035,
    2316882609,
    129318130,
)
VALIDATION_ROOTS = (
    1335195189,
    3385626663,
    4240656593,
    3066758296,
    2060314670,
    357102074,
    3729304458,
    4257202954,
    2983781434,
    2794290390,
    2912879163,
    4124225309,
    716353584,
    3070994606,
    3438016882,
    2705664572,
)
RISK_MODE = "mode_mean"
CVAR_ALPHA = 0.5
MINIMUM_REDUCTION = predecessor.MINIMUM_REDUCTION
FUNNEL_MULTIPLIER = predecessor.FUNNEL_MULTIPLIER
EPISODE_HORIZON = predecessor.EPISODE_HORIZON
LEAKAGE_COST_MODE = predecessor.LEAKAGE_COST_MODE
WORKERS = 16
CPU_PER_TASK = 16
RAM_MB_PER_TASK = 8192
EXPECTED_PATH_COUNT = 64
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "actor_scope": "joint_upper_lower_deterministic_output_heads",
    "search": "eight_fixed_antithetic_rademacher_directions_and_rank_gradient",
    "design_paths": "sixteen_frozen_roots_crossed_with_four_modes",
    "validation_paths": (
        "sixteen_independent_frozen_roots_crossed_with_four_modes"
    ),
    "risk_aggregation": "four_disturbance_mode_means",
    "eligibility": (
        "zero_reward_violations_merit_reduction_at_least_1e-4_and_"
        "worst_violation_within_three_times_baseline"
    ),
    "preflight_gate": "validation_supported_in_all_three_environments",
}


def validate() -> None:
    if EXPECTED_CELL_COUNT != 3:
        raise RuntimeError("v14.21 preflight requires one cell per environment")
    if len(DESIGN_ROOTS) != 16 or len(set(DESIGN_ROOTS)) != 16:
        raise RuntimeError("v14.21 requires sixteen unique design roots")
    if len(VALIDATION_ROOTS) != 16 or len(set(VALIDATION_ROOTS)) != 16:
        raise RuntimeError("v14.21 requires sixteen unique validation roots")
    if set(DESIGN_ROOTS) & set(VALIDATION_ROOTS):
        raise RuntimeError("v14.21 design and validation roots must be disjoint")
    if set(DESIGN_ROOTS + VALIDATION_ROOTS) & set(
        predecessor.DESIGN_ROOTS + predecessor.VALIDATION_ROOTS
    ):
        raise RuntimeError("v14.21 roots must be fresh relative to v14.20")
    if EXPECTED_PATH_COUNT != 4 * len(DESIGN_ROOTS):
        raise RuntimeError("v14.21 crossed path count is inconsistent")
    if WORKERS != CPU_PER_TASK:
        raise RuntimeError("v14.21 workers must match declared scheduler CPUs")
    if RISK_MODE != "mode_mean":
        raise RuntimeError("v14.21 freezes mode-mean distributional risk")


validate()
