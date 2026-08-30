"""Frozen specification for the v14.20 zeroth-order actor preflight."""

from __future__ import annotations

from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as v14_17
from scripts import mujoco_v14_19_bidirectional_router_adapter_screen_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_20_zeroth_order_actor_preflight_v1"
EVIDENCE_ROLE = "post_router_boundary_actor_mechanism_preflight_not_confirmatory"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = (4196455150,)
DIRECTION_COUNT = 8
DIRECTION_SEED = 3538198271
PERTURB_RMS = 1e-6
STEP_RMS_VALUES = (1e-8, 3e-8, 1e-7, 3e-7, 1e-6)
DESIGN_ROOTS = v14_17.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS[:2]
VALIDATION_ROOTS = v14_17.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS[2:]
MINIMUM_REDUCTION = 1e-4
FUNNEL_MULTIPLIER = 3.0
EPISODE_HORIZON = 1000
LEAKAGE_COST_MODE = "power_excess"
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "actor_scope": "joint_upper_lower_deterministic_output_heads",
    "search": "eight_fixed_antithetic_rademacher_directions_and_rank_gradient",
    "design_paths": "first_two_frozen_guard_roots_crossed_with_four_modes",
    "validation_paths": "last_two_frozen_guard_roots_crossed_with_four_modes",
    "eligibility": (
        "zero_reward_violations_merit_reduction_at_least_1e-4_and_"
        "worst_violation_within_three_times_baseline"
    ),
    "preflight_gate": "validation_supported_in_all_three_environments",
}


def validate() -> None:
    if EXPECTED_CELL_COUNT != 3:
        raise RuntimeError("v14.20 preflight requires one cell per environment")
    if set(DESIGN_ROOTS) & set(VALIDATION_ROOTS):
        raise RuntimeError("v14.20 design and validation roots must be disjoint")
    if len(DESIGN_ROOTS) != 2 or len(VALIDATION_ROOTS) != 2:
        raise RuntimeError("v14.20 requires two roots for each path role")
    if any(value > PERTURB_RMS for value in STEP_RMS_VALUES):
        raise RuntimeError("v14.20 line-search steps cannot exceed probe RMS")


validate()
