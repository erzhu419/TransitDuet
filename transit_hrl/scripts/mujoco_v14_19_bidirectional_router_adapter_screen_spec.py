"""Frozen development specification for the v14.19 router adapter screen."""

from __future__ import annotations

from scripts import mujoco_v14_18_router_mechanism_screen_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v14_19_bidirectional_router_adapter_screen_v1"
)
EVIDENCE_ROLE = "post_v14_18_adaptive_mechanism_development_not_confirmatory"
ANCHOR_RUN_NAME = predecessor.ANCHOR_RUN_NAME
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
ROUTER_STRENGTHS = tuple(index / 10.0 for index in range(11))
ACTOR_GAINS = (1.0,)
BASELINE_ROUTER_STRENGTH = 0.5
PROFILE = predecessor.PROFILE
EPISODE_HORIZON = predecessor.EPISODE_HORIZON
LEAKAGE_COST_MODE = predecessor.LEAKAGE_COST_MODE
STRICT_IMPROVEMENT_TOLERANCE = predecessor.STRICT_IMPROVEMENT_TOLERANCE
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "unit": "environment_by_optimizer_seed_anchor",
    "candidate_scope": (
        "same_bidirectional_grid_and_guard_selector_in_every_cell"
    ),
    "manual_environment_tuning": False,
    "eligibility": (
        "zero_reward_violations_and_strict_frequency_merit_reduction_"
        "vs_strength_0.5"
    ),
    "cell_selection_order": (
        "minimize_frequency_merit_then_worst_violation_then_violation_count_"
        "then_distance_from_0.5_then_strength"
    ),
    "mechanism_gate": "eligible_candidate_in_all_nine_cells",
    "development_disclosure": (
        "all_nine_anchor_outcomes_above_strength_0.5_were_inspected_in_v14.18"
    ),
}


def validate() -> None:
    if len(ENVIRONMENTS) != 3 or len(OPTIMIZER_SEEDS) != 3:
        raise RuntimeError("v14.19 requires the frozen 3 by 3 anchor registry")
    if ROUTER_STRENGTHS != tuple(index / 10.0 for index in range(11)):
        raise RuntimeError("v14.19 requires the frozen bidirectional grid")
    if BASELINE_ROUTER_STRENGTH not in ROUTER_STRENGTHS:
        raise RuntimeError("v14.19 baseline is absent from the router grid")
    if ACTOR_GAINS != (1.0,):
        raise RuntimeError("v14.19 must isolate routing from actor contraction")


validate()
