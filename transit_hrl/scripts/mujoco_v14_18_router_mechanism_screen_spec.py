"""Frozen development specification for the MuJoCo v14.18 router screen."""

from __future__ import annotations

from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as predecessor


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_18_router_mechanism_screen_v1"
EVIDENCE_ROLE = "adaptive_mechanism_development_not_confirmatory"
ANCHOR_RUN_NAME = "mujoco_v14_18_router_mechanism_anchors_20260830_r1"
ENVIRONMENTS = predecessor.ENVIRONMENTS
OPTIMIZER_SEEDS = predecessor.OPTIMIZER_SEEDS
ROUTER_STRENGTHS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ACTOR_GAINS = (1.0,)
BASELINE_ROUTER_STRENGTH = 0.5
PROFILE = "v14_17_anchor"
EPISODE_HORIZON = 1000
LEAKAGE_COST_MODE = "power_excess"
DISCOVERY_CELL = ("HalfCheetah-v5", 4196455150)
STRICT_IMPROVEMENT_TOLERANCE = 1e-12
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)

SELECTION_CONTRACT = {
    "unit": "environment_by_optimizer_seed_anchor",
    "candidate_scope": "one_global_router_strength_no_per_environment_tuning",
    "reward_safety": "zero_closed_loop_reward_violations_in_every_cell",
    "frequency_requirement": (
        "strict_frequency_violation_merit_reduction_vs_strength_0.5_in_every_cell"
    ),
    "nomination_order": (
        "maximize_minimum_relative_merit_reduction_then_median_then_mean_"
        "then_minimize_total_frequency_violations_then_smallest_strength"
    ),
    "discovery_disclosure": (
        "HalfCheetah-v5_seed_4196455150_was_inspected_before_this_grid_was_frozen"
    ),
}


def validate() -> None:
    if len(ENVIRONMENTS) != 3 or len(OPTIMIZER_SEEDS) != 3:
        raise RuntimeError("v14.18 requires the frozen 3 by 3 anchor registry")
    if len(set(ENVIRONMENTS)) != len(ENVIRONMENTS):
        raise RuntimeError("v14.18 environments must be unique")
    if len(set(OPTIMIZER_SEEDS)) != len(OPTIMIZER_SEEDS):
        raise RuntimeError("v14.18 optimizer seeds must be unique")
    if ROUTER_STRENGTHS[0] != BASELINE_ROUTER_STRENGTH:
        raise RuntimeError("v14.18 baseline strength must lead the frozen grid")
    if len(set(ROUTER_STRENGTHS)) != len(ROUTER_STRENGTHS):
        raise RuntimeError("v14.18 router strengths must be unique")
    if any(not 0.0 <= value <= 1.0 for value in ROUTER_STRENGTHS):
        raise RuntimeError("v14.18 router strengths must be in [0, 1]")
    if ACTOR_GAINS != (1.0,):
        raise RuntimeError("v14.18 must isolate routing from actor contraction")
    if DISCOVERY_CELL not in {
        (environment, seed)
        for environment in ENVIRONMENTS
        for seed in OPTIMIZER_SEEDS
    }:
        raise RuntimeError("v14.18 discovery cell must belong to the screen")


validate()
