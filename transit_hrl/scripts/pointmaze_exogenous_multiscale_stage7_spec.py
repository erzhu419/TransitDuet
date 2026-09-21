"""Frozen fresh-root confirmation of the PointMaze multiscale factorial."""

from __future__ import annotations

from freq_hrl.experiments.pointmaze_exogenous_routing_attribution import (
    POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
)
from scripts import pointmaze_exogenous_routing_stage6_spec as stage6


PROTOCOL = POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION
EXPERIMENT_PROTOCOL = "pointmaze_exogenous_multiscale_stage7_v1_confirmation"
ALGORITHM_REVISION = "4ed9e8e131235f0f99844e8ea8bfeb737a68276f"
EVIDENCE_STAGE = "confirmation"
STAGE_LABEL = "stage7"
RUNNER_SCRIPT = stage6.RUNNER_SCRIPT
METHODS = (
    "flat_exogenous_history",
    "flat_exogenous_multiscale_all",
    "hrl_exogenous_history",
    "hrl_exogenous_multiscale_all",
)
ENV_ID = stage6.ENV_ID
PREFLIGHT_OPTIMIZER_SEEDS = (194001,)
OPTIMIZER_SEEDS = (
    194007,
    194013,
    194031,
    194043,
    194067,
    194089,
    194113,
    194127,
    194141,
    194159,
    194173,
    194191,
    194207,
    194221,
    194239,
    194257,
)
ITERATIONS = stage6.ITERATIONS
HORIZON = stage6.HORIZON
CHECKPOINT_EVALUATION_INTERVAL = stage6.CHECKPOINT_EVALUATION_INTERVAL
CHECKPOINT_RANK_MODE = stage6.CHECKPOINT_RANK_MODE
REFERENCE_HIDDEN_DIM = stage6.REFERENCE_HIDDEN_DIM
LEARNING_RATE = stage6.LEARNING_RATE
UPPER_PERIOD_SECONDS = stage6.UPPER_PERIOD_SECONDS
HISTORY_SECONDS = stage6.HISTORY_SECONDS
FAST_PERIOD_SECONDS = stage6.FAST_PERIOD_SECONDS
MAXIMUM_SUBGOAL_DELTA = stage6.MAXIMUM_SUBGOAL_DELTA
TARGET_SPEED = stage6.TARGET_SPEED
FORCE_RMS = stage6.FORCE_RMS
FORCE_PERIOD_SECONDS = stage6.FORCE_PERIOD_SECONDS
RUNTIME_EXPECTATIONS = dict(stage6.RUNTIME_EXPECTATIONS)
CLAIM_GATE = {
    "primary_endpoint": "tracking_success_rate",
    "hrl_multiscale_absolute_success_ci_lower": ">=0.50",
    "hrl_multiscale_final_vs_untrained_success": "positive_ci",
    "hrl_multiscale_final_vs_untrained_return": "positive_ci",
    "hrl_multiscale_vs_hrl_history_success": "positive_ci",
    "hrl_multiscale_vs_flat_multiscale_success": "positive_ci",
    "hierarchy_x_multiscale_success_interaction": "positive_ci",
    "statistical_unit": "optimizer_seed_root",
    "sequential_root_extension": "forbidden",
}

_TRAIN_OFFSETS = (11, 17, 23, 39, 41, 53, 67, 71)
_SELECTION_OFFSETS = (
    101, 103, 107, 109, 127, 131, 137, 149,
    151, 163, 167, 173, 179, 181, 191, 197,
)
_EVALUATION_OFFSETS = (
    211, 223, 227, 229, 233, 239, 241, 251,
    257, 263, 269, 271, 277, 281, 283, 293,
)


def seed_roles(optimizer_seed: int) -> dict[str, tuple[int, ...]]:
    root = int(optimizer_seed)
    if root in PREFLIGHT_OPTIMIZER_SEEDS:
        base_seed = 1_890_000
    else:
        try:
            replicate = OPTIMIZER_SEEDS.index(root)
        except ValueError as exc:
            raise ValueError("optimizer seed is not registered") from exc
        base_seed = 1_900_000 + replicate * 1_000
    return {
        "train": tuple(base_seed + value for value in _TRAIN_OFFSETS),
        "selection": tuple(base_seed + value for value in _SELECTION_OFFSETS),
        "evaluation": tuple(base_seed + value for value in _EVALUATION_OFFSETS),
    }


def cells(*, preflight: bool) -> list[tuple[str, int]]:
    roots = PREFLIGHT_OPTIMIZER_SEEDS if preflight else OPTIMIZER_SEEDS
    return [(method, root) for root in roots for method in METHODS]


def cell_options(
    optimizer_seed: int,
    *,
    preflight: bool,
) -> dict[str, object]:
    roles = seed_roles(optimizer_seed)
    if preflight:
        roles = {name: values[:1] for name, values in roles.items()}
    return {
        "env_id": ENV_ID,
        "iterations": 2 if preflight else ITERATIONS,
        "horizon": 64 if preflight else HORIZON,
        "checkpoint_evaluation_interval": (
            1 if preflight else CHECKPOINT_EVALUATION_INTERVAL
        ),
        "checkpoint_rank_mode": CHECKPOINT_RANK_MODE,
        "reference_hidden_dim": REFERENCE_HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "upper_period_seconds": UPPER_PERIOD_SECONDS,
        "history_seconds": HISTORY_SECONDS,
        "fast_period_seconds": FAST_PERIOD_SECONDS,
        "maximum_subgoal_delta": MAXIMUM_SUBGOAL_DELTA,
        "target_speed": TARGET_SPEED,
        "force_rms": FORCE_RMS,
        "force_period_seconds": FORCE_PERIOD_SECONDS,
        **roles,
    }
