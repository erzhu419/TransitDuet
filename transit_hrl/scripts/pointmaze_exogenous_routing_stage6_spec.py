"""Frozen PointMaze frequency-routing attribution protocol."""

from __future__ import annotations

from freq_hrl.experiments.pointmaze_exogenous_routing_attribution import (
    POINTMAZE_EXOGENOUS_ROUTING_METHODS,
    POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
)
from scripts import pointmaze_exogenous_stage5_v2_spec as stage5


PROTOCOL = POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION
EXPERIMENT_PROTOCOL = "pointmaze_exogenous_frequency_routing_stage6_v1"
ALGORITHM_REVISION = "4ed9e8e131235f0f99844e8ea8bfeb737a68276f"
EVIDENCE_STAGE = "development"
STAGE_LABEL = "stage6"
RUNNER_SCRIPT = "scripts/run_pointmaze_exogenous_routing_stage6.py"
METHODS = tuple(POINTMAZE_EXOGENOUS_ROUTING_METHODS)
ENV_ID = stage5.ENV_ID
PREFLIGHT_OPTIMIZER_SEEDS = (184001,)
OPTIMIZER_SEEDS = (
    184007,
    184013,
    184031,
    184043,
    184067,
    184089,
    184113,
    184127,
)
ITERATIONS = stage5.ITERATIONS
HORIZON = stage5.HORIZON
CHECKPOINT_EVALUATION_INTERVAL = stage5.CHECKPOINT_EVALUATION_INTERVAL
CHECKPOINT_RANK_MODE = stage5.CHECKPOINT_RANK_MODE
REFERENCE_HIDDEN_DIM = stage5.REFERENCE_HIDDEN_DIM
LEARNING_RATE = stage5.LEARNING_RATE
UPPER_PERIOD_SECONDS = stage5.UPPER_PERIOD_SECONDS
HISTORY_SECONDS = stage5.HISTORY_SECONDS
FAST_PERIOD_SECONDS = stage5.FAST_PERIOD_SECONDS
MAXIMUM_SUBGOAL_DELTA = stage5.MAXIMUM_SUBGOAL_DELTA
TARGET_SPEED = stage5.TARGET_SPEED
FORCE_RMS = stage5.FORCE_RMS
FORCE_PERIOD_SECONDS = stage5.FORCE_PERIOD_SECONDS
RUNTIME_EXPECTATIONS = dict(stage5.RUNTIME_EXPECTATIONS)
CLAIM_GATE = {
    "primary_endpoint": "tracking_success_rate",
    "routed_absolute_success_ci_lower": ">=0.50",
    "routed_final_vs_untrained_success": "positive_ci",
    "routed_final_vs_untrained_return": "positive_ci",
    "routed_vs_history_success": "positive_ci",
    "routed_vs_causal_filter_success": "positive_ci",
    "routed_vs_all_band_success": "positive_ci",
    "routed_vs_swapped_success": "positive_ci",
    "hierarchy_x_multiscale_success_interaction": "positive_ci",
    "statistical_unit": "optimizer_seed_root",
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
        base_seed = 1_790_000
    else:
        try:
            replicate = OPTIMIZER_SEEDS.index(root)
        except ValueError as exc:
            raise ValueError("optimizer seed is not registered") from exc
        base_seed = 1_800_000 + replicate * 1_000
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
