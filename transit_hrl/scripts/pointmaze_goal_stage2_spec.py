"""Frozen protocol for the PointMaze ordinary-HRL stage-2 gate."""

from __future__ import annotations

from freq_hrl.experiments.pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_METHODS,
)


PROTOCOL = "pointmaze_goal_control_stage2_v1"
ALGORITHM_REVISION = "d8fcbc8cb03b76e023063f061315684a2a6500be"
METHODS = tuple(POINTMAZE_METHODS)
ENV_ID = DEFAULT_ENV_ID
OPTIMIZER_SEEDS = (
    54007,
    54013,
    54037,
    54049,
    54059,
    54083,
    54101,
    54121,
)
ITERATIONS = 768
HORIZON = 300
CHECKPOINT_EVALUATION_INTERVAL = 16
REFERENCE_HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
UPPER_PERIOD_SECONDS = 0.25
MAXIMUM_SUBGOAL_DELTA = 0.75


def seed_roles(optimizer_seed: int) -> dict[str, tuple[int, ...]]:
    try:
        replicate = OPTIMIZER_SEEDS.index(int(optimizer_seed))
    except ValueError as exc:
        raise ValueError("optimizer seed is not registered") from exc
    base = 200_000 + replicate * 1_000
    return {
        "train": tuple(base + value for value in (11, 17, 23, 39)),
        "selection": tuple(base + value for value in (113, 127, 131, 149)),
        "evaluation": tuple(
            base + value
            for value in (211, 223, 227, 239, 251, 263, 269, 281)
        ),
    }


def cells(*, preflight: bool) -> list[tuple[str, int]]:
    optimizer_seeds = OPTIMIZER_SEEDS[:1] if preflight else OPTIMIZER_SEEDS
    return [
        (method, optimizer_seed)
        for optimizer_seed in optimizer_seeds
        for method in METHODS
    ]


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
        "reference_hidden_dim": REFERENCE_HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "upper_period_seconds": UPPER_PERIOD_SECONDS,
        "maximum_subgoal_delta": MAXIMUM_SUBGOAL_DELTA,
        **roles,
    }
