"""Frozen protocol for the multiscale goal-control stage-1 campaign."""

from __future__ import annotations

from freq_hrl.domains.tracking import TRACKING_SCENARIOS
from freq_hrl.experiments.multiscale_tracking_validation import STAGE1_METHODS


PROTOCOL = "multiscale_goal_control_stage1_v1"
ALGORITHM_REVISION = "e83a17f45e330ab290e98b2e1b6c40c90b7941ae"
METHODS = tuple(STAGE1_METHODS)
SCENARIOS = tuple(TRACKING_SCENARIOS)
OPTIMIZER_SEEDS = (
    45007,
    45013,
    45053,
    45061,
    45077,
    45083,
    45119,
    45127,
)
ITERATIONS = 128
HORIZON = 256
CHECKPOINT_EVALUATION_INTERVAL = 8
REFERENCE_HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
TIME_SCALE = {
    "dt_seconds": 0.05,
    "upper_period_seconds": 0.8,
    "history_seconds": 3.2,
    "fast_period_seconds": 0.2,
}


def seed_roles(optimizer_seed: int) -> dict[str, tuple[int, ...]]:
    try:
        replicate = OPTIMIZER_SEEDS.index(int(optimizer_seed))
    except ValueError as exc:
        raise ValueError("optimizer seed is not registered") from exc
    base = 100_000 + replicate * 1_000
    return {
        "train": tuple(base + value for value in (11, 17, 23, 39)),
        "selection": tuple(base + value for value in (113, 127, 131, 149)),
        "evaluation": tuple(
            base + value
            for value in (211, 223, 227, 239, 251, 263, 269, 281)
        ),
    }


def cells(*, preflight: bool) -> list[tuple[str, str, int]]:
    optimizer_seeds = OPTIMIZER_SEEDS[:1] if preflight else OPTIMIZER_SEEDS
    return [
        (scenario, method, optimizer_seed)
        for optimizer_seed in optimizer_seeds
        for scenario in SCENARIOS
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
        "iterations": 2 if preflight else ITERATIONS,
        "horizon": 64 if preflight else HORIZON,
        "checkpoint_evaluation_interval": (
            1 if preflight else CHECKPOINT_EVALUATION_INTERVAL
        ),
        "reference_hidden_dim": REFERENCE_HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        **roles,
        **TIME_SCALE,
    }
