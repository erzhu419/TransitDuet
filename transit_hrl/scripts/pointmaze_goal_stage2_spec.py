"""Frozen protocol for the corrected PointMaze ordinary-HRL stage-2 gate."""

from __future__ import annotations

from freq_hrl.experiments.pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_GOAL_PROTOCOL_VERSION,
    POINTMAZE_METHODS,
)


PROTOCOL = POINTMAZE_GOAL_PROTOCOL_VERSION
ALGORITHM_REVISION = "daacf09ac92ea5b794a516e729e023ad53797c25"
METHODS = tuple(POINTMAZE_METHODS)
ENV_ID = DEFAULT_ENV_ID
OPTIMIZER_SEEDS = (
    74007,
    74013,
    74037,
    74049,
    74059,
    74083,
    74101,
    74121,
)
ITERATIONS = 768
HORIZON = 300
CHECKPOINT_EVALUATION_INTERVAL = 96
REFERENCE_HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
UPPER_PERIOD_SECONDS = 0.25
MAXIMUM_SUBGOAL_DELTA = 0.75
RUNTIME_EXPECTATIONS = {
    "gymnasium": "1.2.0",
    "gymnasium_robotics": "1.4.2",
    "mujoco": "3.2.7",
    "pettingzoo": "1.26.1",
    "scipy": "1.13.1",
    "torch": "2.5.1+cu121",
}


def seed_roles(optimizer_seed: int) -> dict[str, tuple[int, ...]]:
    try:
        replicate = OPTIMIZER_SEEDS.index(int(optimizer_seed))
    except ValueError as exc:
        raise ValueError("optimizer seed is not registered") from exc
    base = 600_000 + replicate * 1_000
    return {
        "train": tuple(base + value for value in (11, 17, 23, 39)),
        "selection": tuple(
            base + value
            for value in (
                101,
                103,
                107,
                109,
                127,
                131,
                137,
                149,
                151,
                163,
                167,
                173,
                179,
                181,
                191,
                197,
            )
        ),
        "evaluation": tuple(
            base + value
            for value in (
                211,
                223,
                227,
                229,
                233,
                239,
                241,
                251,
                257,
                263,
                269,
                271,
                277,
                281,
                283,
                293,
            )
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
