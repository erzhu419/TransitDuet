"""Frozen two-root diagnostic of stochastic versus deterministic termination."""

from __future__ import annotations

from scripts import pointmaze_learned_termination_stage10_spec as stage10


EXPERIMENT_PROTOCOL = "pointmaze_termination_stochastic_v2_diagnostic"
ALGORITHM_REVISION = "152ca283728352e02ea1017df33cd678fb3bd5e7"
RUNNER_SCRIPT = stage10.RUNNER_SCRIPT
POLICY = stage10.POLICY
PREFLIGHT_ROOTS = stage10.PREFLIGHT_OPTIMIZER_SEEDS
DIAGNOSTIC_ROOTS = (209011, 209061)
STOCHASTIC_REPETITIONS = 4


def roots(*, preflight: bool) -> tuple[int, ...]:
    return PREFLIGHT_ROOTS if preflight else DIAGNOSTIC_ROOTS


def cell_options(root: int, *, preflight: bool) -> dict[str, object]:
    if root not in roots(preflight=preflight):
        raise ValueError("stochastic diagnostic root is not registered")
    return stage10.cell_options(root, preflight=preflight)
