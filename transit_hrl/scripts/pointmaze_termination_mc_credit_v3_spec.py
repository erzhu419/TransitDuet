"""Frozen two-root development screen for full-return termination credit."""

from __future__ import annotations

from scripts import pointmaze_termination_stochastic_diagnostic_spec as previous
from freq_hrl.experiments.pointmaze_learned_termination import (
    MC_CREDIT_PROTOCOL_VERSION,
)


EXPERIMENT_PROTOCOL = MC_CREDIT_PROTOCOL_VERSION
ALGORITHM_REVISION = "1fc6dc8ffff597362bfa02bbb962b5915f5a3609"
POLICY = previous.POLICY
PREFLIGHT_ROOTS = previous.PREFLIGHT_ROOTS
DEVELOPMENT_ROOTS = previous.DIAGNOSTIC_ROOTS
STOCHASTIC_REPETITIONS = previous.STOCHASTIC_REPETITIONS
GAE_LAMBDA = 1.0


def roots(*, preflight: bool) -> tuple[int, ...]:
    return PREFLIGHT_ROOTS if preflight else DEVELOPMENT_ROOTS


def cell_options(root: int, *, preflight: bool) -> dict[str, object]:
    if root not in roots(preflight=preflight):
        raise ValueError("MC-credit development root is not registered")
    return previous.cell_options(root, preflight=preflight)
