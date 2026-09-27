"""Frozen two-root screen for 50-step same-budget timing-pair labels."""

from __future__ import annotations

from scripts import pointmaze_timing_pair_stage11_spec as stage11
from freq_hrl.experiments.pointmaze_timing_pair import WINDOWED_PROTOCOL_VERSION


EXPERIMENT_PROTOCOL = WINDOWED_PROTOCOL_VERSION
ALGORITHM_REVISION = "64fe6ab72b1c466f62150284393cc54c6175a796"
RUNNER_SCRIPT = stage11.RUNNER_SCRIPT
POLICY = stage11.POLICY
PREFLIGHT_ROOTS = stage11.PREFLIGHT_ROOTS
DEVELOPMENT_ROOTS = stage11.DEVELOPMENT_ROOTS
CREDIT_WINDOW_STEPS = 50


def roots(*, preflight: bool) -> tuple[int, ...]:
    return PREFLIGHT_ROOTS if preflight else DEVELOPMENT_ROOTS


def cell_options(root: int, *, preflight: bool) -> dict[str, object]:
    return {
        **stage11.cell_options(root, preflight=preflight),
        "credit_window_steps": CREDIT_WINDOW_STEPS,
    }
