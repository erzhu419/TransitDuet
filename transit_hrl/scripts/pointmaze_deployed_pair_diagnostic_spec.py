"""Stage-13 diagnostic on the two revealed Stage-12 roots."""

from __future__ import annotations

from pathlib import Path

from freq_hrl.experiments.pointmaze_deployed_pair_diagnostic import (
    PROTOCOL_VERSION,
    SOURCE_RUN,
)
from scripts import pointmaze_timing_pair_stage12_spec as stage12


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
RUNNER_SCRIPT = "scripts/run_pointmaze_deployed_pair_diagnostic.py"
POLICY = stage12.POLICY
CONTINUATION = "static_factual_schedule_after_one_bin_flip"


def roots(*, preflight: bool) -> tuple[int, ...]:
    return stage12.roots(preflight=preflight)


def source_result(root: int, *, preflight: bool) -> Path:
    if root not in roots(preflight=preflight):
        raise ValueError("Stage-13 root is not registered")
    phase = "preflight" if preflight else "development"
    return (
        Path(__file__).resolve().parents[1] / "results"
        / SOURCE_RUN.format(phase) / "cells" / POLICY
        / f"replicate_{root}" / "result.json"
    )
