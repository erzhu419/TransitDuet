"""Frozen two-root development screen for same-budget timing-pair labels."""

from __future__ import annotations

from scripts import pointmaze_budgeted_trigger_stage9_spec as stage9
from freq_hrl.experiments.pointmaze_timing_pair import PROTOCOL_VERSION


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
ALGORITHM_REVISION = "5c2ab7af7201b8b7f98f77c72bd50dd88472ce01"
RUNNER_SCRIPT = "scripts/run_pointmaze_timing_pair_stage11.py"
POLICY = stage9.POLICY
PREFLIGHT_ROOTS = stage9.PREFLIGHT_OPTIMIZER_SEEDS
DEVELOPMENT_ROOTS = (209011, 209061)
PREFLIGHT_PAIRS_PER_SEED = 2
DEVELOPMENT_PAIRS_PER_SEED = 12


def roots(*, preflight: bool) -> tuple[int, ...]:
    return PREFLIGHT_ROOTS if preflight else DEVELOPMENT_ROOTS


def cell_options(root: int, *, preflight: bool) -> dict[str, object]:
    if root not in roots(preflight=preflight):
        raise ValueError("Stage-11 optimizer root is not registered")
    return {
        **stage9.cell_options(root, preflight=preflight),
        "pairs_per_seed": (
            PREFLIGHT_PAIRS_PER_SEED if preflight
            else DEVELOPMENT_PAIRS_PER_SEED
        ),
    }
