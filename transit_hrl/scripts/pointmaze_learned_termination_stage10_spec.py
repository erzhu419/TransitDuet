"""Frozen development comparison with an on-policy learned termination arm."""

from __future__ import annotations

from scripts import pointmaze_budgeted_trigger_stage9_spec as stage9
from freq_hrl.experiments.pointmaze_learned_termination import (
    PROTOCOL_VERSION,
)


EXPERIMENT_PROTOCOL = "pointmaze_learned_termination_stage10_v1_development"
PROTOCOL = PROTOCOL_VERSION
ALGORITHM_REVISION = "efaba47e42e40684f9d11db8be951ea0932049d8"
RUNNER_SCRIPT = "scripts/run_pointmaze_learned_termination_stage10.py"
POLICY = stage9.POLICY
PREFLIGHT_OPTIMIZER_SEEDS = stage9.PREFLIGHT_OPTIMIZER_SEEDS
OPTIMIZER_SEEDS = stage9.CONFIRMATION_OPTIMIZER_SEEDS
TERMINATION_ITERATIONS = 18
PREFLIGHT_TERMINATION_ITERATIONS = 2
TERMINATION_HIDDEN_DIM = 64
TERMINATION_LEARNING_RATE = 3e-4
RUNTIME_EXPECTATIONS = stage9.RUNTIME_EXPECTATIONS
CLAIM_GATE = {
    "primary_endpoint": "candidate_minus_learned_termination_episode_tracking_ise",
    "baseline_learns_vs_fixed": "positive_root_ci",
    "candidate_beats_learned_termination": "positive_root_ci",
    "statistical_unit": "optimizer_seed_root",
    "sequential_root_extension": "forbidden",
    "authorization_scope": "stage10_development_result_only",
}


def cells(*, preflight: bool) -> list[tuple[str, int]]:
    return [(POLICY, root) for root in (
        PREFLIGHT_OPTIMIZER_SEEDS if preflight else OPTIMIZER_SEEDS
    )]


def cell_options(root: int, *, preflight: bool) -> dict[str, object]:
    if root not in (
        PREFLIGHT_OPTIMIZER_SEEDS if preflight else OPTIMIZER_SEEDS
    ):
        raise ValueError("Stage-10 optimizer root is not registered")
    return {
        **stage9.cell_options(root, preflight=preflight),
        "termination_iterations": (
            PREFLIGHT_TERMINATION_ITERATIONS
            if preflight else TERMINATION_ITERATIONS
        ),
        "termination_hidden_dim": TERMINATION_HIDDEN_DIM,
        "termination_learning_rate": TERMINATION_LEARNING_RATE,
    }
