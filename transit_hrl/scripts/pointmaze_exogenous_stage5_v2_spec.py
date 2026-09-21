"""Frozen fresh-seed confirmation for the repaired Stage-5 substrate."""

from __future__ import annotations

from scripts import pointmaze_exogenous_stage5_spec as base


PROTOCOL = base.PROTOCOL
EXPERIMENT_PROTOCOL = "pointmaze_exogenous_control_stage5_v2_confirmation"
ALGORITHM_REVISION = "a4a64730a9542e690650d3a39856a5324ec51fcc"
EVIDENCE_STAGE = "confirmation"
METHODS = tuple(base.METHODS)
ENV_ID = base.ENV_ID
PREFLIGHT_OPTIMIZER_SEEDS = (174001,)
OPTIMIZER_SEEDS = (
    174007,
    174013,
    174031,
    174043,
    174067,
    174089,
    174113,
    174127,
)
ITERATIONS = base.ITERATIONS
HORIZON = base.HORIZON
CHECKPOINT_EVALUATION_INTERVAL = base.CHECKPOINT_EVALUATION_INTERVAL
CHECKPOINT_RANK_MODE = "success_then_return"
REFERENCE_HIDDEN_DIM = base.REFERENCE_HIDDEN_DIM
LEARNING_RATE = base.LEARNING_RATE
UPPER_PERIOD_SECONDS = base.UPPER_PERIOD_SECONDS
HISTORY_SECONDS = base.HISTORY_SECONDS
FAST_PERIOD_SECONDS = base.FAST_PERIOD_SECONDS
MAXIMUM_SUBGOAL_DELTA = base.MAXIMUM_SUBGOAL_DELTA
TARGET_SPEED = base.TARGET_SPEED
FORCE_RMS = base.FORCE_RMS
FORCE_PERIOD_SECONDS = base.FORCE_PERIOD_SECONDS
RUNTIME_EXPECTATIONS = dict(base.RUNTIME_EXPECTATIONS)
CLAIM_GATE = {
    "hrl_tracking_success_ci_lower": ">=0.50",
    "hrl_final_vs_untrained_tracking_success": "positive_ci",
    "hrl_final_vs_untrained_episode_return": "positive_ci",
    "hrl_vs_flat": "reported_not_gating",
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
        base_seed = 1_690_000
    else:
        try:
            replicate = OPTIMIZER_SEEDS.index(root)
        except ValueError as exc:
            raise ValueError("optimizer seed is not registered") from exc
        base_seed = 1_700_000 + replicate * 1_000
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

