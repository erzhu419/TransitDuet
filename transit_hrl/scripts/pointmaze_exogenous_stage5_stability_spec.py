"""Frozen development screen for Stage-5 optimization stability."""

from __future__ import annotations

from scripts import pointmaze_exogenous_stage5_spec as base


PROTOCOL = "pointmaze_exogenous_stage5_optimizer_stability_screen_v1"
BASE_PROTOCOL = base.PROTOCOL
ALGORITHM_REVISION = "a4a64730a9542e690650d3a39856a5324ec51fcc"
METHOD = "hrl_exogenous_history"
ARMS = (
    "v1_control",
    "dense_rank",
    "more_rollouts",
    "dense_rank_more_rollouts",
)
ARM_CONFIG = {
    "v1_control": {
        "checkpoint_rank_mode": "success_then_return",
        "train_seed_count": 4,
    },
    "dense_rank": {
        "checkpoint_rank_mode": "return_then_success",
        "train_seed_count": 4,
    },
    "more_rollouts": {
        "checkpoint_rank_mode": "success_then_return",
        "train_seed_count": 8,
    },
    "dense_rank_more_rollouts": {
        "checkpoint_rank_mode": "return_then_success",
        "train_seed_count": 8,
    },
}
OPTIMIZER_SEEDS = (134113, 134127)
PREFLIGHT_OPTIMIZER_SEED = 164001
ITERATIONS = 768
MIN_PAIRED_ROOT_SUCCESS_GAIN = 0.05
RUNTIME_EXPECTATIONS = dict(base.RUNTIME_EXPECTATIONS)

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
    if root == PREFLIGHT_OPTIMIZER_SEED:
        base_seed = 1_590_000
    else:
        try:
            replicate = OPTIMIZER_SEEDS.index(root)
        except ValueError as exc:
            raise ValueError("optimizer seed is not registered") from exc
        base_seed = 1_600_000 + replicate * 1_000
    return {
        "train": tuple(base_seed + value for value in _TRAIN_OFFSETS),
        "selection": tuple(base_seed + value for value in _SELECTION_OFFSETS),
        "evaluation": tuple(base_seed + value for value in _EVALUATION_OFFSETS),
    }


def cells(*, preflight: bool) -> list[tuple[str, int]]:
    roots = (PREFLIGHT_OPTIMIZER_SEED,) if preflight else OPTIMIZER_SEEDS
    return [(arm, root) for root in roots for arm in ARMS]


def cell_options(
    arm: str,
    optimizer_seed: int,
    *,
    preflight: bool,
) -> dict[str, object]:
    if arm not in ARM_CONFIG:
        raise ValueError(f"unknown stability arm: {arm}")
    config = ARM_CONFIG[arm]
    roles = seed_roles(optimizer_seed)
    train_count = int(config["train_seed_count"])
    return {
        "env_id": base.ENV_ID,
        "iterations": 2 if preflight else ITERATIONS,
        "horizon": 64 if preflight else base.HORIZON,
        "checkpoint_evaluation_interval": (
            1 if preflight else base.CHECKPOINT_EVALUATION_INTERVAL
        ),
        "checkpoint_rank_mode": str(config["checkpoint_rank_mode"]),
        "reference_hidden_dim": base.REFERENCE_HIDDEN_DIM,
        "learning_rate": base.LEARNING_RATE,
        "upper_period_seconds": base.UPPER_PERIOD_SECONDS,
        "history_seconds": base.HISTORY_SECONDS,
        "fast_period_seconds": base.FAST_PERIOD_SECONDS,
        "maximum_subgoal_delta": base.MAXIMUM_SUBGOAL_DELTA,
        "target_speed": base.TARGET_SPEED,
        "force_rms": base.FORCE_RMS,
        "force_period_seconds": base.FORCE_PERIOD_SECONDS,
        "train": roles["train"][:train_count],
        "selection": roles["selection"][:1] if preflight else roles["selection"],
        "evaluation": roles["evaluation"][:1] if preflight else roles["evaluation"],
    }

