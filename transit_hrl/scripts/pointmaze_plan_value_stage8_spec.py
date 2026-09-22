"""Frozen Stage-8 qualification of plan value before learned belief work."""

from __future__ import annotations

from freq_hrl.experiments.pointmaze_goal_validation import DEFAULT_ENV_ID
from freq_hrl.experiments.pointmaze_plan_value_qualification import (
    DEFAULT_DISTRACTOR_AMPLITUDE,
    DEFAULT_DISTRACTOR_DWELL_SECONDS,
    DEFAULT_EVENT_WINDOW_SECONDS,
    DEFAULT_FORCE_PULSE_AMPLITUDE,
    DEFAULT_FORCE_PULSE_DURATION_SECONDS,
    DEFAULT_FORCE_PULSE_GAP_SECONDS,
    DEFAULT_REGIME_DWELL_SECONDS,
    DEFAULT_WAYPOINT_PERTURBATION,
    POINTMAZE_PLAN_VALUE_METHODS,
    POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION,
)
from freq_hrl.domains.mujoco import POINTMAZE_REGIME_SPEEDS


PROTOCOL = POINTMAZE_PLAN_VALUE_PROTOCOL_VERSION
EXPERIMENT_PROTOCOL = "pointmaze_plan_value_stage8_v1_development"
ALGORITHM_REVISION = "92735a4050eb175cc76d9a2418401e805fc8ca1b"
EVIDENCE_STAGE = "task_qualification_development"
STAGE_LABEL = "stage8"
RUNNER_SCRIPT = "scripts/run_pointmaze_plan_value_stage8.py"
METHODS = tuple(POINTMAZE_PLAN_VALUE_METHODS)
ENV_ID = DEFAULT_ENV_ID
PREFLIGHT_OPTIMIZER_SEEDS = (204901,)
OPTIMIZER_SEEDS = (
    204907,
    204913,
    204931,
    204943,
    204967,
    204989,
    205013,
    205027,
)
ITERATIONS = 384
HORIZON = 1200
PREFLIGHT_HORIZON = 240
CHECKPOINT_EVALUATION_INTERVAL = 8
REFERENCE_HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
UPPER_PERIOD_SECONDS = 0.50
HISTORY_SECONDS = 0.64
FAST_PERIOD_SECONDS = 0.04
MAXIMUM_SUBGOAL_DELTA = 0.75
WAYPOINT_PERTURBATION = DEFAULT_WAYPOINT_PERTURBATION
EVENT_WINDOW_SECONDS = DEFAULT_EVENT_WINDOW_SECONDS
REGIME_DWELL_SECONDS = DEFAULT_REGIME_DWELL_SECONDS
TARGET_SPEED_MODES = POINTMAZE_REGIME_SPEEDS
FORCE_PULSE_AMPLITUDE = DEFAULT_FORCE_PULSE_AMPLITUDE
FORCE_PULSE_DURATION_SECONDS = DEFAULT_FORCE_PULSE_DURATION_SECONDS
FORCE_PULSE_GAP_SECONDS = DEFAULT_FORCE_PULSE_GAP_SECONDS
DISTRACTOR_AMPLITUDE = DEFAULT_DISTRACTOR_AMPLITUDE
DISTRACTOR_DWELL_SECONDS = DEFAULT_DISTRACTOR_DWELL_SECONDS
RUNTIME_EXPECTATIONS = {
    "gymnasium": "1.2.0",
    "gymnasium_robotics": "1.4.2",
    "mujoco": "3.2.7",
    "pettingzoo": "1.26.1",
    "scipy": "1.13.1",
    "torch": "2.5.1+cu121",
}
CLAIM_GATE = {
    "evidence_role": "task_qualification_not_algorithm_confirmation",
    "primary_endpoint": "tracking_squared_error_integral",
    "history_controller_learned": "positive_root_paired_ci",
    "plan_refresh_has_value": "positive_root_paired_ci",
    "plan_content_has_value": "positive_root_paired_ci",
    "current_regime_information_has_value": "positive_root_paired_ci",
    "same_budget_event_timing_has_value": "positive_root_paired_ci",
    "quarter_second_delay_has_cost": "positive_root_paired_ci",
    "causal_observation_precedes_quarter_second_cost": True,
    "statistical_unit": "optimizer_seed_root",
    "sequential_root_extension": "forbidden",
}

_TRAIN_OFFSETS = (11, 17, 23, 39, 41, 53, 67, 71)
_SELECTION_OFFSETS = (101, 103, 107, 109, 127, 131, 137, 149)
_EVALUATION_OFFSETS = (
    211, 223, 227, 229, 233, 239, 241, 251,
    257, 263, 269, 271, 277, 281, 283, 293,
)


def seed_roles(optimizer_seed: int) -> dict[str, tuple[int, ...]]:
    root = int(optimizer_seed)
    if root in PREFLIGHT_OPTIMIZER_SEEDS:
        base_seed = 2_090_000
    else:
        try:
            replicate = OPTIMIZER_SEEDS.index(root)
        except ValueError as exc:
            raise ValueError("optimizer seed is not registered") from exc
        base_seed = 2_100_000 + replicate * 1_000
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
        "horizon": PREFLIGHT_HORIZON if preflight else HORIZON,
        "checkpoint_evaluation_interval": (
            1 if preflight else CHECKPOINT_EVALUATION_INTERVAL
        ),
        "reference_hidden_dim": REFERENCE_HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "upper_period_seconds": UPPER_PERIOD_SECONDS,
        "history_seconds": HISTORY_SECONDS,
        "fast_period_seconds": FAST_PERIOD_SECONDS,
        "maximum_subgoal_delta": MAXIMUM_SUBGOAL_DELTA,
        "waypoint_perturbation": WAYPOINT_PERTURBATION,
        "event_window_seconds": EVENT_WINDOW_SECONDS,
        "regime_dwell_seconds": REGIME_DWELL_SECONDS,
        "target_speed_modes": TARGET_SPEED_MODES,
        "force_pulse_amplitude": FORCE_PULSE_AMPLITUDE,
        "force_pulse_duration_seconds": FORCE_PULSE_DURATION_SECONDS,
        "force_pulse_gap_seconds": FORCE_PULSE_GAP_SECONDS,
        "distractor_amplitude": DISTRACTOR_AMPLITUDE,
        "distractor_dwell_seconds": DISTRACTOR_DWELL_SECONDS,
        **roles,
    }
