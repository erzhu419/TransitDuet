"""Frozen Stage-9 development matrix for causal budgeted replanning."""

from __future__ import annotations

from freq_hrl.domains.mujoco import POINTMAZE_REGIME_SPEEDS
from freq_hrl.experiments.pointmaze_budgeted_trigger import (
    POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION,
    TRIGGER_MODES,
)
from freq_hrl.experiments.pointmaze_compact_plan_validity import (
    DEFAULT_RIDGE_ALPHA_GRID,
)
from freq_hrl.experiments.pointmaze_goal_validation import DEFAULT_ENV_ID
from freq_hrl.experiments.pointmaze_plan_value_qualification import (
    DEFAULT_DISTRACTOR_AMPLITUDE,
    DEFAULT_DISTRACTOR_DWELL_SECONDS,
    DEFAULT_FORCE_PULSE_AMPLITUDE,
    DEFAULT_FORCE_PULSE_DURATION_SECONDS,
    DEFAULT_FORCE_PULSE_GAP_SECONDS,
    DEFAULT_REGIME_DWELL_SECONDS,
)


PROTOCOL = POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION
EXPERIMENT_PROTOCOL = "pointmaze_budgeted_trigger_stage9_v1_development"
ALGORITHM_REVISION = "91b3e6919bcffe3a75d4942b0df31ae8478d69f6"
RUNNER_SCRIPT = "scripts/run_pointmaze_budgeted_trigger_stage9.py"
POLICY = "hrl_regime_history"
ENV_ID = DEFAULT_ENV_ID
PREFLIGHT_OPTIMIZER_SEEDS = (208001,)
OPTIMIZER_SEEDS = (
    208011, 208023, 208037, 208049,
    208061, 208073, 208089, 208101,
)
ITERATIONS = 384
HORIZON = 1200
PREFLIGHT_HORIZON = 300
CHECKPOINT_EVALUATION_INTERVAL = 8
REFERENCE_HIDDEN_DIM = 128
LEARNING_RATE = 3e-4
UPPER_PERIOD_SECONDS = 0.50
HISTORY_SECONDS = 0.64
FAST_PERIOD_SECONDS = 0.04
MAXIMUM_SUBGOAL_DELTA = 0.75
BRANCH_WINDOW_SECONDS = 0.50
MAX_EVENTS_PER_CLASS = 4
RIDGE_ALPHA_GRID = DEFAULT_RIDGE_ALPHA_GRID
THRESHOLD_QUANTILE = 0.75
MAX_OFFSET_STEPS = 25
CHECK_STRIDE_STEPS = 5
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
    "primary_endpoint": "candidate_minus_fixed_episode_tracking_ise",
    "controller_learned": "positive_root_ci",
    "candidate_beats_fixed_ise": "positive_root_ci",
    "candidate_beats_random_ise": "positive_root_ci",
    "candidate_beats_current_only_ise": "positive_root_ci",
    "candidate_return_beats_fixed": "positive_root_ci",
    "statistical_unit": "optimizer_seed_root",
    "sequential_root_extension": "forbidden",
    "authorization_scope": "stage9_development_result_only",
}

_TRAIN_OFFSETS = (11, 17, 23, 29, 41, 47, 59, 61)
_SELECTION_OFFSETS = (101, 103, 109, 127, 131, 137, 149, 151)
_BRANCH_FIT_OFFSETS = (201, 211, 223, 227, 233, 239, 251, 257)
_TRIGGER_EVAL_OFFSETS = (
    301, 307, 311, 313, 317, 331, 337, 347,
    349, 353, 359, 367, 373, 379, 383, 389,
)


def seed_roles(optimizer_seed: int) -> dict[str, tuple[int, ...]]:
    root = int(optimizer_seed)
    if root in PREFLIGHT_OPTIMIZER_SEEDS:
        base = 2_159_000
    else:
        try:
            replicate = OPTIMIZER_SEEDS.index(root)
        except ValueError as exc:
            raise ValueError("Stage-9 optimizer seed is not registered") from exc
        base = 2_160_000 + replicate * 1_000
    return {
        "train": tuple(base + x for x in _TRAIN_OFFSETS),
        "selection": tuple(base + x for x in _SELECTION_OFFSETS),
        "branch_fit": tuple(base + x for x in _BRANCH_FIT_OFFSETS),
        "trigger_eval": tuple(base + x for x in _TRIGGER_EVAL_OFFSETS),
    }


def cells(*, preflight: bool) -> list[tuple[str, int]]:
    roots = PREFLIGHT_OPTIMIZER_SEEDS if preflight else OPTIMIZER_SEEDS
    return [(POLICY, root) for root in roots]


def cell_options(optimizer_seed: int, *, preflight: bool) -> dict[str, object]:
    roles = seed_roles(optimizer_seed)
    if preflight:
        roles = {
            "train": roles["train"][:1],
            "selection": roles["selection"][:1],
            "branch_fit": roles["branch_fit"][:2],
            "trigger_eval": roles["trigger_eval"][:2],
        }
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
        "branch_window_seconds": BRANCH_WINDOW_SECONDS,
        "max_events_per_class": 1 if preflight else MAX_EVENTS_PER_CLASS,
        "ridge_alpha_grid": RIDGE_ALPHA_GRID,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "max_offset_steps": MAX_OFFSET_STEPS,
        "check_stride_steps": CHECK_STRIDE_STEPS,
        "regime_dwell_seconds": REGIME_DWELL_SECONDS,
        "target_speed_modes": TARGET_SPEED_MODES,
        "force_pulse_amplitude": FORCE_PULSE_AMPLITUDE,
        "force_pulse_duration_seconds": FORCE_PULSE_DURATION_SECONDS,
        "force_pulse_gap_seconds": FORCE_PULSE_GAP_SECONDS,
        "distractor_amplitude": DISTRACTOR_AMPLITUDE,
        "distractor_dwell_seconds": DISTRACTOR_DWELL_SECONDS,
        "methods": TRIGGER_MODES,
        **roles,
    }
