"""Frozen Stage-8C qualification of a compact causal plan-validity model."""

from __future__ import annotations

from freq_hrl.domains.mujoco import POINTMAZE_REGIME_SPEEDS
from freq_hrl.experiments.pointmaze_compact_plan_validity import (
    COMPACT_PLAN_VALIDITY_PREDICTORS,
    DEFAULT_RIDGE_ALPHA_GRID,
    POINTMAZE_COMPACT_PLAN_VALIDITY_POLICY,
    POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION,
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


PROTOCOL = POINTMAZE_COMPACT_PLAN_VALIDITY_PROTOCOL_VERSION
EXPERIMENT_PROTOCOL = "pointmaze_compact_plan_validity_stage8c_v1_development"
ALGORITHM_REVISION = "78e8493c6ba4399b3d68df4da4d353aaab03104e"
EVIDENCE_STAGE = "compact_plan_validity_predictor_qualification"
STAGE_LABEL = "stage8c"
RUNNER_SCRIPT = "scripts/run_pointmaze_compact_plan_validity_stage8c.py"
POLICY = POINTMAZE_COMPACT_PLAN_VALIDITY_POLICY
ENV_ID = DEFAULT_ENV_ID
PREFLIGHT_OPTIMIZER_SEEDS = (207001,)
OPTIMIZER_SEEDS = (
    207011,
    207023,
    207037,
    207049,
    207061,
    207073,
    207089,
    207101,
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
SELECTION_RATE = 0.25
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
    "evidence_role": "predictor_qualification_not_closed_loop_confirmation",
    "primary_endpoint": (
        "causal_validity_interactions_selected_utility_minus_"
        "current_compact_quadratic"
    ),
    "controller_learned": "positive_root_ci",
    "delayed_renewal_has_value": "positive_root_ci",
    "candidate_rank_is_positive": "positive_root_ci",
    "candidate_selected_utility_is_positive": "positive_root_ci",
    "candidate_utility_beats_current_only": "positive_root_ci",
    "candidate_rank_beats_current_only": "positive_root_ci",
    "candidate_rank_beats_generic_dynamic": "positive_root_ci",
    "statistical_unit": "optimizer_seed_root",
    "sequential_root_extension": "forbidden",
    "authorization_scope": "stage9_budgeted_trigger_development_only",
}

_TRAIN_OFFSETS = (11, 17, 23, 29, 41, 47, 59, 61)
_SELECTION_OFFSETS = (101, 103, 109, 127, 131, 137, 149, 151)
_BRANCH_FIT_OFFSETS = (201, 211, 223, 227, 233, 239, 251, 257)
_BRANCH_EVAL_OFFSETS = (
    301, 307, 311, 313, 317, 331, 337, 347,
    349, 353, 359, 367, 373, 379, 383, 389,
)


def seed_roles(optimizer_seed: int) -> dict[str, tuple[int, ...]]:
    root = int(optimizer_seed)
    if root in PREFLIGHT_OPTIMIZER_SEEDS:
        base_seed = 2_145_000
    else:
        try:
            replicate = OPTIMIZER_SEEDS.index(root)
        except ValueError as exc:
            raise ValueError("optimizer seed is not registered") from exc
        base_seed = 2_150_000 + replicate * 1_000
    return {
        "train": tuple(base_seed + value for value in _TRAIN_OFFSETS),
        "selection": tuple(
            base_seed + value for value in _SELECTION_OFFSETS
        ),
        "branch_fit": tuple(
            base_seed + value for value in _BRANCH_FIT_OFFSETS
        ),
        "branch_eval": tuple(
            base_seed + value for value in _BRANCH_EVAL_OFFSETS
        ),
    }


def cells(*, preflight: bool) -> list[tuple[str, int]]:
    roots = PREFLIGHT_OPTIMIZER_SEEDS if preflight else OPTIMIZER_SEEDS
    return [(POLICY, root) for root in roots]


def cell_options(
    optimizer_seed: int,
    *,
    preflight: bool,
) -> dict[str, object]:
    roles = seed_roles(optimizer_seed)
    if preflight:
        roles = {
            "train": roles["train"][:1],
            "selection": roles["selection"][:1],
            "branch_fit": roles["branch_fit"][:2],
            "branch_eval": roles["branch_eval"][:2],
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
        "selection_rate": SELECTION_RATE,
        "regime_dwell_seconds": REGIME_DWELL_SECONDS,
        "target_speed_modes": TARGET_SPEED_MODES,
        "force_pulse_amplitude": FORCE_PULSE_AMPLITUDE,
        "force_pulse_duration_seconds": FORCE_PULSE_DURATION_SECONDS,
        "force_pulse_gap_seconds": FORCE_PULSE_GAP_SECONDS,
        "distractor_amplitude": DISTRACTOR_AMPLITUDE,
        "distractor_dwell_seconds": DISTRACTOR_DWELL_SECONDS,
        "predictors": COMPACT_PLAN_VALIDITY_PREDICTORS,
        **roles,
    }
