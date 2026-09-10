"""Frozen development screen for causal upper projection targets."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v23_causal_upper_projection_target_development_v1"
)
EVIDENCE_ROLE = "fresh_root_causal_upper_target_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v23_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = (
    "72643f24204543a3655dde395717bfdbb60ffd2a"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v23_causal_upper_projection_target_training"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from NumPy Generator(230091), checked against integer
# literals in earlier MuJoCo scripts, and frozen before v23 execution.
OPTIMIZER_SEEDS = (2296835635, 2641275687, 170362010, 2341363371)
TRAIN_SEEDS = (4147697775, 2071720330, 829772164, 3132020711)
SELECTION_SEEDS = (557539860, 1302833233, 139840218, 2964503741)
EVALUATION_SEEDS = (
    1425796162,
    2580603057,
    3802360576,
    3584493532,
    615200512,
    685602160,
    439012508,
    1601161006,
)

TERMINAL_RESERVE_ZERO = "terminal_reserve_consistency_000"
MACRO_MEAN_UNIFORM_010 = "terminal_reserve_macro_mean_uniform_010"
LOWER_ONLY_DECISION_TIME_010 = "terminal_reserve_lower_only_decision_time_010"
DECISION_TIME_UNIFORM_010 = "terminal_reserve_decision_time_uniform_010"
PRIMARY_PROJECTED_BASELINE = TERMINAL_RESERVE_ZERO
PRIMARY_OLD_CONTROL = MACRO_MEAN_UNIFORM_010
CANDIDATES = (LOWER_ONLY_DECISION_TIME_010, DECISION_TIME_UNIFORM_010)


def _arm(
    *,
    role: str,
    upper_coef: float,
    lower_coef: float,
    target_aggregation: str,
    training_schedule: str,
    warmup_fraction: float,
    ramp_fraction: float,
) -> dict[str, object]:
    return {
        "method": "freq_hrl",
        "arm_role": str(role),
        "terminal_reserve_context": True,
        "terminal_reserve_projection": True,
        "upper_projection_consistency_coef": float(upper_coef),
        "lower_projection_consistency_coef": float(lower_coef),
        "upper_projection_target_aggregation": str(target_aggregation),
        "projection_consistency_update_mode": "scalarized",
        "projection_consistency_weighting": "uniform",
        "projection_consistency_advantage_temperature": 1.0,
        "projection_consistency_advantage_weight_clip": 5.0,
        "projection_consistency_step_scale": 1.0,
        "projection_consistency_max_backtracks": 8,
        "projection_consistency_reward_tolerance": 0.0,
        "projection_consistency_training_schedule": str(training_schedule),
        "projection_consistency_warmup_fraction": float(warmup_fraction),
        "projection_consistency_ramp_fraction": float(ramp_fraction),
        "responsibility_mode": "additive",
        "upper_action_decoder_mode": "hold",
        "lower_action_router_mode": "direct",
        "lower_action_router_alpha": 0.20,
        "lower_action_router_strength": 1.0,
        "leakage_constraint_scope": "responsibility",
        "leakage_cost_mode": "power_excess",
        "upper_constraint_mode": "static_reward_penalty",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 0.0,
        "lower_dual_lr": 0.0,
    }


ARMS = {
    TERMINAL_RESERVE_ZERO: _arm(
        role="terminal_reserve_without_actor_consistency",
        upper_coef=0.0,
        lower_coef=0.0,
        target_aggregation="macro_mean",
        training_schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    ),
    MACRO_MEAN_UNIFORM_010: _arm(
        role="v22_macro_mean_upper_and_lower_consistency_control",
        upper_coef=0.10,
        lower_coef=0.10,
        target_aggregation="macro_mean",
        training_schedule="delayed_linear",
        warmup_fraction=0.50,
        ramp_fraction=0.25,
    ),
    LOWER_ONLY_DECISION_TIME_010: _arm(
        role="lower_consistency_only_upper_interference_ablation",
        upper_coef=0.0,
        lower_coef=0.10,
        target_aggregation="decision_time",
        training_schedule="delayed_linear",
        warmup_fraction=0.50,
        ramp_fraction=0.25,
    ),
    DECISION_TIME_UNIFORM_010: _arm(
        role="causal_decision_time_upper_and_lower_consistency_candidate",
        upper_coef=0.10,
        lower_coef=0.10,
        target_aggregation="decision_time",
        training_schedule="delayed_linear",
        warmup_fraction=0.50,
        ramp_fraction=0.25,
    ),
}

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 512
UPPER_PERIOD = 16
HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
PPO_CLIP_RATIO = 0.10
LOWER_LF_RMS_BUDGET = 0.0475
UPPER_HF_RMS_BUDGET = 0.075
TERMINAL_RESERVE_UPPER_WINDOW = 8
TERMINAL_RESERVE_LOWER_WINDOW = 32
UPPER_ACTION_SCALE = 1.0
LOWER_ACTION_SCALE = 1.0
CHECKPOINT_SELECTION_MODE = "crossed_conditions"
CHECKPOINT_SCORE_MODE = "mean_reward"
CHECKPOINT_SMOOTHING_WINDOW = 1
CHECKPOINT_MIN_DELTA = 0.0
CHECKPOINT_MINIMUM_ITERATION = 383
CHECKPOINT_EVALUATION_INTERVAL = 16

MAXIMUM_REWARD_REGRESSION_FRACTION = 0.05
MINIMUM_TOTAL_REWARD_WINS = 8
MINIMUM_REWARD_WINS_PER_ENVIRONMENT = 2
MINIMUM_REWARD_IMPROVED_ENVIRONMENTS_VS_OLD = 2
MINIMUM_CORRECTION_REDUCTION_FRACTION = 0.05
MAXIMUM_CORRECTION_REGRESSION_FRACTION = 0.05
MINIMUM_CORRECTION_IMPROVED_ENVIRONMENTS = 2
MAXIMUM_MEAN_TOTAL_CORRECTION_RMS = 0.25
MAXIMUM_RECURSIVE_FALLBACK_RATE = 0.05
MINIMUM_PROJECTION_CONVERGED_RATE = 0.95
POWER_TOLERANCE = 1e-8

EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
ADVANCES_STATUS = "v23_causal_upper_projection_target_development_advances"
STOPS_STATUS = "v23_causal_upper_projection_target_development_stops"

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_root",
    "question": (
        "does a decision-time upper projection target remove hindsight macro "
        "label contamination and improve reward-correction tradeoffs"
    ),
    "matched_control": (
        "all four arms use terminal-reserve projection, identical capacity, "
        "training paths, heldout paths, budgets, and checkpoint window"
    ),
    "causal_change": (
        "the old control averages all projected lower-step targets in a macro; "
        "the candidate trains the one upper action only against the first "
        "same-observation projected target"
    ),
    "isolation_arm": (
        "the lower-only arm keeps lower consistency while setting upper "
        "consistency to zero, isolating upper-label interference"
    ),
    "validity_gate": (
        "every arm has zero certificate violations, convergence at least "
        "0.95, fallback at most 0.05, and prefix power within frozen budgets"
    ),
    "reward_gate": (
        "a candidate stays within five percent of both zero and old controls "
        "in every environment, wins at least eight of twelve paired roots and "
        "at least two of four roots per environment, and improves mean reward "
        "over old control in at least two environments"
    ),
    "correction_gate": (
        "component and total correction each improve at least five percent "
        "over zero in two environments, regress no more than five percent in "
        "any environment versus zero or old control, and Hopper mean total "
        "correction remains at most 0.25"
    ),
    "selection_rule": (
        "advance the full decision-time candidate if it passes; otherwise "
        "advance lower-only only if it passes its gates. No metric-dependent "
        "parameter or threshold changes are permitted after execution"
    ),
    "outcome_use": (
        "development screen only; a pass authorizes fresh-root confirmation"
    ),
    "claim_boundary": (
        "no confirmatory, superiority, generalization, or manuscript claim "
        "may be made from four optimizer roots"
    ),
}


def validate() -> None:
    roles = (OPTIMIZER_SEEDS, TRAIN_SEEDS, SELECTION_SEEDS, EVALUATION_SEEDS)
    flattened = tuple(seed for values in roles for seed in values)
    if len(flattened) != 20 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v23 requires twenty disjoint fresh roots")
    if set(ARMS) != {
        TERMINAL_RESERVE_ZERO,
        MACRO_MEAN_UNIFORM_010,
        LOWER_ONLY_DECISION_TIME_010,
        DECISION_TIME_UNIFORM_010,
    }:
        raise RuntimeError("v23 arm registry is incomplete")
    if any(
        not bool(arm["terminal_reserve_context"])
        or not bool(arm["terminal_reserve_projection"])
        for arm in ARMS.values()
    ):
        raise RuntimeError("v23 requires projected terminal-reserve arms")
    old = ARMS[MACRO_MEAN_UNIFORM_010]
    causal = ARMS[DECISION_TIME_UNIFORM_010]
    ignored = {"arm_role", "upper_projection_target_aggregation"}
    if (
        {key: value for key, value in old.items() if key not in ignored}
        != {key: value for key, value in causal.items() if key not in ignored}
    ):
        raise RuntimeError("v23 full arms must differ only in target aggregation")
    lower_only = ARMS[LOWER_ONLY_DECISION_TIME_010]
    ignored = {"arm_role", "upper_projection_consistency_coef"}
    if (
        {key: value for key, value in lower_only.items() if key not in ignored}
        != {key: value for key, value in causal.items() if key not in ignored}
    ):
        raise RuntimeError("v23 causal arms must differ only in upper coefficient")
    if CHECKPOINT_MINIMUM_ITERATION != int(ITERATIONS * 0.75) - 1:
        raise RuntimeError("v23 checkpoint eligibility must begin after the ramp")
    if EXPECTED_CELL_COUNT != 48 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v23 matrix dimensions drifted")


validate()
