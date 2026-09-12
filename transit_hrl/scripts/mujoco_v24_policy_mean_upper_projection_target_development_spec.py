"""Frozen development screen for a policy-mean causal upper target."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v24_policy_mean_upper_projection_target_development_v1"
)
EVIDENCE_ROLE = "fresh_root_policy_mean_upper_target_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v24_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = (
    "8e3b571185a84d0adb00307421a89c0f38a81412"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v24_policy_mean_upper_projection_target_training"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from NumPy Generator(240091), checked against integer
# literals in earlier MuJoCo scripts, and frozen before v24 execution.
OPTIMIZER_SEEDS = (484849350, 258618141, 2646551552, 3409328541)
TRAIN_SEEDS = (3096957538, 922193267, 94336199, 1682946743)
SELECTION_SEEDS = (84601147, 1244592955, 3870808872, 597143134)
EVALUATION_SEEDS = (
    3249365916,
    2260141802,
    931810172,
    79750395,
    1505559812,
    1240798192,
    2238439710,
    3759950514,
)

TERMINAL_RESERVE_ZERO = "terminal_reserve_consistency_000"
MACRO_MEAN_UNIFORM_010 = "terminal_reserve_macro_mean_uniform_010"
DECISION_TIME_UNIFORM_010 = "terminal_reserve_decision_time_uniform_010"
POLICY_MEAN_UNIFORM_010 = "terminal_reserve_policy_mean_uniform_010"
PRIMARY_PROJECTED_BASELINE = TERMINAL_RESERVE_ZERO
PRIMARY_CAUSAL_CONTROL = DECISION_TIME_UNIFORM_010
DIAGNOSTIC_HINDSIGHT_CONTROL = MACRO_MEAN_UNIFORM_010
CANDIDATES = (POLICY_MEAN_UNIFORM_010,)


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
        role="hindsight_macro_mean_upper_and_lower_consistency_diagnostic",
        upper_coef=0.10,
        lower_coef=0.10,
        target_aggregation="macro_mean",
        training_schedule="delayed_linear",
        warmup_fraction=0.50,
        ramp_fraction=0.25,
    ),
    DECISION_TIME_UNIFORM_010: _arm(
        role="v23_stochastic_first_sample_causal_control",
        upper_coef=0.10,
        lower_coef=0.10,
        target_aggregation="decision_time",
        training_schedule="delayed_linear",
        warmup_fraction=0.50,
        ramp_fraction=0.25,
    ),
    POLICY_MEAN_UNIFORM_010: _arm(
        role="same_state_deterministic_lower_policy_mean_causal_candidate",
        upper_coef=0.10,
        lower_coef=0.10,
        target_aggregation="decision_policy_mean",
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
MINIMUM_REWARD_IMPROVED_ENVIRONMENTS = 2
MINIMUM_CORRECTION_REDUCTION_FRACTION = 0.05
MAXIMUM_CORRECTION_REGRESSION_FRACTION = 0.05
MINIMUM_CORRECTION_IMPROVED_ENVIRONMENTS = 2
MAXIMUM_MEAN_TOTAL_CORRECTION_RMS = 0.25
MAXIMUM_RECURSIVE_FALLBACK_RATE = 0.05
MINIMUM_UPPER_MSE_REDUCTION_FRACTION = 0.05
MAXIMUM_UPPER_MSE_REGRESSION_FRACTION = 0.05
MINIMUM_UPPER_MSE_IMPROVED_ENVIRONMENTS = 2
POWER_TOLERANCE = 1e-8

EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
ADVANCES_STATUS = "v24_policy_mean_upper_projection_target_development_advances"
STOPS_STATUS = "v24_policy_mean_upper_projection_target_development_stops"

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_root",
    "question": (
        "does a deterministic lower-policy mean evaluated at the same causal "
        "decision state reduce upper-target noise without sacrificing reward "
        "or terminal-reserve correction"
    ),
    "matched_control": (
        "all four arms use terminal-reserve projection, identical capacity, "
        "training paths, heldout paths, budgets, and checkpoint window"
    ),
    "isolated_change": (
        "the v23 causal control projects the sampled lower action at the upper "
        "decision; the v24 candidate previews the same projector with the "
        "lower policy distribution mean from that same forward pass"
    ),
    "hindsight_diagnostic": (
        "macro_mean remains a non-adoptable diagnostic because it uses later "
        "states in the current macro interval"
    ),
    "safety_gate": (
        "every arm has zero certificate violations, fallback at most 0.05, "
        "and realized prefix power within the frozen budgets"
    ),
    "numerical_diagnostic": (
        "Dykstra step-tolerance convergence is reported but is not treated as "
        "certificate feasibility; every returned action is checked separately"
    ),
    "mechanism_gate": (
        "only the policy-mean arm records same-state policy-mean targets, with "
        "positive sampled-versus-mean target delta during training; its upper "
        "consistency MSE improves by five percent in at least two environments "
        "and regresses by no more than five percent in any environment versus "
        "the v23 first-sample control"
    ),
    "reward_gate": (
        "the candidate stays within five percent of zero, v23 causal, and "
        "hindsight controls in every environment; wins at least eight of twelve "
        "paired roots and at least two of four roots per environment versus v23; "
        "and improves mean reward versus v23 in at least two environments"
    ),
    "correction_gate": (
        "component and total correction each improve at least five percent "
        "over zero in two environments, regress no more than five percent in "
        "any environment versus all three controls, and Hopper mean total "
        "correction remains at most 0.25"
    ),
    "selection_rule": (
        "advance only the policy-mean candidate if every frozen gate passes; "
        "otherwise stop. No metric-dependent parameter or threshold changes "
        "are permitted after execution"
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
        raise RuntimeError("v24 requires twenty disjoint fresh roots")
    if set(ARMS) != {
        TERMINAL_RESERVE_ZERO,
        MACRO_MEAN_UNIFORM_010,
        DECISION_TIME_UNIFORM_010,
        POLICY_MEAN_UNIFORM_010,
    }:
        raise RuntimeError("v24 arm registry is incomplete")
    if any(
        not bool(arm["terminal_reserve_context"])
        or not bool(arm["terminal_reserve_projection"])
        for arm in ARMS.values()
    ):
        raise RuntimeError("v24 requires projected terminal-reserve arms")
    first_sample = ARMS[DECISION_TIME_UNIFORM_010]
    policy_mean = ARMS[POLICY_MEAN_UNIFORM_010]
    ignored = {"arm_role", "upper_projection_target_aggregation"}
    if (
        {key: value for key, value in first_sample.items() if key not in ignored}
        != {key: value for key, value in policy_mean.items() if key not in ignored}
    ):
        raise RuntimeError("v24 causal arms must differ only in target estimator")
    if CHECKPOINT_MINIMUM_ITERATION != int(ITERATIONS * 0.75) - 1:
        raise RuntimeError("v24 checkpoint eligibility must begin after the ramp")
    if EXPECTED_CELL_COUNT != 48 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v24 matrix dimensions drifted")


validate()
