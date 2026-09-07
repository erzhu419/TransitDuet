"""Frozen fresh-root preflight for reward-selective feasible-action learning."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v21_reward_selective_feasible_action_preflight_v1"
)
EVIDENCE_ROLE = (
    "fresh_root_reward_selective_feasible_action_preflight_not_confirmatory"
)
PREREGISTRATION_STATUS = "frozen_before_v21_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = (
    "f215434f404c4adee1203e206b0a414d21369994"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v21_reward_selective_feasible_action_training"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from numpy Generator(210031), checked against all earlier
# MuJoCo script literals, and frozen before any v21 rollout was executed.
OPTIMIZER_SEEDS = (
    3050844425,
    1093779803,
    1203861316,
    290837716,
)
TRAIN_SEEDS = (1944078224, 2475918680, 2253500024, 1698098636)
SELECTION_SEEDS = (1842391225, 3142520896, 4009458608, 3655980172)
EVALUATION_SEEDS = (
    2767903569,
    3980367566,
    911381254,
    1122097198,
    1328371807,
    3493335395,
    3471387283,
    3611221946,
)

RAW_CONTEXT = "raw_context_v21_preflight"
TERMINAL_RESERVE_ZERO = "terminal_reserve_consistency_000"
DELAYED_UNIFORM_010 = "terminal_reserve_delayed_uniform_010"
DELAYED_REWARD_SELECTIVE_010 = (
    "terminal_reserve_delayed_reward_selective_010"
)
PRIMARY_RAW_REFERENCE = RAW_CONTEXT
PRIMARY_MECHANISM_BASELINE = TERMINAL_RESERVE_ZERO
PRIMARY_UNIFORM_BASELINE = DELAYED_UNIFORM_010
CANDIDATES = (DELAYED_REWARD_SELECTIVE_010,)


def _arm(
    *,
    role: str,
    projection: bool,
    consistency_coef: float,
    consistency_weighting: str,
    training_schedule: str,
    warmup_fraction: float,
    ramp_fraction: float,
) -> dict[str, object]:
    return {
        "method": "freq_hrl",
        "arm_role": str(role),
        "terminal_reserve_context": True,
        "terminal_reserve_projection": bool(projection),
        "upper_projection_consistency_coef": float(consistency_coef),
        "lower_projection_consistency_coef": float(consistency_coef),
        "projection_consistency_update_mode": "scalarized",
        "projection_consistency_weighting": str(consistency_weighting),
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
    RAW_CONTEXT: _arm(
        role="capacity_matched_unprojected_reference",
        projection=False,
        consistency_coef=0.0,
        consistency_weighting="uniform",
        training_schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    ),
    TERMINAL_RESERVE_ZERO: _arm(
        role="terminal_reserve_without_actor_consistency",
        projection=True,
        consistency_coef=0.0,
        consistency_weighting="uniform",
        training_schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    ),
    DELAYED_UNIFORM_010: _arm(
        role="delayed_uniform_certified_action_consistency_0_10",
        projection=True,
        consistency_coef=0.10,
        consistency_weighting="uniform",
        training_schedule="delayed_linear",
        warmup_fraction=0.50,
        ramp_fraction=0.25,
    ),
    DELAYED_REWARD_SELECTIVE_010: _arm(
        role="delayed_reward_selective_certified_action_consistency_0_10",
        projection=True,
        consistency_coef=0.10,
        consistency_weighting="exp_reward_advantage",
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

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 21003191
CONFIDENCE = 0.95

MAXIMUM_REWARD_REGRESSION_FRACTION_VS_UNIFORM = 0.05
MINIMUM_REWARD_IMPROVED_ENVIRONMENTS = 2
MINIMUM_POOLED_NORMALIZED_REWARD_DELTA = 0.0
MINIMUM_CORRECTION_REDUCTION_FRACTION_VS_RESERVE = 0.05
MAXIMUM_CORRECTION_REGRESSION_FRACTION_VS_UNIFORM = 0.05
MINIMUM_CORRECTION_IMPROVED_ENVIRONMENTS = 2
MAXIMUM_MEAN_TOTAL_CORRECTION_RMS = 0.25
MAXIMUM_MEAN_TOTAL_ACTION_CHANGE_RATE = 0.50
MAXIMUM_RECURSIVE_FALLBACK_RATE = 0.05
MINIMUM_PROJECTION_CONVERGED_RATE = 0.95
WEIGHT_MEAN_TOLERANCE = 1e-5
MINIMUM_SELECTIVE_WEIGHT_MAX = 1.05
POWER_TOLERANCE = 1e-8

EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
SUPPORTED_STATUS = (
    "v21_reward_selective_feasible_action_preflight_advances"
)
NOT_SUPPORTED_STATUS = (
    "v21_reward_selective_feasible_action_preflight_stops"
)

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_root",
    "question": (
        "does unit-mean reward-advantage weighting improve heldout reward "
        "relative to uniform certified-action consistency without giving back "
        "its correction reduction"
    ),
    "capacity_control": (
        "all arms use the same terminal-reserve context, hidden size, training "
        "budget, checkpoint window, train paths, and heldout paths"
    ),
    "weighting_control": (
        "uniform and reward-selective arms use the same 0.10 coefficient and "
        "delayed-linear schedule; reward-selective exponential weights are "
        "detached, clipped before normalization, and normalized to unit "
        "minibatch mean"
    ),
    "validity_gate": (
        "all projected cells have zero certificate violations, convergence at "
        "least 0.95, fallback at most 0.05, and realized prefix powers within "
        "the frozen budgets"
    ),
    "reward_gate": (
        "reward-selective reward improves over uniform in at least two "
        "environments, has positive pooled optimizer-root normalized delta, "
        "and is no worse than five percent below uniform in any environment"
    ),
    "correction_gate": (
        "component and total correction each improve at least five percent "
        "over zero-consistency reserve in at least two environments and do not "
        "regress more than five percent from uniform in any environment"
    ),
    "weight_audit_gate": (
        "both levels have active weighted updates in every candidate cell, "
        "unit mean weight within tolerance, selective maximum weight above "
        "1.05, and finite positive weighted and unweighted MSE"
    ),
    "physical_burden_gate": (
        "candidate environment means have total correction RMS at most 0.25 "
        "and total action-change rate at most 0.50"
    ),
    "stopping_rule": (
        "a failed gate stops this weighting mechanism; these roots cannot be "
        "reused for temperature, clip, coefficient, or schedule tuning"
    ),
    "outcome_use": (
        "development preflight only; a pass authorizes a separately committed "
        "multi-seed development protocol, not confirmation"
    ),
    "claim_boundary": (
        "no manuscript, confirmatory, generalization, no-tradeoff, or "
        "superiority claim may be made from this preflight"
    ),
}


def validate() -> None:
    roles = (OPTIMIZER_SEEDS, TRAIN_SEEDS, SELECTION_SEEDS, EVALUATION_SEEDS)
    flattened = tuple(seed for values in roles for seed in values)
    if len(flattened) != 20 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v21 requires twenty disjoint fresh seed roots")
    if set(ARMS) != {
        RAW_CONTEXT,
        TERMINAL_RESERVE_ZERO,
        DELAYED_UNIFORM_010,
        DELAYED_REWARD_SELECTIVE_010,
    }:
        raise RuntimeError("v21 arm registry is incomplete")
    if any(not bool(arm["terminal_reserve_context"]) for arm in ARMS.values()):
        raise RuntimeError("v21 capacity control requires context in every arm")
    if bool(ARMS[RAW_CONTEXT]["terminal_reserve_projection"]):
        raise RuntimeError("v21 raw reference must remain unprojected")
    if any(
        not bool(ARMS[arm]["terminal_reserve_projection"])
        for arm in (
            TERMINAL_RESERVE_ZERO,
            DELAYED_UNIFORM_010,
            DELAYED_REWARD_SELECTIVE_010,
        )
    ):
        raise RuntimeError("v21 mechanism arms must execute terminal projection")
    ignored = {"arm_role", "projection_consistency_weighting"}
    uniform = {
        key: value
        for key, value in ARMS[DELAYED_UNIFORM_010].items()
        if key not in ignored
    }
    selective = {
        key: value
        for key, value in ARMS[DELAYED_REWARD_SELECTIVE_010].items()
        if key not in ignored
    }
    if uniform != selective:
        raise RuntimeError("v21 weighting arms must differ only in weighting")
    if CHECKPOINT_MINIMUM_ITERATION != int(ITERATIONS * 0.75) - 1:
        raise RuntimeError("v21 checkpoint eligibility must begin after the ramp")
    if EXPECTED_CELL_COUNT != 48 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v21 matrix dimensions drifted")


validate()
