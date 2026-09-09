"""Fresh-root confirmation of unchanged uniform terminal-reserve learning."""

from __future__ import annotations


CONFIRMATION_PROTOCOL_VERSION = (
    "mujoco_v22_uniform_terminal_reserve_confirmation_v1"
)
EVIDENCE_ROLE = (
    "fresh_root_confirmation_of_unchanged_v21_uniform_control"
)
PREREGISTRATION_STATUS = "frozen_before_v22_training_or_heldout_access"
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

# Generated once from NumPy Generator(220091), checked against integer
# literals in all earlier MuJoCo scripts, and frozen before v22 execution.
OPTIMIZER_SEEDS = (
    1794002840,
    597672289,
    1591213397,
    1084883939,
    2051183718,
    2931641124,
    1541748580,
    1986713541,
    583450238,
    4009566154,
    4101524110,
    3484809059,
    2347830862,
    3233000298,
    3456182624,
    958556050,
    2242180897,
    188499802,
    2437940829,
    2975284689,
    666859139,
    3410901072,
    2784006636,
    1939611561,
    2772092650,
    1347125503,
    3953885964,
    1453516718,
    466744269,
    3490586229,
    2835703161,
    1867015079,
)
TRAIN_SEEDS = (3465118021, 3651981531, 4153959286, 3783282708)
SELECTION_SEEDS = (1417968496, 3108225556, 1376084330, 1609951530)
EVALUATION_SEEDS = (
    3871809708,
    2565254957,
    4069924302,
    4109154919,
    2670762119,
    766992686,
    1342016651,
    2978256823,
)

RAW_CONTEXT = "raw_context_v22_confirmation"
TERMINAL_RESERVE_ZERO = "terminal_reserve_consistency_000"
DELAYED_UNIFORM_010 = "terminal_reserve_delayed_uniform_010"
PRIMARY_RAW_REFERENCE = RAW_CONTEXT
PRIMARY_PROJECTED_BASELINE = TERMINAL_RESERVE_ZERO
CANDIDATE = DELAYED_UNIFORM_010


def _arm(
    *,
    role: str,
    projection: bool,
    consistency_coef: float,
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
    RAW_CONTEXT: _arm(
        role="capacity_matched_unprojected_reference",
        projection=False,
        consistency_coef=0.0,
        training_schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    ),
    TERMINAL_RESERVE_ZERO: _arm(
        role="terminal_reserve_without_actor_consistency",
        projection=True,
        consistency_coef=0.0,
        training_schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    ),
    DELAYED_UNIFORM_010: _arm(
        role="unchanged_v21_delayed_uniform_consistency_0_10",
        projection=True,
        consistency_coef=0.10,
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

BOOTSTRAP_DRAWS = 50_000
BOOTSTRAP_SEED = 22009117
CONFIDENCE = 0.95
ENVIRONMENT_FAMILY_CONFIDENCE = 1.0 - (1.0 - CONFIDENCE) / len(ENVIRONMENTS)

REWARD_NONINFERIORITY_MARGIN = 0.05
MINIMUM_CORRECTION_REDUCTION_FRACTION = 0.05
MAXIMUM_CORRECTION_REGRESSION_FRACTION = 0.05
MINIMUM_IMPROVED_ENVIRONMENTS = 2
MAXIMUM_MEAN_TOTAL_CORRECTION_RMS = 0.25
MAXIMUM_RECURSIVE_FALLBACK_RATE = 0.05
MINIMUM_PROJECTION_CONVERGED_RATE = 0.95
UNIFORM_WEIGHT_TOLERANCE = 1e-5
POWER_TOLERANCE = 1e-8

EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
SUPPORTED_STATUS = "v22_uniform_terminal_reserve_confirmation_supported"
NOT_SUPPORTED_STATUS = (
    "v22_uniform_terminal_reserve_confirmation_not_supported"
)

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_root",
    "question": (
        "does the unchanged v21 delayed uniform consistency mechanism reduce "
        "terminal-reserve correction while preserving projected reward"
    ),
    "capacity_control": (
        "all arms retain identical terminal-reserve context, network capacity, "
        "training paths, checkpoint window, and heldout paths"
    ),
    "algorithm_freeze": (
        "the candidate is the unchanged v21 uniform control at coefficient "
        "0.10, 512 iterations, and delayed-linear schedule; no v21 result was "
        "used to alter its algorithm or hyperparameters"
    ),
    "primary_reward_gate": (
        "paired family-adjusted lower bounds must exceed the negative five "
        "percent reserve-reward margin in every environment, and the pooled "
        "paired lower bound must exceed the same margin"
    ),
    "primary_correction_gate": (
        "component and total correction must each have a pooled paired lower "
        "bound above five percent reduction, improve by at least five percent "
        "in at least two environment point estimates, and have no environment "
        "family-adjusted lower bound below negative five percent"
    ),
    "secondary_reward_superiority": (
        "reward-superiority intervals and root wins are reported but are not "
        "required for the registered no-tradeoff confirmation"
    ),
    "validity_gate": (
        "all projected cells must have zero certificate violations, at least "
        "0.95 projection convergence, at most 0.05 recursive fallback, and "
        "realized prefix powers within the frozen budgets"
    ),
    "burden_gate": (
        "candidate mean total correction RMS must not exceed 0.25 in any "
        "environment; action-change rate is reported but is not substituted "
        "for correction magnitude"
    ),
    "stopping_rule": (
        "any failed primary gate rejects the joint confirmation; roots cannot "
        "be reused to tune coefficient, schedule, thresholds, or checkpoints"
    ),
    "claim_boundary": (
        "support establishes only fresh-root MuJoCo evidence for correction "
        "reduction with projected-reward noninferiority; it does not establish "
        "reward superiority, cross-domain superiority, or deployment validity"
    ),
}


def validate() -> None:
    roles = (OPTIMIZER_SEEDS, TRAIN_SEEDS, SELECTION_SEEDS, EVALUATION_SEEDS)
    flattened = tuple(seed for values in roles for seed in values)
    if len(flattened) != 48 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v22 requires forty-eight disjoint fresh roots")
    if set(ARMS) != {
        RAW_CONTEXT,
        TERMINAL_RESERVE_ZERO,
        DELAYED_UNIFORM_010,
    }:
        raise RuntimeError("v22 arm registry is incomplete")
    if any(not bool(arm["terminal_reserve_context"]) for arm in ARMS.values()):
        raise RuntimeError("v22 capacity control requires context in every arm")
    if bool(ARMS[RAW_CONTEXT]["terminal_reserve_projection"]):
        raise RuntimeError("v22 raw reference must remain unprojected")
    if any(
        not bool(ARMS[arm]["terminal_reserve_projection"])
        for arm in (TERMINAL_RESERVE_ZERO, DELAYED_UNIFORM_010)
    ):
        raise RuntimeError("v22 projected arms must execute terminal reserve")
    candidate = ARMS[DELAYED_UNIFORM_010]
    if (
        candidate["projection_consistency_weighting"] != "uniform"
        or candidate["upper_projection_consistency_coef"] != 0.10
        or candidate["lower_projection_consistency_coef"] != 0.10
        or candidate["projection_consistency_training_schedule"]
        != "delayed_linear"
        or candidate["projection_consistency_warmup_fraction"] != 0.50
        or candidate["projection_consistency_ramp_fraction"] != 0.25
    ):
        raise RuntimeError("v22 candidate drifted from the v21 uniform control")
    if CHECKPOINT_MINIMUM_ITERATION != int(ITERATIONS * 0.75) - 1:
        raise RuntimeError("v22 checkpoint eligibility must begin after ramp")
    if EXPECTED_CELL_COUNT != 288 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v22 matrix dimensions drifted")


validate()
