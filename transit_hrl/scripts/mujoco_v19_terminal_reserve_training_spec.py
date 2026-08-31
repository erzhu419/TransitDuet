"""Frozen fresh-seed development design for v19 terminal-reserve training."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v19_terminal_reserve_training_v1"
EVIDENCE_ROLE = "fresh_seed_terminal_reserve_training_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v19_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = "eaea91fad9bdbf1a243f54bd96ae15d23f693665"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v19_terminal_reserve_training"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from numpy Generator(190031), then checked against all earlier
# MuJoCo role literals and frozen before any v19 rollout was executed.
OPTIMIZER_SEEDS = (
    3088197441,
    1883930713,
    3546468138,
    1531997931,
    3189865153,
    3396850489,
    2166810815,
    1835781706,
    3071522062,
    1223463720,
    3550609777,
    2673201213,
)
TRAIN_SEEDS = (2801407646, 3538378772, 3039542946, 1842923961)
SELECTION_SEEDS = (1808810181, 721346637, 3866862330, 2288818196)
EVALUATION_SEEDS = (
    3134170705,
    1071564934,
    800746795,
    2808561187,
    3015723292,
    1137889904,
    1545700430,
    2812576900,
)

RAW_CONTEXT_BASELINE = "raw_context_baseline"
TERMINAL_RESERVE_NO_CONSISTENCY = "terminal_reserve_consistency_000"
TERMINAL_RESERVE_CONSISTENCY_001 = "terminal_reserve_consistency_001"
TERMINAL_RESERVE_CONSISTENCY_003 = "terminal_reserve_consistency_003"
TERMINAL_RESERVE_CONSISTENCY_010 = "terminal_reserve_consistency_010"
PRIMARY_MECHANISM_BASELINE = TERMINAL_RESERVE_NO_CONSISTENCY
CONSISTENCY_CANDIDATES = (
    TERMINAL_RESERVE_CONSISTENCY_001,
    TERMINAL_RESERVE_CONSISTENCY_003,
    TERMINAL_RESERVE_CONSISTENCY_010,
)


def _arm(
    *,
    role: str,
    projection: bool,
    consistency_coef: float,
) -> dict[str, object]:
    return {
        "method": "freq_hrl",
        "arm_role": str(role),
        "terminal_reserve_context": True,
        "terminal_reserve_projection": bool(projection),
        "upper_projection_consistency_coef": float(consistency_coef),
        "lower_projection_consistency_coef": float(consistency_coef),
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
    RAW_CONTEXT_BASELINE: _arm(
        role="capacity_matched_unprojected_behavioral_reference",
        projection=False,
        consistency_coef=0.0,
    ),
    TERMINAL_RESERVE_NO_CONSISTENCY: _arm(
        role="terminal_reserve_execution_without_actor_consistency",
        projection=True,
        consistency_coef=0.0,
    ),
    TERMINAL_RESERVE_CONSISTENCY_001: _arm(
        role="terminal_reserve_with_fixed_consistency_0_01",
        projection=True,
        consistency_coef=0.01,
    ),
    TERMINAL_RESERVE_CONSISTENCY_003: _arm(
        role="terminal_reserve_with_fixed_consistency_0_03",
        projection=True,
        consistency_coef=0.03,
    ),
    TERMINAL_RESERVE_CONSISTENCY_010: _arm(
        role="terminal_reserve_with_fixed_consistency_0_10",
        projection=True,
        consistency_coef=0.10,
    ),
}

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 128
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
CHECKPOINT_MINIMUM_ITERATION = 31
CHECKPOINT_EVALUATION_INTERVAL = 8

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 19003191
CONFIDENCE = 0.95
REWARD_NONINFERIORITY_FRACTION_VS_RESERVE = 0.05
REWARD_NONINFERIORITY_FRACTION_VS_RAW = 0.10
MINIMUM_COMPONENT_CORRECTION_RELATIVE_REDUCTION = 0.05
MINIMUM_SUPPORTED_ENVIRONMENTS = 2
MAXIMUM_RECURSIVE_FALLBACK_RATE = 0.05
MINIMUM_PROJECTION_CONVERGED_RATE = 0.95
POWER_TOLERANCE = 1e-8
EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
SUPPORTED_STATUS = "v19_terminal_reserve_training_supported_for_confirmation_freeze"
NOT_SUPPORTED_STATUS = "v19_terminal_reserve_training_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_seed",
    "capacity_control": (
        "all arms receive the same terminal-reserve causal context and hidden "
        "dimension; the raw arm audits but does not project its action"
    ),
    "checkpoint_selection": (
        "reward only on crossed checkpoint-selection roots every eight "
        "iterations; terminal metrics and heldout rows cannot select checkpoints"
    ),
    "mechanism_comparison": (
        "each fixed consistency coefficient is paired against the projected "
        "zero-consistency arm on identical optimizer and heldout path roots"
    ),
    "validity_gate": (
        "every projected heldout path has zero terminal-certificate violations, "
        "projection convergence rate at least 0.95, recursive fallback rate at "
        "most 0.05, and realized upper/lower prefix powers within frozen budgets"
    ),
    "reward_gate": (
        "paired optimizer-root bootstrap lower confidence bound is above the "
        "predeclared 5 percent noninferiority margin versus projected "
        "zero-consistency and 10 percent versus the unprojected raw reference "
        "in every environment"
    ),
    "consistency_gate": (
        "component-correction RMS improves by at least 5 percent with a paired "
        "bootstrap lower confidence bound above zero in the pooled analysis and "
        "in at least two of three environments"
    ),
    "selection_order": (
        "eligible candidates only; maximize worst-environment estimated component "
        "correction reduction, then pooled reduction, then choose the smaller "
        "predeclared coefficient"
    ),
    "stopping_rule": (
        "if no candidate passes all gates, v19 development stops without changing "
        "coefficients or thresholds on these roots"
    ),
    "outcome_use": (
        "development-only coefficient selection; a pass authorizes a separately "
        "committed and seed-disjoint confirmation protocol"
    ),
    "claim_boundary": (
        "no manuscript, confirmatory, generalization, no-tradeoff, or superiority "
        "claim may be made from this panel"
    ),
}


def validate() -> None:
    roles = (OPTIMIZER_SEEDS, TRAIN_SEEDS, SELECTION_SEEDS, EVALUATION_SEEDS)
    flattened = tuple(seed for values in roles for seed in values)
    if len(flattened) != 28 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v19 requires twenty-eight disjoint fresh seed roots")
    if set(ARMS) != {
        RAW_CONTEXT_BASELINE,
        TERMINAL_RESERVE_NO_CONSISTENCY,
        *CONSISTENCY_CANDIDATES,
    }:
        raise RuntimeError("v19 arm registry is incomplete")
    if any(not bool(arm["terminal_reserve_context"]) for arm in ARMS.values()):
        raise RuntimeError("v19 capacity control requires context in every arm")
    if bool(ARMS[RAW_CONTEXT_BASELINE]["terminal_reserve_projection"]):
        raise RuntimeError("v19 raw reference must remain unprojected")
    if any(
        not bool(ARMS[arm]["terminal_reserve_projection"])
        for arm in (PRIMARY_MECHANISM_BASELINE, *CONSISTENCY_CANDIDATES)
    ):
        raise RuntimeError("v19 mechanism arms must execute terminal projection")
    if EXPECTED_CELL_COUNT != 180 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v19 matrix dimensions drifted")


validate()
