"""Frozen fresh-seed development design for v20 reserve training."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v20_reward_guarded_terminal_reserve_training_v1"
)
EVIDENCE_ROLE = (
    "fresh_seed_long_horizon_reward_guarded_reserve_development_not_confirmatory"
)
PREREGISTRATION_STATUS = "frozen_before_v20_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = "26cc2e45117bcbff15ae7cbdf0458f4582a9bf0e"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v20_reward_guarded_terminal_reserve_training"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from numpy Generator(200031), checked against all earlier
# MuJoCo role literals, and frozen before any v20 rollout was executed.
OPTIMIZER_SEEDS = (
    3161196872,
    184116694,
    4260783900,
    1357059719,
    1578548752,
    424835688,
    204024486,
    3798039993,
    609520533,
    1667063007,
    2944402381,
    1951942280,
)
TRAIN_SEEDS = (1949949083, 1314210472, 4066567685, 101143503)
SELECTION_SEEDS = (1227604259, 1234900727, 3182162935, 513536403)
EVALUATION_SEEDS = (
    1290727953,
    3739763510,
    2990616636,
    90109290,
    3126399748,
    904315824,
    1130863732,
    1283862557,
)

RAW_CONTEXT_LONG = "raw_context_long"
TERMINAL_RESERVE_LONG = "terminal_reserve_long_consistency_000"
DELAYED_SCALARIZED_010 = "terminal_reserve_delayed_scalarized_010"
DELAYED_REWARD_GUARDED_010 = "terminal_reserve_delayed_guarded_010"
PRIMARY_RAW_BASELINE = RAW_CONTEXT_LONG
PRIMARY_MECHANISM_BASELINE = TERMINAL_RESERVE_LONG
CANDIDATES = (
    DELAYED_SCALARIZED_010,
    DELAYED_REWARD_GUARDED_010,
)


def _arm(
    *,
    role: str,
    projection: bool,
    consistency_coef: float,
    update_mode: str,
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
        "projection_consistency_update_mode": str(update_mode),
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
    RAW_CONTEXT_LONG: _arm(
        role="capacity_matched_unprojected_long_horizon_reference",
        projection=False,
        consistency_coef=0.0,
        update_mode="scalarized",
        training_schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    ),
    TERMINAL_RESERVE_LONG: _arm(
        role="long_horizon_reserve_execution_without_actor_consistency",
        projection=True,
        consistency_coef=0.0,
        update_mode="scalarized",
        training_schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    ),
    DELAYED_SCALARIZED_010: _arm(
        role="delayed_linear_scalarized_consistency_ablation_0_10",
        projection=True,
        consistency_coef=0.10,
        update_mode="scalarized",
        training_schedule="delayed_linear",
        warmup_fraction=0.50,
        ramp_fraction=0.25,
    ),
    DELAYED_REWARD_GUARDED_010: _arm(
        role="delayed_linear_reward_guarded_consistency_0_10",
        projection=True,
        consistency_coef=0.10,
        update_mode="reward_guarded_projection",
        training_schedule="delayed_linear",
        warmup_fraction=0.50,
        ramp_fraction=0.25,
    ),
}

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 384
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
# The first eligible checkpoint is evaluated only after the delayed ramp has
# reached its full target at 75 percent of the training budget.
CHECKPOINT_MINIMUM_ITERATION = 287
CHECKPOINT_EVALUATION_INTERVAL = 16

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20003191
CONFIDENCE = 0.95
REWARD_NONINFERIORITY_FRACTION_VS_RESERVE = 0.05
REWARD_NONINFERIORITY_FRACTION_VS_RAW = 0.10
MINIMUM_COMPONENT_CORRECTION_RELATIVE_REDUCTION = 0.05
MINIMUM_TOTAL_CORRECTION_RELATIVE_REDUCTION = 0.05
MINIMUM_SUPPORTED_ENVIRONMENTS = 2
MAXIMUM_MEAN_TOTAL_CORRECTION_RMS = 0.25
MAXIMUM_MEAN_TOTAL_ACTION_CHANGE_RATE = 0.50
MAXIMUM_RECURSIVE_FALLBACK_RATE = 0.05
MINIMUM_PROJECTION_CONVERGED_RATE = 0.95
POWER_TOLERANCE = 1e-8
EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
SUPPORTED_STATUS = (
    "v20_reward_guarded_reserve_training_supported_for_confirmation_freeze"
)
NOT_SUPPORTED_STATUS = "v20_reward_guarded_reserve_training_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_seed",
    "capacity_control": (
        "all arms use the same terminal-reserve causal context, hidden size, "
        "training budget, checkpoint eligibility window, and heldout paths"
    ),
    "long_horizon_control": (
        "all arms train for 384 iterations and select only checkpoints at or "
        "after iteration 287; this isolates horizon from consistency mode"
    ),
    "delayed_consistency": (
        "candidate consistency is zero for the first half of training, ramps "
        "linearly over the next quarter, and remains at 0.10 for the final quarter"
    ),
    "reward_guard": (
        "the guarded arm applies consistency only through projected backtracking "
        "steps that do not worsen the same-minibatch PPO reward surrogate or an "
        "active native leakage surrogate; this is not a heldout reward guarantee"
    ),
    "validity_gate": (
        "every projected heldout path has zero certificate violations, projection "
        "convergence at least 0.95, fallback at most 0.05, and prefix powers within "
        "the frozen budgets"
    ),
    "reward_gate": (
        "each candidate's paired optimizer-root bootstrap lower bound is above "
        "the 5 percent noninferiority margin versus long projected reserve and "
        "the 10 percent margin versus the long unprojected reference in every environment"
    ),
    "correction_gate": (
        "component and total correction RMS each improve by at least 5 percent "
        "with pooled lower confidence bounds above zero; component correction is "
        "also supported in at least two of three environments"
    ),
    "physical_burden_gate": (
        "each candidate environment mean has total correction RMS at most 0.25 "
        "and total action change rate at most 0.50"
    ),
    "guard_audit_gate": (
        "the guarded arm must execute and accept at least one consistency step, "
        "reduce its training consistency loss, and report no positive maximum "
        "reward or active native-constraint surrogate delta"
    ),
    "selection_order": (
        "eligible candidates only; maximize worst-environment component correction "
        "reduction, then pooled total correction reduction, then pooled reward "
        "difference versus reserve; exact ties prefer the guarded arm"
    ),
    "stopping_rule": (
        "if neither candidate passes every registered gate, v20 stops and these "
        "roots cannot be reused for another coefficient or schedule search"
    ),
    "outcome_use": (
        "development-only mechanism and schedule selection; a pass authorizes a "
        "separately committed seed-disjoint confirmation protocol"
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
        raise RuntimeError("v20 requires twenty-eight disjoint fresh seed roots")
    if set(ARMS) != {
        RAW_CONTEXT_LONG,
        TERMINAL_RESERVE_LONG,
        *CANDIDATES,
    }:
        raise RuntimeError("v20 arm registry is incomplete")
    if any(not bool(arm["terminal_reserve_context"]) for arm in ARMS.values()):
        raise RuntimeError("v20 capacity control requires context in every arm")
    if bool(ARMS[RAW_CONTEXT_LONG]["terminal_reserve_projection"]):
        raise RuntimeError("v20 raw reference must remain unprojected")
    if any(
        not bool(ARMS[arm]["terminal_reserve_projection"])
        for arm in (PRIMARY_MECHANISM_BASELINE, *CANDIDATES)
    ):
        raise RuntimeError("v20 mechanism arms must execute terminal projection")
    if CHECKPOINT_MINIMUM_ITERATION != int(
        ITERATIONS * 0.75
    ) - 1:
        raise RuntimeError("v20 checkpoint eligibility must begin after the ramp")
    if EXPECTED_CELL_COUNT != 144 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v20 matrix dimensions drifted")


validate()
