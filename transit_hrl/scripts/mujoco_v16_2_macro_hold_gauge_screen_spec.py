"""Frozen development design for the MuJoCo v16.2 macro-hold gauge."""

from __future__ import annotations

from scripts import mujoco_v16_1_audit_gauge_paired_preflight_spec as v16_1


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v16_2_macro_hold_gauge_screen_v1"
EVIDENCE_ROLE = "macro_rate_gauge_mechanism_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v16_2_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = "9d57aaeb96ed62a06c5145dfe351e69fda623bc2"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "e75a5e351432457cbea3e61e40de46da8369359c87c24e114269d90a078d2f88"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v16_2_macro_hold_gauge"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from numpy Generator(16200), then frozen before dispatch.
OPTIMIZER_SEEDS = (3336873242, 1294171483, 3155548334)
TRAIN_SEEDS = (2695297615, 1369212792, 3641061460, 3358205932)
SELECTION_SEEDS = (3071541655, 1600450167, 2169056837, 134566248)
EVALUATION_SEEDS = (
    3603352854,
    4077345503,
    1719984135,
    4241012650,
    3502084011,
    2968493775,
    1448626019,
    4171413154,
)

DIRECT_CONTROL = "direct_reward_control"
PRIMITIVE_GAUGE_CONTROL = "primitive_audit_gauge_control"
MACRO_HOLD_CANDIDATE = "macro_hold_audit_gauge_candidate"
PRIMARY_COMPARATOR_ARM = DIRECT_CONTROL
PRIMARY_CANDIDATE_ARM = MACRO_HOLD_CANDIDATE


def _arm(*, role: str, router_mode: str) -> dict[str, object]:
    return {
        "method": "freq_hrl",
        "arm_role": str(role),
        "responsibility_mode": "additive",
        "lower_action_router_mode": str(router_mode),
        "lower_action_router_alpha": 0.20,
        "lower_action_router_strength": 1.0,
        "lower_action_router_training_schedule": "constant",
        "lower_action_router_warmup_fraction": 0.0,
        "lower_action_router_ramp_fraction": 0.0,
        "lower_action_router_observe_strength": False,
        "leakage_constraint_scope": "joint_behavior_latent",
        "leakage_cost_mode": "power_excess",
        "upper_constraint_mode": "primal_dual",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": 0.0,
        "lower_dual_lr": 0.0,
        "constraint_dual_normalization": "none",
        "constraint_dual_scale_ema_beta": 0.95,
        "constraint_dual_scale_floor": 1e-6,
    }


ARMS = {
    DIRECT_CONTROL: _arm(role="matched_direct_reward_control", router_mode="direct"),
    PRIMITIVE_GAUGE_CONTROL: _arm(
        role="primitive_rate_adaptive_gauge_control",
        router_mode="causal_audit_aligned_gauge",
    ),
    MACRO_HOLD_CANDIDATE: _arm(
        role="macro_rate_adaptive_gauge_candidate",
        router_mode="causal_macro_hold_audit_gauge",
    ),
}

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 64
UPPER_PERIOD = 16
HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
LOWER_LF_RMS_BUDGET = 0.0475
UPPER_HF_RMS_BUDGET = 0.075
UPPER_ACTION_SCALE = 1.0
LOWER_ACTION_SCALE = 1.0
CHECKPOINT_SELECTION_MODE = "crossed_conditions"
CHECKPOINT_SCORE_MODE = "mean_reward"
CHECKPOINT_SMOOTHING_WINDOW = 1
CHECKPOINT_MIN_DELTA = 0.0
CHECKPOINT_MINIMUM_ITERATION = 15
CHECKPOINT_EVALUATION_INTERVAL = 4

REWARD_NONINFERIORITY_FRACTION = 0.05
MINIMUM_LOWER_LF_RELATIVE_REDUCTION = 0.10
MINIMUM_JOINT_MERIT_RELATIVE_REDUCTION = 0.10
MAXIMUM_ROUTER_CLIP_RATE = 0.0
RECONSTRUCTION_RMS_TOLERANCE = 1e-7
MINIMUM_SUPPORTED_SEEDS_PER_ENVIRONMENT = 2
EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
SUPPORTED_STATUS = "macro_hold_gauge_screen_supported"
NOT_SUPPORTED_STATUS = "macro_hold_gauge_screen_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_seed",
    "training_comparison": (
        "capacity_matched_direct_primitive_adaptive_and_macro_hold_adaptive"
    ),
    "checkpoint_selection": "reward_only_on_crossed_selection_roots",
    "heldout_gate": (
        "reward_noninferiority_exact_reconstruction_zero_router_clipping_"
        "upper_hpf8_absolute_budget_lower_lpf32_reduction_and_joint_merit_"
        "reduction_in_two_of_three_seeds_per_environment"
    ),
    "outcome_use": "development_only_freeze_new_confirmatory_roots_if_supported",
}


def validate() -> None:
    roles = (OPTIMIZER_SEEDS, TRAIN_SEEDS, SELECTION_SEEDS, EVALUATION_SEEDS)
    flattened = tuple(seed for values in roles for seed in values)
    consumed = set(
        v16_1.OPTIMIZER_SEEDS
        + v16_1.PRETRAIN_SEEDS
        + v16_1.PRETRAIN_SELECTION_SEEDS
        + v16_1.CONTINUATION_TRAIN_SEEDS
        + v16_1.CONTINUATION_SELECTION_SEEDS
        + v16_1.EVALUATION_SEEDS
    )
    if len(flattened) != 19 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v16.2 requires nineteen disjoint seed roots")
    if consumed & set(flattened):
        raise RuntimeError("v16.2 seed roots overlap v16.1")
    if set(ARMS) != {
        DIRECT_CONTROL,
        PRIMITIVE_GAUGE_CONTROL,
        MACRO_HOLD_CANDIDATE,
    }:
        raise RuntimeError("v16.2 arm registry is incomplete")
    if ARMS[MACRO_HOLD_CANDIDATE]["lower_action_router_mode"] != (
        "causal_macro_hold_audit_gauge"
    ):
        raise RuntimeError("v16.2 candidate must use the macro-hold gauge")
    if any(
        float(arm[key]) != 0.0
        for arm in ARMS.values()
        for key in ("upper_hf_penalty_coef", "upper_dual_lr", "lower_dual_lr")
    ):
        raise RuntimeError("v16.2 is a reward-only routing comparison")
    if EXPECTED_CELL_COUNT != 27 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v16.2 matrix dimensions drifted")


validate()
