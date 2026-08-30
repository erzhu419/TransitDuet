"""Frozen paired-continuation design for the MuJoCo v16.1 audit gauge."""

from __future__ import annotations

from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as v14_17
from scripts import mujoco_v14_29_fresh_anchor_spec as v14_29
from scripts import mujoco_v15_2_multisource_distillation_preflight_spec as v15_2
from scripts import mujoco_v16_gauge_training_preflight_spec as v16


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v16_1_audit_gauge_paired_preflight_v1"
EVIDENCE_ROLE = "paired_training_mechanism_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v16_1_training_outcome_access"
FROZEN_ALGORITHM_REVISION = "8adf34ec37afffcbe25affe3c9a4891a904f8307"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "bcfd034bba3e9eeeab4c0695adccb30a32179bd43ab336f8ab56e7931906b8aa"
)
FROZEN_CORE_PROTOCOL_VERSION = v14_17.FROZEN_CORE_PROTOCOL_VERSION

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (
    *TRAINING_DISTURBANCE_MODES,
    "ood_chirp",
)

# Generated once from numpy Generator(16101), then frozen before dispatch.
OPTIMIZER_SEEDS = (1086114340, 1653726169, 2868862553)
PRETRAIN_SEEDS = (2413118206, 3146869205, 3399252035, 3387091244)
PRETRAIN_SELECTION_SEEDS = (
    3221601274,
    3354138110,
    4208001754,
    921126613,
)
CONTINUATION_TRAIN_SEEDS = (
    2128330667,
    1580197368,
    3693570184,
    3641131291,
)
CONTINUATION_SELECTION_SEEDS = (
    1961462541,
    4220458689,
    1905531089,
    586919525,
)
EVALUATION_SEEDS = (
    2601952734,
    157959330,
    2812269300,
    2893559944,
    1067955446,
    3056119537,
    3829565418,
    1595406502,
)

ANCHOR_ARM = "audit_gauge_reward_anchor"
REWARD_CONTINUATION_CONTROL = "audit_gauge_reward_continuation_control"
PRIMAL_DUAL_CANDIDATE = "audit_gauge_primal_dual_reward_floor_candidate"
PRIMARY_COMPARATOR_ARM = REWARD_CONTINUATION_CONTROL
PRIMARY_CANDIDATE_ARM = PRIMAL_DUAL_CANDIDATE


def _arm(*, role: str, dual_lr: float, checkpoint_score_mode: str) -> dict[str, object]:
    return {
        "method": "freq_hrl",
        "arm_role": str(role),
        "responsibility_mode": "additive",
        "lower_action_router_mode": "causal_audit_aligned_gauge",
        "lower_action_router_alpha": 0.20,
        "lower_action_router_strength": 1.0,
        "lower_action_router_training_schedule": "constant",
        "lower_action_router_warmup_fraction": 0.0,
        "lower_action_router_ramp_fraction": 0.0,
        "lower_action_router_observe_strength": False,
        "leakage_constraint_scope": "joint_behavior_latent",
        "leakage_cost_mode": "power_excess",
        # A zero-dual arm retains both cost critics, so capacity is identical.
        "upper_constraint_mode": "primal_dual",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": float(dual_lr),
        "lower_dual_lr": float(dual_lr),
        "constraint_dual_normalization": (
            "ema_abs" if float(dual_lr) > 0.0 else "none"
        ),
        "constraint_dual_scale_ema_beta": 0.95,
        "constraint_dual_scale_floor": 1e-6,
        "checkpoint_score_mode": str(checkpoint_score_mode),
        "upper_deployment_frequency_reference_reduction_fraction": (
            0.0 if checkpoint_score_mode == "mean_reward" else 0.05
        ),
        "lower_deployment_frequency_reference_reduction_fraction": (
            0.0 if checkpoint_score_mode == "mean_reward" else 0.05
        ),
    }


ANCHOR_SPEC = _arm(
    role="reward_compatible_adaptive_gauge_anchor",
    dual_lr=0.0,
    checkpoint_score_mode="mean_reward",
)
ARMS = {
    REWARD_CONTINUATION_CONTROL: _arm(
        role="compute_matched_zero_dual_continuation",
        dual_lr=0.0,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
    ),
    PRIMAL_DUAL_CANDIDATE: _arm(
        role="adaptive_gauge_primal_dual_with_paired_reward_floor",
        dual_lr=0.03,
        checkpoint_score_mode="paired_relative_frequency_feasibility_first",
    ),
}

STEPS = 512
EPISODE_HORIZON = 1000
PRETRAIN_ITERATIONS = 64
CONTINUATION_ITERATIONS = 32
UPPER_PERIOD = 16
HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
LOWER_LF_RMS_BUDGET = 0.0475
UPPER_HF_RMS_BUDGET = 0.075
UPPER_ACTION_SCALE = 1.0
LOWER_ACTION_SCALE = 1.0
CHECKPOINT_SELECTION_MODE = "crossed_conditions"
CHECKPOINT_SMOOTHING_WINDOW = 1
CHECKPOINT_MIN_DELTA = 0.0
PRETRAIN_CHECKPOINT_MINIMUM_ITERATION = 15
CONTINUATION_CHECKPOINT_MINIMUM_ITERATION = -1
CHECKPOINT_EVALUATION_INTERVAL = 4

REWARD_NONINFERIORITY_FRACTION = 0.05
CANONICAL_MIN_RELATIVE_REDUCTION = 0.10
LATENT_NONINFERIORITY_FRACTION = 0.05
MINIMUM_CANONICAL_IMPROVEMENT_CELLS = 6
MINIMUM_ENVIRONMENT_IMPROVEMENT_CELLS = 2
RECONSTRUCTION_RMS_TOLERANCE = 1e-7
EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_ANCHOR_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)
EXPECTED_CONTINUATION_CELL_COUNT = EXPECTED_ANCHOR_CELL_COUNT * len(ARMS)
SUPPORTED_STATUS = "audit_gauge_paired_preflight_supported"
NOT_SUPPORTED_STATUS = "audit_gauge_paired_preflight_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_seed_paired_continuation",
    "anchor": "reward_only_adaptive_gauge_with_matched_cost_critic_capacity",
    "primary_comparison": (
        "primal_dual_vs_zero_dual_from_identical_checkpoint_and_optimizer_state"
    ),
    "checkpoint_gate": (
        "paired_selection_path_reward_floor_and_five_frequency_endpoints_"
        "with_anchor_fallback"
    ),
    "heldout_gate": (
        "reward_noninferiority_exact_reconstruction_latent_noninferiority_and_"
        "canonical_frequency_reduction_in_two_of_three_seeds_per_environment"
    ),
    "outcome_use": "development_only_expand_with_new_roots_if_supported",
}


def validate() -> None:
    roles = (
        OPTIMIZER_SEEDS,
        PRETRAIN_SEEDS,
        PRETRAIN_SELECTION_SEEDS,
        CONTINUATION_TRAIN_SEEDS,
        CONTINUATION_SELECTION_SEEDS,
        EVALUATION_SEEDS,
    )
    flattened = tuple(seed for role in roles for seed in role)
    consumed = set(
        v14_17.OPTIMIZER_SEEDS
        + v14_17.PRETRAIN_SEEDS
        + v14_17.PRETRAIN_SELECTION_SEEDS
        + v14_17.CONTINUATION_TRAIN_SEEDS
        + v14_17.CONTINUATION_SELECTION_SEEDS
        + v14_17.DEVELOPMENT_EVALUATION_SEEDS
        + v14_29.OPTIMIZER_SEEDS
        + v14_29.PRETRAIN_SEEDS
        + v14_29.PRETRAIN_SELECTION_SEEDS
        + v14_29.DEVELOPMENT_EVALUATION_SEEDS
        + v15_2.DISTILL_ROOTS
        + v15_2.DESIGN_ROOTS
        + v15_2.VALIDATION_ROOTS
        + v16.OPTIMIZER_SEEDS
        + v16.TRAIN_SEEDS
        + v16.SELECTION_SEEDS
        + v16.EVALUATION_SEEDS
    )
    if len(flattened) != 27 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v16.1 requires twenty-seven disjoint seed roots")
    if consumed & set(flattened):
        raise RuntimeError("v16.1 seed roots overlap prior development roles")
    if set(ARMS) != {REWARD_CONTINUATION_CONTROL, PRIMAL_DUAL_CANDIDATE}:
        raise RuntimeError("v16.1 continuation arm registry is incomplete")
    if any(
        arm["lower_action_router_mode"] != "causal_audit_aligned_gauge"
        or arm["lower_action_router_strength"] != 1.0
        for arm in (ANCHOR_SPEC, *ARMS.values())
    ):
        raise RuntimeError("v16.1 must use the full adaptive gauge throughout")
    if ANCHOR_SPEC["upper_dual_lr"] != 0.0:
        raise RuntimeError("v16.1 anchor must not update constraint duals")
    if ARMS[PRIMAL_DUAL_CANDIDATE]["upper_dual_lr"] <= 0.0:
        raise RuntimeError("v16.1 candidate must enable primal-dual updates")
    if MINIMUM_CANONICAL_IMPROVEMENT_CELLS != (
        len(ENVIRONMENTS) * MINIMUM_ENVIRONMENT_IMPROVEMENT_CELLS
    ):
        raise RuntimeError("v16.1 aggregate and per-environment gates disagree")


validate()
