"""Frozen training-time gauge-fixing MuJoCo development preflight."""

from __future__ import annotations

from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as base
from scripts import mujoco_v14_29_fresh_anchor_spec as fresh_anchors
from scripts import mujoco_v15_2_multisource_distillation_preflight_spec as v15_2


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v16_gauge_training_preflight_v1"
EVIDENCE_ROLE = "training_time_mechanism_development_not_confirmatory"
FROZEN_ALGORITHM_REVISION = "2f9b3fed920261a9bee849488dec4f9f0c2755af"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "3726f294f9cd2c975212a613350c04eb14fe2bc0b215ab523b00d4faba061caa"
)
FROZEN_CORE_PROTOCOL_VERSION = base.FROZEN_CORE_PROTOCOL_VERSION
PREREGISTRATION_STATUS = "frozen_before_v16_training_outcome_access"

ENVIRONMENTS = base.ENVIRONMENTS
TRAINING_DISTURBANCE_MODES = base.TRAINING_DISTURBANCE_MODES
EVALUATION_DISTURBANCE_MODES = base.EVALUATION_DISTURBANCE_MODES

OPTIMIZER_SEEDS = (2747832873, 2863209335, 2359482451)
TRAIN_SEEDS = (1457761625, 587726797, 3907779710, 1840989119)
SELECTION_SEEDS = (2709886061, 2771595972, 3892654083, 3745426373)
EVALUATION_SEEDS = (
    4033698169,
    102659042,
    1673939509,
    1557074295,
    2777186273,
    3319287545,
    1389976933,
    1211442780,
)

JOINT_PD_CONTROL = "joint_band_primal_dual_control"
GAUGE_REWARD_CONTROL = "gauge_reward_only_control"
GAUGE_PD_CANDIDATE = "gauge_primal_dual_candidate"
PRIMARY_CANDIDATE_ARM = GAUGE_PD_CANDIDATE
PRIMARY_COMPARATOR_ARM = JOINT_PD_CONTROL


def _arm(
    *,
    role: str,
    router_mode: str,
    router_strength: float,
    dual_lr: float,
) -> dict[str, object]:
    return {
        "method": "freq_hrl",
        "arm_role": str(role),
        "responsibility_mode": "additive",
        "lower_action_router_mode": str(router_mode),
        "lower_action_router_alpha": 0.04,
        "lower_action_router_strength": float(router_strength),
        "lower_action_router_training_schedule": "constant",
        "lower_action_router_warmup_fraction": 0.0,
        "lower_action_router_ramp_fraction": 0.0,
        "lower_action_router_observe_strength": False,
        "leakage_constraint_scope": "joint_behavior_latent",
        "leakage_cost_mode": "power_excess",
        # The upper cost critic exists in every arm, including the zero-dual
        # control, so parameter capacity and initialization remain matched.
        "upper_constraint_mode": "primal_dual",
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": float(dual_lr),
        "lower_dual_lr": float(dual_lr),
        "constraint_dual_normalization": (
            "ema_abs" if float(dual_lr) > 0.0 else "none"
        ),
        "constraint_dual_scale_ema_beta": 0.95,
        "constraint_dual_scale_floor": 1e-6,
    }


ARMS = {
    JOINT_PD_CONTROL: _arm(
        role="established_joint_band_primal_dual_control",
        router_mode="causal_joint_band_projection",
        router_strength=0.5,
        dual_lr=0.03,
    ),
    GAUGE_REWARD_CONTROL: _arm(
        role="gauge_coordinate_without_constraint_updates",
        router_mode="causal_total_action_gauge",
        router_strength=1.0,
        dual_lr=0.0,
    ),
    GAUGE_PD_CANDIDATE: _arm(
        role="training_time_gauge_with_primal_dual_constraints",
        router_mode="causal_total_action_gauge",
        router_strength=1.0,
        dual_lr=0.03,
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
CHECKPOINT_SCORE_MODE = "latent_behavior_feasibility_first"
CHECKPOINT_SMOOTHING_WINDOW = 1
CHECKPOINT_MIN_DELTA = 0.0
CHECKPOINT_MINIMUM_ITERATION = 15
CHECKPOINT_EVALUATION_INTERVAL = 4

REWARD_NONINFERIORITY_FRACTION = 0.05
LATENT_NONINFERIORITY_FRACTION = 0.05
CANONICAL_MIN_RELATIVE_REDUCTION = 0.20
LATENT_MIN_RELATIVE_REDUCTION = 0.05
MINIMUM_LATENT_IMPROVEMENT_CELLS = 6
RECONSTRUCTION_RMS_TOLERANCE = 1e-7

EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
SUPPORTED_STATUS = "training_time_gauge_preflight_supported"
NOT_SUPPORTED_STATUS = "training_time_gauge_preflight_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_optimizer_seed",
    "capacity": "identical_actor_critic_architecture_and_initialization_by_seed",
    "primary_comparison": (
        "full_strength_total_action_gauge_primal_dual_vs_established_"
        "joint_band_primal_dual"
    ),
    "mechanism_control": (
        "full_strength_total_action_gauge_with_zero_dual_learning_rate"
    ),
    "support_gate": (
        "all_cells_exact_reconstruction_reward_noninferiority_canonical_"
        "frequency_reduction_and_latent_noninferiority_plus_at_least_two_of_"
        "three_seeds_per_environment_with_latent_constraint_improvement"
    ),
    "outcome_use": "development_only_expand_with_new_seeds_if_supported",
}


def validate() -> None:
    roles = (OPTIMIZER_SEEDS, TRAIN_SEEDS, SELECTION_SEEDS, EVALUATION_SEEDS)
    flattened = tuple(seed for role in roles for seed in role)
    consumed = set(
        base.OPTIMIZER_SEEDS
        + base.PRETRAIN_SEEDS
        + base.PRETRAIN_SELECTION_SEEDS
        + base.CONTINUATION_TRAIN_SEEDS
        + base.CONTINUATION_SELECTION_SEEDS
        + base.DEVELOPMENT_EVALUATION_SEEDS
        + fresh_anchors.OPTIMIZER_SEEDS
        + fresh_anchors.PRETRAIN_SEEDS
        + fresh_anchors.PRETRAIN_SELECTION_SEEDS
        + fresh_anchors.DEVELOPMENT_EVALUATION_SEEDS
        + v15_2.DISTILL_ROOTS
        + v15_2.DESIGN_ROOTS
        + v15_2.VALIDATION_ROOTS
    )
    if len(flattened) != 19 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v16 requires nineteen disjoint seed roots")
    if consumed & set(flattened):
        raise RuntimeError("v16 seed roots overlap prior development roles")
    if set(ARMS) != {
        JOINT_PD_CONTROL,
        GAUGE_REWARD_CONTROL,
        GAUGE_PD_CANDIDATE,
    }:
        raise RuntimeError("v16 arm registry is incomplete")
    if any(arm["upper_constraint_mode"] != "primal_dual" for arm in ARMS.values()):
        raise RuntimeError("v16 upper cost-critic capacity is not matched")
    if ARMS[GAUGE_REWARD_CONTROL]["upper_dual_lr"] != 0.0:
        raise RuntimeError("v16 gauge mechanism control must disable dual updates")
    if any(
        ARMS[arm]["lower_action_router_mode"] != "causal_total_action_gauge"
        or ARMS[arm]["lower_action_router_strength"] != 1.0
        for arm in (GAUGE_REWARD_CONTROL, GAUGE_PD_CANDIDATE)
    ):
        raise RuntimeError("v16 gauge arms must use the identified full gauge")
    if MINIMUM_LATENT_IMPROVEMENT_CELLS != 2 * len(ENVIRONMENTS):
        raise RuntimeError("v16 latent gate must require two seeds per environment")


validate()
