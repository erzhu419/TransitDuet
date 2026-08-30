"""Frozen development design for the MuJoCo v17 raw-action architecture."""

from __future__ import annotations

from scripts import mujoco_v16_2_macro_hold_gauge_screen_spec as v16_2


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_zero_dc_plan_screen_v1"
EVIDENCE_ROLE = "raw_action_frequency_architecture_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v17_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = "62c4cbf5dbc44bf1873c198a2cdd4cfeefbcb30a"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "4c779fa40701eeefacc20791100be2dd4392c4946fa6db90d41b14ea0a2f742c"
)
FROZEN_CORE_PROTOCOL_VERSION = "freq_hrl_mujoco_shared_core_v17_zero_dc_plan"
ZERO_DC_ROUTER_CONTRACT = (
    "causal_bounded_lower_projection_with_exact_zero_sum_on_each_complete_"
    "upper_macro_interval_v1"
)
SMOOTH_PLAN_CONTRACT = "boundary_sampled_c1_smoothstep_primitive_execution_v1"

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from numpy Generator(17000), then frozen before dispatch.
OPTIMIZER_SEEDS = (1016138490, 1584069686, 333347051)
TRAIN_SEEDS = (2219478712, 846652140, 1575311654, 4153019297)
SELECTION_SEEDS = (3884505759, 836238374, 3470013715, 763675809)
EVALUATION_SEEDS = (
    3055595555,
    3670858713,
    1124240632,
    2772403977,
    2053281724,
    3299861743,
    3972158380,
    4028727757,
)

HOLD_DIRECT_CONTROL = "hold_direct_control"
SMOOTH_DIRECT_CONTROL = "smooth_direct_control"
ZERO_DC_PLAN_CANDIDATE = "zero_dc_plan_candidate"
PRIMARY_COMPARATOR_ARM = HOLD_DIRECT_CONTROL
MECHANISM_COMPARATOR_ARM = SMOOTH_DIRECT_CONTROL
PRIMARY_CANDIDATE_ARM = ZERO_DC_PLAN_CANDIDATE


def _arm(
    *,
    role: str,
    upper_action_decoder_mode: str,
    lower_action_router_mode: str,
) -> dict[str, object]:
    return {
        "method": "freq_hrl",
        "arm_role": str(role),
        "responsibility_mode": "additive",
        "upper_action_decoder_mode": str(upper_action_decoder_mode),
        "lower_action_router_mode": str(lower_action_router_mode),
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
    HOLD_DIRECT_CONTROL: _arm(
        role="zero_order_hold_direct_reward_control",
        upper_action_decoder_mode="hold",
        lower_action_router_mode="direct",
    ),
    SMOOTH_DIRECT_CONTROL: _arm(
        role="smooth_upper_plan_direct_lower_ablation",
        upper_action_decoder_mode="causal_smoothstep_plan",
        lower_action_router_mode="direct",
    ),
    ZERO_DC_PLAN_CANDIDATE: _arm(
        role="smooth_upper_plan_causal_zero_dc_lower_candidate",
        upper_action_decoder_mode="causal_smoothstep_plan",
        lower_action_router_mode="causal_macro_zero_dc",
    ),
}

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 128
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
CHECKPOINT_MINIMUM_ITERATION = 31
CHECKPOINT_EVALUATION_INTERVAL = 4

REWARD_NONINFERIORITY_FRACTION = 0.05
MINIMUM_UPPER_HF_RELATIVE_REDUCTION = 0.10
MINIMUM_LOWER_LF_RELATIVE_REDUCTION = 0.10
MINIMUM_JOINT_MERIT_RELATIVE_REDUCTION = 0.10
MACRO_COMPLETION_ERROR_TOLERANCE = 1e-7
RESPONSIBILITY_RECONSTRUCTION_TOLERANCE = 1e-7
MINIMUM_PROJECTION_RATE = 1e-12
MINIMUM_SUPPORTED_SEEDS_PER_ENVIRONMENT = 2
EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
SUPPORTED_STATUS = "zero_dc_plan_screen_supported"
NOT_SUPPORTED_STATUS = "zero_dc_plan_screen_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_fresh_optimizer_seed",
    "training_comparison": (
        "capacity_matched_hold_direct_smooth_direct_and_smooth_zero_dc"
    ),
    "checkpoint_selection": "reward_only_on_crossed_selection_roots",
    "heldout_gate": (
        "reward_noninferiority_smooth_upper_hpf8_reduction_raw_lower_lpf32_"
        "reduction_vs_smooth_direct_and_latent_proposal_raw_joint_merit_"
        "reduction_exact_complete_macro_zero_sum_active_projection_and_exact_"
        "responsibility_reconstruction_in_two_of_three_seeds_per_environment"
    ),
    "outcome_use": "development_only_freeze_new_confirmatory_roots_if_supported",
}


def validate() -> None:
    roles = (OPTIMIZER_SEEDS, TRAIN_SEEDS, SELECTION_SEEDS, EVALUATION_SEEDS)
    flattened = tuple(seed for values in roles for seed in values)
    consumed = set(
        v16_2.v16_1.OPTIMIZER_SEEDS
        + v16_2.v16_1.PRETRAIN_SEEDS
        + v16_2.v16_1.PRETRAIN_SELECTION_SEEDS
        + v16_2.v16_1.CONTINUATION_TRAIN_SEEDS
        + v16_2.v16_1.CONTINUATION_SELECTION_SEEDS
        + v16_2.v16_1.EVALUATION_SEEDS
        + v16_2.OPTIMIZER_SEEDS
        + v16_2.TRAIN_SEEDS
        + v16_2.SELECTION_SEEDS
        + v16_2.EVALUATION_SEEDS
    )
    if len(flattened) != 19 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v17 requires nineteen disjoint seed roots")
    if consumed & set(flattened):
        raise RuntimeError("v17 seed roots overlap v16.1 or v16.2")
    if set(ARMS) != {
        HOLD_DIRECT_CONTROL,
        SMOOTH_DIRECT_CONTROL,
        ZERO_DC_PLAN_CANDIDATE,
    }:
        raise RuntimeError("v17 arm registry is incomplete")
    candidate = ARMS[ZERO_DC_PLAN_CANDIDATE]
    if (
        candidate["upper_action_decoder_mode"] != "causal_smoothstep_plan"
        or candidate["lower_action_router_mode"] != "causal_macro_zero_dc"
        or float(candidate["lower_action_router_strength"]) != 1.0
    ):
        raise RuntimeError("v17 candidate must use the full raw-action architecture")
    if any(
        float(arm[key]) != 0.0
        for arm in ARMS.values()
        for key in ("upper_hf_penalty_coef", "upper_dual_lr", "lower_dual_lr")
    ):
        raise RuntimeError("v17 is a reward-only architecture comparison")
    if EXPECTED_CELL_COUNT != 27 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v17 matrix dimensions drifted")


validate()
