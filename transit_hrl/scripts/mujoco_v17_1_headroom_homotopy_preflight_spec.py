"""Frozen development design for MuJoCo v17.1 headroom homotopy."""

from __future__ import annotations

from scripts import mujoco_v17_zero_dc_plan_screen_spec as v17


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_1_headroom_homotopy_preflight_v1"
EVIDENCE_ROLE = "headroom_homotopy_architecture_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v17_1_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = "c732391d5f1ebb494f8c082c668a6fe24e197b13"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "a4e3a63e04d170e8f07f44df8b330c6a0fa24454a79da44ddd25926b621c280a"
)
DIRECT_CORE_PROTOCOL_VERSION = "freq_hrl_mujoco_shared_core_v17_zero_dc_plan"
CANDIDATE_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v17_1_headroom_homotopy_promotion"
)
HEADROOM_ROUTER_CONTRACT = (
    "causal_upper_plan_headroom_feasible_lower_homotopy_with_exact_zero_"
    "sum_at_full_strength_and_function_continuity_at_zero_strength_v1"
)
HEADROOM_ACTION_CONTRACT = (
    "frozen_upper_macro_suffix_reserves_per_step_total_action_headroom_"
    "before_environment_disturbance_v1"
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

# Generated once from numpy Generator(17100), then frozen before dispatch.
OPTIMIZER_SEEDS = (650181723,)
TRAIN_SEEDS = (1875668430, 1950958734, 3077312552, 3628923846)
SELECTION_SEEDS = (3839501204, 4013251988, 2971288153, 935966037)
SAFETY_SELECTION_SEEDS = (3523393901, 486112647)
EVALUATION_SEEDS = (
    4032637170,
    2105933771,
    604298204,
    404839443,
    3535357519,
    1550254159,
    293149975,
    2801601310,
)

SMOOTH_DIRECT_CONTROL = "smooth_direct_control"
HEADROOM_EXACT = "headroom_exact"
HEADROOM_HOMOTOPY = "headroom_homotopy"
HEADROOM_HOMOTOPY_PROMOTION_05 = "headroom_homotopy_promotion_05"
HEADROOM_HOMOTOPY_PROMOTION_10 = "headroom_homotopy_promotion_10"
CANDIDATE_ARMS = (
    HEADROOM_EXACT,
    HEADROOM_HOMOTOPY,
    HEADROOM_HOMOTOPY_PROMOTION_05,
    HEADROOM_HOMOTOPY_PROMOTION_10,
)


def _arm(
    *,
    role: str,
    router_mode: str,
    schedule: str,
    warmup_fraction: float,
    ramp_fraction: float,
    promotion_gain: float,
) -> dict[str, object]:
    candidate = router_mode == "causal_macro_zero_dc_headroom"
    return {
        "method": "freq_hrl",
        "arm_role": str(role),
        "responsibility_mode": "additive",
        "upper_action_decoder_mode": "causal_smoothstep_plan",
        "upper_promotion_gain": float(promotion_gain),
        "lower_action_router_mode": str(router_mode),
        "lower_action_router_alpha": 0.20,
        "lower_action_router_strength": 1.0 if candidate else 0.0,
        "lower_action_router_training_schedule": str(schedule),
        "lower_action_router_warmup_fraction": float(warmup_fraction),
        "lower_action_router_ramp_fraction": float(ramp_fraction),
        "lower_action_router_observe_strength": True,
        "control_protocol_version": (
            CANDIDATE_CORE_PROTOCOL_VERSION
            if candidate else DIRECT_CORE_PROTOCOL_VERSION
        ),
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
    SMOOTH_DIRECT_CONTROL: _arm(
        role="smooth_upper_plan_direct_lower_reward_control",
        router_mode="direct",
        schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
        promotion_gain=0.0,
    ),
    HEADROOM_EXACT: _arm(
        role="headroom_feasible_exact_zero_dc_without_curriculum",
        router_mode="causal_macro_zero_dc_headroom",
        schedule="constant",
        warmup_fraction=0.0,
        ramp_fraction=0.0,
        promotion_gain=0.0,
    ),
    HEADROOM_HOMOTOPY: _arm(
        role="headroom_feasible_zero_dc_with_causal_homotopy",
        router_mode="causal_macro_zero_dc_headroom",
        schedule="delayed_cosine",
        warmup_fraction=0.25,
        ramp_fraction=0.50,
        promotion_gain=0.0,
    ),
    HEADROOM_HOMOTOPY_PROMOTION_05: _arm(
        role="homotopy_with_half_gain_causal_upper_replanning",
        router_mode="causal_macro_zero_dc_headroom",
        schedule="delayed_cosine",
        warmup_fraction=0.25,
        ramp_fraction=0.50,
        promotion_gain=0.5,
    ),
    HEADROOM_HOMOTOPY_PROMOTION_10: _arm(
        role="homotopy_with_full_gain_causal_upper_replanning",
        router_mode="causal_macro_zero_dc_headroom",
        schedule="delayed_cosine",
        warmup_fraction=0.25,
        ramp_fraction=0.50,
        promotion_gain=1.0,
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
MAXIMUM_UPPER_HF_RELATIVE_INCREASE = 0.10
MINIMUM_LOWER_LF_RELATIVE_REDUCTION = 0.10
MINIMUM_JOINT_MERIT_RELATIVE_REDUCTION = 0.10
MACRO_COMPLETION_ERROR_TOLERANCE = 1e-7
RESPONSIBILITY_RECONSTRUCTION_TOLERANCE = 1e-7
ADDITIVE_CLIP_RATE_TOLERANCE = 1e-12
MINIMUM_PROJECTION_RATE = 1e-12
MINIMUM_PROMOTION_RMS = 1e-12
MINIMUM_PERFORMANCE_ENVIRONMENTS = 2
EXPECTED_EVALUATION_ROWS_PER_CELL = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS) * len(ARMS)
SUPPORTED_STATUS = "headroom_homotopy_preflight_supported"
NOT_SUPPORTED_STATUS = "headroom_homotopy_preflight_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_single_fresh_optimizer_seed_development_only",
    "training_comparison": (
        "capacity_matched_smooth_direct_headroom_exact_and_three_homotopy_"
        "promotion_variants"
    ),
    "checkpoint_selection": "reward_only_on_crossed_selection_roots",
    "candidate_eligibility": (
        "exact_headroom_mechanics_and_reward_noninferiority_in_all_three_"
        "environments_with_upper_nonworsening_lower_lf_reduction_and_joint_"
        "merit_reduction_in_at_least_two_environments"
    ),
    "global_arm_selection": (
        "maximize_worst_environment_reward_margin_then_median_joint_merit_"
        "reduction_among_eligible_arms"
    ),
    "outcome_use": (
        "development_only_freeze_one_global_arm_on_fresh_multiseed_roots_if_"
        "supported"
    ),
}


def validate() -> None:
    roles = (
        OPTIMIZER_SEEDS,
        TRAIN_SEEDS,
        SELECTION_SEEDS,
        SAFETY_SELECTION_SEEDS,
        EVALUATION_SEEDS,
    )
    flattened = tuple(seed for values in roles for seed in values)
    consumed = set(
        v17.OPTIMIZER_SEEDS
        + v17.TRAIN_SEEDS
        + v17.SELECTION_SEEDS
        + v17.EVALUATION_SEEDS
    )
    if len(flattened) != 19 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v17.1 requires nineteen disjoint fresh seed roots")
    if consumed & set(flattened):
        raise RuntimeError("v17.1 seed roots overlap v17")
    if tuple(arm for arm in ARMS if arm != SMOOTH_DIRECT_CONTROL) != CANDIDATE_ARMS:
        raise RuntimeError("v17.1 candidate arm registry drifted")
    for name, arm in ARMS.items():
        candidate = name in CANDIDATE_ARMS
        if bool(arm["lower_action_router_observe_strength"]) is not True:
            raise RuntimeError("v17.1 arms must have capacity-matched state")
        if candidate and (
            arm["lower_action_router_mode"]
            != "causal_macro_zero_dc_headroom"
            or float(arm["lower_action_router_strength"]) != 1.0
            or arm["control_protocol_version"]
            != CANDIDATE_CORE_PROTOCOL_VERSION
        ):
            raise RuntimeError("v17.1 candidate architecture drifted")
    for name in (
        HEADROOM_HOMOTOPY,
        HEADROOM_HOMOTOPY_PROMOTION_05,
        HEADROOM_HOMOTOPY_PROMOTION_10,
    ):
        arm = ARMS[name]
        if (
            arm["lower_action_router_training_schedule"] != "delayed_cosine"
            or float(arm["lower_action_router_warmup_fraction"]) != 0.25
            or float(arm["lower_action_router_ramp_fraction"]) != 0.50
        ):
            raise RuntimeError("v17.1 homotopy schedule drifted")
    if any(
        float(arm[key]) != 0.0
        for arm in ARMS.values()
        for key in ("upper_hf_penalty_coef", "upper_dual_lr", "lower_dual_lr")
    ):
        raise RuntimeError("v17.1 is a reward-only architecture comparison")
    if EXPECTED_CELL_COUNT != 15 or EXPECTED_EVALUATION_ROWS_PER_CELL != 40:
        raise RuntimeError("v17.1 matrix dimensions drifted")


validate()
