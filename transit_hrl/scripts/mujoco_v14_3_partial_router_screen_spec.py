"""Frozen development design for the MuJoCo v14.3 mechanism screen."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_3_partial_router_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_3_partial_action_router"
)
FROZEN_ALGORITHM_REVISION = "8d3aafd9c9447de3fd18664fe311cda79ad811ab"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "e683b7212e19943d89ffc716d5f08594760e3401f9ad22d6ededf6c56e597d25"
)

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


def _arm(
    *,
    method: str,
    responsibility_mode: str,
    router_mode: str,
    router_alpha: float,
    router_strength: float,
    upper_constraint_mode: str,
    upper_dual_lr: float,
    lower_dual_lr: float,
    checkpoint_score_mode: str,
    checkpoint_constraint_penalty: float,
) -> dict[str, object]:
    return {
        "method": str(method),
        "responsibility_mode": str(responsibility_mode),
        "lower_action_router_mode": str(router_mode),
        "lower_action_router_alpha": float(router_alpha),
        "lower_action_router_strength": float(router_strength),
        "leakage_constraint_scope": "joint_behavior",
        "leakage_cost_mode": "power_excess",
        "upper_constraint_mode": str(upper_constraint_mode),
        "upper_hf_penalty_coef": 0.0,
        "upper_dual_lr": float(upper_dual_lr),
        "lower_dual_lr": float(lower_dual_lr),
        "checkpoint_score_mode": str(checkpoint_score_mode),
        "checkpoint_constraint_penalty": float(
            checkpoint_constraint_penalty
        ),
    }


ARMS = {
    "additive_reward_baseline": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="additive",
        router_mode="direct",
        router_alpha=0.10,
        router_strength=1.0,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s006_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.06,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s008_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.08,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s010_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.10,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s015_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.15,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a010_s006_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.10,
        router_strength=0.06,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a010_s010_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.10,
        router_strength=0.10,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a010_s015_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.10,
        router_strength=0.15,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
}
BASELINE_ARM = "additive_reward_baseline"
CANDIDATE_ARMS = tuple(arm for arm in ARMS if arm != BASELINE_ARM)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    3333425400, 3974972019, 3893910007, 3436741594,
    1575618435, 458274522, 1067918395, 552637040,
    464552979, 2210242093, 4008495173, 4112573355,
    3722525949, 4142040046, 2391983371, 2667478682,
)
TRAIN_SEEDS = (943876846, 1271244249, 143897444, 1968261658)
CHECKPOINT_SELECTION_SEEDS = (3501432136, 1555677388)
DEVELOPMENT_EVALUATION_SEEDS = (
    3006570167, 3996347095, 3305979521, 4198935788,
    431230652, 1102348647, 2654924164, 2083582081,
)

STEPS = 512
EPISODE_HORIZON = 1000
ITERATIONS = 64
UPPER_PERIOD = 16
HIDDEN_DIM = 64
LEARNING_RATE = 3e-4
LOWER_LF_RMS_BUDGET = 0.05
UPPER_HF_RMS_BUDGET = 0.04
UPPER_HF_REPORTING_GATE = 0.10
UPPER_ACTION_SCALE = 1.0
LOWER_ACTION_SCALE = 1.0
LOWER_CONSTRAINT_UPDATE_MODE = "reward_guarded_adam_projection"
UPPER_CONSTRAINT_UPDATE_MODE = "reward_guarded_adam_projection"
CHECKPOINT_SELECTION_MODE = "crossed_conditions"
CHECKPOINT_SMOOTHING_WINDOW = 8
CHECKPOINT_MIN_DELTA = 1e-3
CHECKPOINT_EVALUATION_INTERVAL = 4

RETURN_NONINFERIORITY_MARGIN_FRACTION = 0.02
MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION = 0.10
MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION = 0.10
DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER = 0.01
DRIFT_MATERIALITY_FLOOR = (
    DRIFT_MATERIALITY_FLOOR_FRACTION_OF_BUDGET_POWER
    * LOWER_LF_RMS_BUDGET * LOWER_LF_RMS_BUDGET
)
MINIMUM_EFFECTIVE_LOWER_ACTION_RMS_FRACTION = 0.25
MAXIMUM_ROUTER_CLIP_RATE = 0.05
MAXIMUM_RECONSTRUCTION_RMS = 1e-7
MINIMUM_TRAINED_CHECKPOINT_FRACTION = 0.75
DUAL_SATURATION_THRESHOLD = 19.8
MAXIMUM_DUAL_SATURATION_FRACTION = 0.25
MINIMUM_STRICT_RESPONSIBILITY_IMPROVEMENT_CONDITIONS = 1
MINIMUM_STRICT_RAW_IMPROVEMENT_CONDITIONS = 1
SELECTION_CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 20_000

DEVELOPMENT_DISCLOSURE = (
    "This v14.3 screen was designed after inspecting the failed v14.2 "
    "development screen. It isolates seven partial-strength causal EMA "
    "lower-action routers against the unchanged additive reward baseline on "
    "fresh development seeds. The minimum strength is fixed above the 10% "
    "ideal DC-power reduction threshold. No primal-dual arm is included because "
    "v14.2 showed saturation and confounded the router comparison. At-floor "
    "conditions count only as absolute noninferiority, never as strict "
    "improvement. Any selected arm requires a new frozen v15 confirmation."
)


def validate_frozen_design() -> None:
    sequences = {
        "optimizer": OPTIMIZER_SEEDS,
        "training": TRAIN_SEEDS,
        "checkpoint_selection": CHECKPOINT_SELECTION_SEEDS,
        "development_evaluation": DEVELOPMENT_EVALUATION_SEEDS,
    }
    for name, values in sequences.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"invalid v14.3 development seed registry: {name}")
    roles = {name: set(values) for name, values in sequences.items()}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if roles[left] & roles[right]:
                raise ValueError(
                    f"v14.3 development seed roles overlap: {left}/{right}"
                )
    if len(OPTIMIZER_SEEDS) != 16:
        raise ValueError("v14.3 screen requires sixteen optimizer replicates")
    if len(ARMS) != 8 or BASELINE_ARM not in ARMS:
        raise ValueError("v14.3 algorithm-screen arm registry drifted")
    for name, arm in ARMS.items():
        if (
            str(arm["method"]) != "freq_hrl_no_leakage"
            or str(arm["upper_constraint_mode"]) != "disabled"
            or float(arm["upper_dual_lr"]) != 0.0
            or float(arm["lower_dual_lr"]) != 0.0
            or str(arm["checkpoint_score_mode"]) != "mean_reward"
        ):
            raise ValueError(f"v14.3 reward-only arm is invalid: {name}")
        if name != BASELINE_ARM:
            strength = float(arm["lower_action_router_strength"])
            ideal_dc_reduction = 1.0 - (1.0 - strength) ** 2
            if (
                str(arm["lower_action_router_mode"])
                != "causal_ema_high_pass"
                or ideal_dc_reduction
                < MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION
            ):
                raise ValueError(f"v14.3 partial router is too weak: {name}")
    if not 0.0 < DRIFT_MATERIALITY_FLOOR < LOWER_LF_RMS_BUDGET ** 2:
        raise ValueError("v14.3 materiality floor must be within the LF budget")


validate_frozen_design()
