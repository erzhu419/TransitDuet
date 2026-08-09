"""Frozen development design for the MuJoCo v14.4 router-homotopy screen."""

from __future__ import annotations

import math


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_4_router_homotopy_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_4_router_homotopy"
)
FROZEN_ALGORITHM_REVISION = "03482136f155bdb86f1cf421cd320555bcb42c81"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "ed3f1db546d7171ceb11b05938890caead3b1f685a0fbcb4618fd4d2b4fef974"
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
    router_training_schedule: str,
    router_warmup_fraction: float,
    router_ramp_fraction: float,
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
        "lower_action_router_training_schedule": str(
            router_training_schedule
        ),
        "lower_action_router_warmup_fraction": float(
            router_warmup_fraction
        ),
        "lower_action_router_ramp_fraction": float(router_ramp_fraction),
        "lower_action_router_observe_strength": True,
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
    "causal_transfer_direct_baseline": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="direct",
        router_alpha=0.10,
        router_strength=0.0,
        router_training_schedule="constant",
        router_warmup_fraction=0.0,
        router_ramp_fraction=0.0,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s010_constant": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.10,
        router_training_schedule="constant",
        router_warmup_fraction=0.0,
        router_ramp_fraction=0.0,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s010_linear_w000_r025": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.10,
        router_training_schedule="delayed_linear",
        router_warmup_fraction=0.0,
        router_ramp_fraction=0.25,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s010_linear_w0125_r0375": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.10,
        router_training_schedule="delayed_linear",
        router_warmup_fraction=0.125,
        router_ramp_fraction=0.375,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s010_linear_w025_r050": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.10,
        router_training_schedule="delayed_linear",
        router_warmup_fraction=0.25,
        router_ramp_fraction=0.50,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s010_cosine_w025_r050": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.10,
        router_training_schedule="delayed_cosine",
        router_warmup_fraction=0.25,
        router_ramp_fraction=0.50,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s015_linear_w0125_r0375": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.15,
        router_training_schedule="delayed_linear",
        router_warmup_fraction=0.125,
        router_ramp_fraction=0.375,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "router_a004_s015_cosine_w025_r050": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        router_strength=0.15,
        router_training_schedule="delayed_cosine",
        router_warmup_fraction=0.25,
        router_ramp_fraction=0.50,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
}
BASELINE_ARM = "causal_transfer_direct_baseline"
CANDIDATE_ARMS = tuple(arm for arm in ARMS if arm != BASELINE_ARM)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    2309397706, 2358696663, 3287323368, 1626003633,
    971451679, 656933695, 2584760063, 2581273354,
    2400409935, 997854913, 2164697843, 1855685873,
    647903500, 413439144, 553947963, 1150144395,
)
TRAIN_SEEDS = (1736553203, 4175955086, 3352342059, 1491969400)
CHECKPOINT_SELECTION_SEEDS = (3106923666, 703554134)
DEVELOPMENT_EVALUATION_SEEDS = (
    4193231370, 2203172137, 998548770, 4190405223,
    3607247798, 1802997917, 122805917, 3690010267,
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
# Responsibility transfer is held fixed in this mechanism-isolation screen.
# It must not worsen; strict improvement is required only for raw behavior.
MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION = 0.0
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
MINIMUM_STRICT_RESPONSIBILITY_IMPROVEMENT_CONDITIONS = 0
MINIMUM_STRICT_RAW_IMPROVEMENT_CONDITIONS = 1
SELECTION_CONFIDENCE = 0.90
BOOTSTRAP_DRAWS = 20_000

DEVELOPMENT_DISCLOSURE = (
    "This v14.4 screen was designed after inspecting the failed v14.3 fixed-"
    "strength development screen. It tests whether exposing router strength "
    "and increasing it between training rollouts avoids early optimizer-path "
    "divergence while held-out behavior is evaluated only at the frozen target. "
    "Every arm has the same policy-state dimension and causal responsibility "
    "mode; the direct baseline observes zero strength. The constant-strength "
    "arm is a fresh replication anchor, not a reuse of v14.3 outcomes. No "
    "primal-dual arm is included. Responsibility drift is a no-worsening gate "
    "because responsibility transfer is held fixed; raw physical drift must "
    "strictly improve. At-floor conditions count only as absolute "
    "noninferiority. This is development evidence; "
    "any selected arm requires a new frozen v15 confirmation."
)


def expected_router_training_strengths(arm_name: str) -> tuple[float, ...]:
    arm = ARMS[str(arm_name)]
    target = float(arm["lower_action_router_strength"])
    schedule = str(arm["lower_action_router_training_schedule"])
    warmup = float(arm["lower_action_router_warmup_fraction"])
    ramp = float(arm["lower_action_router_ramp_fraction"])
    strengths: list[float] = []
    for iteration in range(ITERATIONS):
        if schedule == "constant" or target == 0.0:
            strengths.append(target)
            continue
        progress = float(iteration + 1) / float(ITERATIONS)
        phase = min(max((progress - warmup) / ramp, 0.0), 1.0)
        if schedule == "delayed_cosine":
            phase = 0.5 - 0.5 * math.cos(math.pi * phase)
        strengths.append(target * phase)
    return tuple(strengths)


def validate_frozen_design() -> None:
    sequences = {
        "optimizer": OPTIMIZER_SEEDS,
        "training": TRAIN_SEEDS,
        "checkpoint_selection": CHECKPOINT_SELECTION_SEEDS,
        "development_evaluation": DEVELOPMENT_EVALUATION_SEEDS,
    }
    for name, values in sequences.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"invalid v14.4 development seed registry: {name}")
    roles = {name: set(values) for name, values in sequences.items()}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if roles[left] & roles[right]:
                raise ValueError(
                    f"v14.4 development seed roles overlap: {left}/{right}"
                )
    if len(OPTIMIZER_SEEDS) != 16:
        raise ValueError("v14.4 screen requires sixteen optimizer replicates")
    if len(ARMS) != 8 or BASELINE_ARM not in ARMS:
        raise ValueError("v14.4 algorithm-screen arm registry drifted")
    for name, arm in ARMS.items():
        if (
            str(arm["method"]) != "freq_hrl_no_leakage"
            or str(arm["upper_constraint_mode"]) != "disabled"
            or float(arm["upper_dual_lr"]) != 0.0
            or float(arm["lower_dual_lr"]) != 0.0
            or str(arm["checkpoint_score_mode"]) != "mean_reward"
        ):
            raise ValueError(f"v14.4 reward-only arm is invalid: {name}")
        if not bool(arm["lower_action_router_observe_strength"]):
            raise ValueError(f"v14.4 arm hides router strength: {name}")
        if str(arm["responsibility_mode"]) != "causal_lf_transfer":
            raise ValueError(f"v14.4 responsibility mode drifted: {name}")
        if name == BASELINE_ARM:
            if (
                str(arm["lower_action_router_mode"]) != "direct"
                or float(arm["lower_action_router_strength"]) != 0.0
                or str(arm["lower_action_router_training_schedule"])
                != "constant"
            ):
                raise ValueError("v14.4 direct baseline is invalid")
        else:
            strength = float(arm["lower_action_router_strength"])
            ideal_dc_reduction = 1.0 - (1.0 - strength) ** 2
            if (
                str(arm["lower_action_router_mode"])
                != "causal_ema_high_pass"
                or ideal_dc_reduction
                < MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION
            ):
                raise ValueError(f"v14.4 target router is too weak: {name}")
            schedule = str(arm["lower_action_router_training_schedule"])
            warmup = float(arm["lower_action_router_warmup_fraction"])
            ramp = float(arm["lower_action_router_ramp_fraction"])
            if schedule == "constant":
                if warmup != 0.0 or ramp != 0.0:
                    raise ValueError("v14.4 constant schedule has a ramp")
            elif schedule not in ("delayed_linear", "delayed_cosine"):
                raise ValueError(f"v14.4 schedule is invalid: {name}")
            elif ramp <= 0.0 or warmup + ramp > 1.0:
                raise ValueError(f"v14.4 ramp is invalid: {name}")
        strengths = expected_router_training_strengths(name)
        if len(strengths) != ITERATIONS:
            raise ValueError(f"v14.4 strength schedule drifted: {name}")
        if any(value < 0.0 or value > 1.0 for value in strengths):
            raise ValueError(f"v14.4 strength schedule is out of bounds: {name}")
        if abs(strengths[-1] - float(
            arm["lower_action_router_strength"]
        )) > 1e-12:
            raise ValueError(f"v14.4 schedule misses its target: {name}")
    if not 0.0 < DRIFT_MATERIALITY_FLOOR < LOWER_LF_RMS_BUDGET ** 2:
        raise ValueError("v14.4 materiality floor must be within the LF budget")


validate_frozen_design()
