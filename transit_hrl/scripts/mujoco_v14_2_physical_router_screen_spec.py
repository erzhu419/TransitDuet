"""Frozen development design for the MuJoCo v14.2 mechanism screen."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v14_2_physical_router_screen_v1"
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_2_physical_cost_action_router"
)
FROZEN_ALGORITHM_REVISION = "1eeb36dbb0807fdcd54fef08ece456fd89fb9afc"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "966f217a0414a7dc9a035855be33e38597eb8344c221ad71f0a8cf136a95b7f2"
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
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "crossed_direct_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="direct",
        router_alpha=0.10,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "crossed_router_a004_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "crossed_router_a010_reward": _arm(
        method="freq_hrl_no_leakage",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.10,
        upper_constraint_mode="disabled",
        upper_dual_lr=0.0,
        lower_dual_lr=0.0,
        checkpoint_score_mode="mean_reward",
        checkpoint_constraint_penalty=0.0,
    ),
    "crossed_direct_pd_u2_l8": _arm(
        method="freq_hrl",
        responsibility_mode="causal_lf_transfer",
        router_mode="direct",
        router_alpha=0.10,
        upper_constraint_mode="primal_dual",
        upper_dual_lr=2.0,
        lower_dual_lr=8.0,
        checkpoint_score_mode="behavior_robust",
        checkpoint_constraint_penalty=0.10,
    ),
    "crossed_router_a004_pd_u2_l8": _arm(
        method="freq_hrl",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.04,
        upper_constraint_mode="primal_dual",
        upper_dual_lr=2.0,
        lower_dual_lr=8.0,
        checkpoint_score_mode="behavior_robust",
        checkpoint_constraint_penalty=0.10,
    ),
    "crossed_router_a010_pd_u0p5_l2": _arm(
        method="freq_hrl",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.10,
        upper_constraint_mode="primal_dual",
        upper_dual_lr=0.5,
        lower_dual_lr=2.0,
        checkpoint_score_mode="behavior_robust",
        checkpoint_constraint_penalty=0.10,
    ),
    "crossed_router_a010_pd_u2_l8": _arm(
        method="freq_hrl",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.10,
        upper_constraint_mode="primal_dual",
        upper_dual_lr=2.0,
        lower_dual_lr=8.0,
        checkpoint_score_mode="behavior_robust",
        checkpoint_constraint_penalty=0.10,
    ),
    "crossed_router_a010_pd_u8_l32": _arm(
        method="freq_hrl",
        responsibility_mode="causal_lf_transfer",
        router_mode="causal_ema_high_pass",
        router_alpha=0.10,
        upper_constraint_mode="primal_dual",
        upper_dual_lr=8.0,
        lower_dual_lr=32.0,
        checkpoint_score_mode="behavior_robust",
        checkpoint_constraint_penalty=0.10,
    ),
}
BASELINE_ARM = "additive_reward_baseline"
CANDIDATE_ARMS = tuple(arm for arm in ARMS if arm != BASELINE_ARM)

# Fresh development-only namespace. A confirmatory v15 must use new values.
OPTIMIZER_SEEDS = (
    3320051508, 2388055573, 918487018, 943209995,
    3709299629, 1782700034, 3480822045, 1450781257,
    2903124363, 4004180011, 2509403045, 3766125187,
    3611399670, 3498525233, 3133204500, 3286106997,
)
TRAIN_SEEDS = (3057363066, 2079494737, 1689331277, 3106901212)
CHECKPOINT_SELECTION_SEEDS = (2542458127, 4258299041)
DEVELOPMENT_EVALUATION_SEEDS = (
    2341729786, 552267050, 2595304585, 692283048,
    95319261, 2606488110, 1558473317, 3541213198,
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
    "This v14.2 screen was designed after inspecting the failed v14.1 "
    "development screen. It jointly tests physical power-excess costs, a "
    "causal observable lower-action router, router and constraint ablations, "
    "and non-saturating dual rates on fresh development seeds. At-floor "
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
            raise ValueError(f"invalid v14.2 development seed registry: {name}")
    roles = {name: set(values) for name, values in sequences.items()}
    names = list(roles)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if roles[left] & roles[right]:
                raise ValueError(
                    f"v14.2 development seed roles overlap: {left}/{right}"
                )
    if len(OPTIMIZER_SEEDS) != 16:
        raise ValueError("v14.2 screen requires sixteen optimizer replicates")
    if len(ARMS) != 9 or BASELINE_ARM not in ARMS:
        raise ValueError("v14.2 algorithm-screen arm registry drifted")
    for name, arm in ARMS.items():
        constrained = str(arm["method"]) == "freq_hrl"
        if constrained != (str(arm["upper_constraint_mode"]) == "primal_dual"):
            raise ValueError(f"v14.2 constraint contract drifted: {name}")
        if constrained and (
            float(arm["upper_dual_lr"]) <= 0.0
            or float(arm["lower_dual_lr"]) <= 0.0
            or str(arm["leakage_cost_mode"]) != "power_excess"
        ):
            raise ValueError(f"v14.2 physical dual arm is invalid: {name}")
    if not 0.0 < DRIFT_MATERIALITY_FLOOR < LOWER_LF_RMS_BUDGET ** 2:
        raise ValueError("v14.2 materiality floor must be within the LF budget")


validate_frozen_design()
