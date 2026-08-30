"""Frozen development design for the MuJoCo v17.2 smooth macro gauge."""

from __future__ import annotations

from scripts import mujoco_v17_1_headroom_homotopy_preflight_spec as v17_1


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_2_smooth_macro_gauge_preflight_v1"
EVIDENCE_ROLE = "paired_smooth_macro_gauge_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v17_2_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = "22b77b5ed820276644e6d512ebff9dd8a6b3fe77"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "c7f6a535ff4bd1db0fdf77019882d8d96435e8cead08dc12617a5634fe072b63"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v17_2_smooth_macro_gauge"
)
ROUTER_CONTRACT = (
    "causal_prior_total_low_pass_macro_target_with_frozen_smooth_curve_"
    "bounded_components_and_exact_pre_split_action_execution_v1"
)
SMOOTH_PLAN_CONTRACT = (
    "boundary_sampled_endpoint_exact_c1_smoothstep_primitive_execution_v2"
)
POLICY_STATE_CONTRACT = (
    "causal_total_action_low_pass_context_independent_of_gauge_strength_v1"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from numpy Generator(17200), then frozen before dispatch.
OPTIMIZER_SEEDS = (2548289420,)
TRAIN_SEEDS = (2140296180, 62438583, 2893357541, 3862291383)
SELECTION_SEEDS = (2078100333, 2527831988, 2199604115, 1001230619)
EVALUATION_SEEDS = (
    4109662726,
    2350227856,
    4239705722,
    2492073394,
    21969384,
    47387049,
    1885267174,
    375454845,
)

ALPHA_ARMS = {
    "alpha_005": 0.05,
    "alpha_010": 0.10,
    "alpha_020": 0.20,
}
CONTROL_STRENGTH = 0.0
CANDIDATE_STRENGTH = 1.0
CONTROL_INTERVENTION = "gauge_strength_0_control"
CANDIDATE_INTERVENTION = "gauge_strength_1_candidate"

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

TRACE_MATCH_TOLERANCE = 0
REWARD_ABSOLUTE_TOLERANCE = 1e-9
LATENT_METRIC_ABSOLUTE_TOLERANCE = 1e-12
RECONSTRUCTION_RMS_TOLERANCE = 1e-7
MINIMUM_UPPER_HF_RELATIVE_REDUCTION = 0.10
MINIMUM_LOWER_LF_RELATIVE_REDUCTION = 0.10
MINIMUM_JOINT_MERIT_RELATIVE_REDUCTION = 0.10
MAXIMUM_COMPONENT_CLIP_RATE = 0.25
EXPECTED_PATHS_PER_INTERVENTION = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_EVALUATION_ROWS_PER_CELL = 2 * EXPECTED_PATHS_PER_INTERVENTION
EXPECTED_CELL_COUNT = (
    len(ENVIRONMENTS) * len(ALPHA_ARMS) * len(OPTIMIZER_SEEDS)
)
SUPPORTED_STATUS = "smooth_macro_gauge_preflight_supported"
NOT_SUPPORTED_STATUS = "smooth_macro_gauge_preflight_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_alpha_single_fresh_optimizer_seed",
    "training": (
        "one_reward_only_strength_zero_policy_per_environment_and_alpha"
    ),
    "paired_intervention": (
        "same_frozen_checkpoint_same_heldout_paths_strength_zero_vs_one"
    ),
    "mechanics_gate": (
        "exact_reward_executed_and_latent_trace_identity_exact_additive_"
        "reconstruction_and_bounded_component_projection"
    ),
    "frequency_gate": (
        "at_least_ten_percent_mean_upper_hpf8_lower_lpf32_and_joint_"
        "normalized_merit_reduction_in_all_three_environments"
    ),
    "global_selection": (
        "maximize_worst_environment_joint_merit_reduction_then_median_"
        "joint_merit_reduction_among_eligible_alphas"
    ),
    "outcome_use": (
        "development_only_freeze_one_alpha_for_fresh_leakage_active_"
        "multiseed_training_if_supported"
    ),
}


def validate() -> None:
    roles = (OPTIMIZER_SEEDS, TRAIN_SEEDS, SELECTION_SEEDS, EVALUATION_SEEDS)
    flattened = tuple(seed for values in roles for seed in values)
    consumed = set(
        v17_1.OPTIMIZER_SEEDS
        + v17_1.TRAIN_SEEDS
        + v17_1.SELECTION_SEEDS
        + v17_1.SAFETY_SELECTION_SEEDS
        + v17_1.EVALUATION_SEEDS
    )
    if len(flattened) != 17 or len(set(flattened)) != len(flattened):
        raise RuntimeError("v17.2 requires seventeen disjoint fresh seed roots")
    if consumed & set(flattened):
        raise RuntimeError("v17.2 seed roots overlap v17.1")
    if tuple(ALPHA_ARMS) != ("alpha_005", "alpha_010", "alpha_020"):
        raise RuntimeError("v17.2 alpha registry drifted")
    if any(not 0.0 < float(alpha) <= 1.0 for alpha in ALPHA_ARMS.values()):
        raise RuntimeError("v17.2 alpha values must be in (0, 1]")
    if CONTROL_STRENGTH != 0.0 or CANDIDATE_STRENGTH != 1.0:
        raise RuntimeError("v17.2 paired intervention strengths drifted")
    if EXPECTED_CELL_COUNT != 9 or EXPECTED_PATHS_PER_INTERVENTION != 40:
        raise RuntimeError("v17.2 matrix dimensions drifted")


validate()
