"""Frozen development design for the MuJoCo v17.4 streaming projection."""

from __future__ import annotations


DEVELOPMENT_PROTOCOL_VERSION = (
    "mujoco_v17_4_streaming_audit_projection_preflight_v1"
)
EVIDENCE_ROLE = "paired_streaming_audit_projection_development_not_confirmatory"
PREREGISTRATION_STATUS = "frozen_before_v17_4_training_or_heldout_access"
FROZEN_ALGORITHM_REVISION = "91451c1ee0b3bbc488152fc1b4994a3ed5e0436c"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "f371ecffe5182d83c778ba42033ae7f44f50881aa635924ff8f406eaa7927ab6"
)
FROZEN_CORE_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v17_4_streaming_audit_projection"
)
ROUTER_CONTRACT = (
    "causal_realized_total_receding_constant_tail_hpf8_lpf32_projection_"
    "with_complete_fir_state_bounded_components_and_exact_pre_split_"
    "action_execution_v1"
)
SMOOTH_PLAN_CONTRACT = (
    "boundary_sampled_endpoint_exact_c1_smoothstep_primitive_execution_v2"
)
POLICY_STATE_CONTRACT = (
    "complete_right_aligned_upper_hpf8_and_lower_lpf32_fir_histories_"
    "with_valid_counts_independent_of_gauge_strength_v1"
)

ENVIRONMENTS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
EVALUATION_DISTURBANCE_MODES = (*TRAINING_DISTURBANCE_MODES, "ood_chirp")

# Generated once from numpy Generator(17400), then frozen before dispatch.
OPTIMIZER_SEEDS = (3105897127,)
TRAIN_SEEDS = (3713326665, 1307281748, 3265003748, 2195722866)
SELECTION_SEEDS = (394517701, 3530217567, 942310045, 4024287184)
EVALUATION_SEEDS = (
    4009024190,
    2843731921,
    4003206626,
    547164892,
    2802248628,
    2335590642,
    1716353770,
    294864529,
)

ROUTER_ALPHA = 0.20
CONTROL_STRENGTH = 0.0
CANDIDATE_STRENGTH = 1.0
CONTROL_INTERVENTION = "streaming_projection_strength_0_control"
CANDIDATE_INTERVENTION = "streaming_projection_strength_1_candidate"

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
POWER_BUDGET_TOLERANCE = 1e-12
MINIMUM_LOWER_LF_RELATIVE_REDUCTION = 0.10
MINIMUM_JOINT_MERIT_RELATIVE_REDUCTION = 0.10
MINIMUM_UPPER_BUDGET_FEASIBLE_RATE = 0.99
MAXIMUM_UPPER_BUDGET_VIOLATION_RMS = 1e-7
EXPECTED_PATHS_PER_INTERVENTION = (
    len(EVALUATION_SEEDS) * len(EVALUATION_DISTURBANCE_MODES)
)
EXPECTED_EVALUATION_ROWS_PER_CELL = 2 * EXPECTED_PATHS_PER_INTERVENTION
EXPECTED_CELL_COUNT = len(ENVIRONMENTS) * len(OPTIMIZER_SEEDS)
SUPPORTED_STATUS = "streaming_audit_projection_preflight_supported"
NOT_SUPPORTED_STATUS = "streaming_audit_projection_preflight_not_supported"

SELECTION_CONTRACT = {
    "unit": "environment_by_single_fresh_optimizer_seed",
    "training": (
        "one strength-zero policy with complete FIR audit state is trained once "
        "per environment"
    ),
    "paired_intervention": (
        "the same frozen checkpoint and heldout path is evaluated at strengths "
        "zero and one"
    ),
    "mechanics_gate": (
        "all reward, executed-action, and latent-policy traces match exactly; "
        "explicit transition structure is valid; router and responsibility "
        "reconstruction RMS are at most 1e-7"
    ),
    "frequency_gate": (
        "every environment requires candidate upper HPF8 power at most 0.075^2; "
        "candidate lower LPF32 power at most 0.0475^2 and at least 10% below "
        "control; normalized joint merit must improve by at least 10%"
    ),
    "feasibility_gate": (
        "mean current-step upper-budget feasibility is at least 0.99 and maximum "
        "reported upper-budget violation RMS is at most 1e-7"
    ),
    "expansion_rule": (
        "advance to fresh optimizer multiseed only if all three environments pass"
    ),
    "failure_rule": (
        "any mechanics, feasibility, or frequency failure stops expansion"
    ),
    "claim_boundary": (
        "single-optimizer paired development evidence only; no reward, training, "
        "cross-seed, or publication claim"
    ),
}
