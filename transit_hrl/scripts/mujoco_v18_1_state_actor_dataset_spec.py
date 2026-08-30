"""Frozen reused-path state-trace design for MuJoCo v18.1."""

from __future__ import annotations

from scripts import (  # noqa: F401
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v18_1_state_actor_dataset_v1"
EVIDENCE_ROLE = (
    "reused_path_causal_actor_state_export_not_model_selection_or_confirmatory"
)
FROZEN_CORE_REVISION = "f94f1f4a6a35d70f6b6d144bd886644e7efb2393"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "b06a97fc8f18129a2e1a9c23a52acb01ea683d1445c672074b9a6901ece23af6"
)
SOURCE_RUN_NAME = (
    "mujoco_v17_4_streaming_audit_projection_preflight_20260831_r1"
)
ENVIRONMENTS = v17_4.ENVIRONMENTS
DISTURBANCE_MODES = v17_4.EVALUATION_DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v17_4.EVALUATION_SEEDS
OPTIMIZER_SEED = v17_4.OPTIMIZER_SEEDS[0]

EXPECTED_ACTION_DIMENSION = {
    "HalfCheetah-v5": 6,
    "Hopper-v5": 3,
    "Walker2d-v5": 6,
}
EXPECTED_OBSERVATION_DIMENSION = {
    "HalfCheetah-v5": 17,
    "Hopper-v5": 11,
    "Walker2d-v5": 17,
}
EXPECTED_LOWER_STATE_DIMENSION = {
    "HalfCheetah-v5": 265,
    "Hopper-v5": 136,
    "Walker2d-v5": 265,
}
EXPECTED_PATH_COUNT = (
    len(ENVIRONMENTS)
    * len(DISTURBANCE_MODES)
    * len(REUSED_SELECTION_SEEDS)
)

TRACE_KEYS = (
    "baseline_lower_action",
    "baseline_upper_action",
    "disturbance",
    "episode_step",
    "executed_action",
    "latent_lower_action",
    "lower_policy_state",
    "observation",
    "total_action",
    "upper_decision",
    "upper_policy_action",
)

DATASET_CONTRACT = {
    "source": (
        "exact deterministic replay of the frozen v17.4 checkpoint on the "
        "unchanged 120 reused development paths"
    ),
    "causality": (
        "each lower-policy state and observation is captured immediately "
        "before the corresponding action and environment transition"
    ),
    "labels": (
        "no v17.12 target or actor-floor label is read or stored during state "
        "export; labels remain a separate server-only selection input"
    ),
    "storage": (
        "NPZ traces remain server-only on node003; only compact path markers "
        "are synchronized locally"
    ),
    "claim_boundary": (
        "reused-path causal state export only; no model selection, reward, "
        "fresh-seed, online-learning, or manuscript performance claim"
    ),
}
