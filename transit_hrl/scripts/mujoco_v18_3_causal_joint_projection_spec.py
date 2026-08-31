"""Frozen reused-path design for MuJoCo v18.3 joint projection."""

from __future__ import annotations

from scripts import mujoco_v17_13_causal_actor_adapter_spec as v17_13


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v18_3_causal_joint_projection_v1"
EVIDENCE_ROLE = (
    "reused_path_label_free_causal_joint_projection_not_confirmatory"
)
FROZEN_CORE_REVISION = "212c571e630512d37682f9984727b7f4016740e8"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "11a7a0087c9369e744b6516cd7f6571a9f2b0562eae649f7d5d89cf62052a3a2"
)

REFERENCE_DATASET_RUN = "mujoco_v17_8_causal_fir_dataset_20260831_r1"
ENVIRONMENTS = v17_13.ENVIRONMENTS
DISTURBANCE_MODES = v17_13.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v17_13.REUSED_SELECTION_SEEDS
EXPECTED_PATH_COUNT = v17_13.EXPECTED_PATH_COUNT
EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT = (
    v17_13.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
)
EXPECTED_ACTOR_FLOOR_PATH_COUNT = v17_13.EXPECTED_ACTOR_FLOOR_PATH_COUNT
EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED = (
    v17_13.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
)

UPPER_RMS_BUDGET = v17_13.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v17_13.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_13.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v17_13.LOWER_ACTION_LIMIT
UPPER_WINDOW = v17_13.UPPER_WINDOW
LOWER_WINDOW = v17_13.LOWER_WINDOW
POWER_TOLERANCE = v17_13.POWER_TOLERANCE
EXECUTED_ACTION_LIMIT = v17_13.EXECUTED_ACTION_LIMIT
RECONSTRUCTION_TOLERANCE = 1e-7
BOUND_TOLERANCE = 1e-7
CORRECTION_ABS_MAX_GATE = 0.05
REFERENCE_CORRECTION_RMS_MAX_GATE = 0.01
ACTOR_FLOOR_CORRECTION_RMS_MAX_GATE = 0.015
EXECUTED_CORRECTION_RMS_MIN_GATE = (
    v17_13.EXECUTED_CORRECTION_RMS_MIN_GATE
)

CANDIDATES = {
    "joint_projection_instantaneous": {
        "budget_mode": "instantaneous",
    },
    "joint_projection_prefix_ledger": {
        "budget_mode": "prefix_ledger",
    },
}

SELECTION_CONTRACT = {
    "input": (
        "unchanged v17.8 baseline upper/lower/total action traces; no v17.12 "
        "actor target, actor-floor label, observation, reward, or future action "
        "is available to the projector"
    ),
    "causality": (
        "at step t the projector uses only current proposed components, past "
        "projected components, registered causal HPF8/LPF32 definitions, and "
        "fixed budgets"
    ),
    "lexicographic_action_rule": (
        "preserve the proposed total action whenever a causal feasible split "
        "exists; otherwise project the current upper and lower components to "
        "their nearest causal feasible sets"
    ),
    "candidate_set": (
        "instantaneous per-step residual budgets and cumulative prefix-energy "
        "ledger budgets; no fitted parameter or data-selected threshold"
    ),
    "selection_order": (
        "valid path count, direct joint-feasible count, exact-oracle feasible "
        "count, actor-floor recovery, reference preservation, lower maximum "
        "reference correction, lower maximum actor-floor correction, then id"
    ),
    "advancement_gate": (
        "120/120 valid and directly joint feasible, 120/120 exact-oracle "
        "feasible, 113/113 reference paths preserved, all seven actor-floor "
        "paths and both seed groups recovered, nonzero executed correction on "
        "all floor paths, total correction absolute maximum at most 0.05, "
        "reference correction RMS maximum at most 0.01, and actor-floor "
        "correction RMS maximum at most 0.015"
    ),
    "fresh_access_rule": (
        "the reused-path screen never reads a fresh path; a complete pass only "
        "authorizes a separately committed closed-loop validation freeze"
    ),
    "claim_boundary": (
        "label-free causal mechanism selection on reused paths only; no reward, "
        "fresh-seed, learned-policy, no-tradeoff, or manuscript claim"
    ),
}
