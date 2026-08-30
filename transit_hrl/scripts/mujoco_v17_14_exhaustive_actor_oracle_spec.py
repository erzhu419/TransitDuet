"""Frozen design for the v17.14 exhaustive actor-adapter oracle audit."""

from __future__ import annotations

from scripts import mujoco_v17_13_causal_actor_adapter_spec as v17_13


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_14_exhaustive_actor_oracle_v1"
EVIDENCE_ROLE = (
    "exhaustive_reused_path_full_grid_exact_oracle_not_confirmatory"
)
FROZEN_CORE_REVISION = "5c382979eeffaf7fde19be99835ee0ddc9e9b986"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "32dadb19d67f9b5bea6be95043d01a36ef1f4a6d39df3bdb8925c781d7f4b41d"
)
SOURCE_DATASET_RUN = v17_13.SOURCE_DATASET_RUN
SOURCE_TARGET_RUN = v17_13.SOURCE_TARGET_RUN
SOURCE_V17_13_SELECTION_RUN = (
    "mujoco_v17_13_causal_actor_adapter_selection_20260831_r1"
)
EXPECTED_FULL_GRID_CANDIDATE_COUNT = (
    len(v17_13.FIR_WINDOWS)
    * len(v17_13.RIDGE_PENALTIES)
    * len(v17_13.ACTOR_FLOOR_PATH_WEIGHTS)
    * len(v17_13.OUTPUT_GAINS)
    * len(v17_13.CORRECTION_ABS_LIMITS)
)
EXPECTED_V17_13_ORACLE_CANDIDATE_COUNT = 48
EXPECTED_REMAINDER_CANDIDATE_COUNT = (
    EXPECTED_FULL_GRID_CANDIDATE_COUNT
    - EXPECTED_V17_13_ORACLE_CANDIDATE_COUNT
)
PROGRESS_INTERVAL = 25

SELECTION_CONTRACT = {
    "input": (
        "the unchanged server-only v17.8 paths, seven v17.12 targets, and "
        "registered v17.13 exact-oracle candidate summaries"
    ),
    "candidate_set": (
        "all 852 members of the frozen v17.13 900-grid that did not receive a "
        "v17.13 exact oracle; no outcome-adaptive prefilter and no new "
        "hyperparameter"
    ),
    "evaluation": (
        "eight grouped leave-one-seed-out causal predictions followed by exact "
        "full-horizon responsibility oracles on every remainder candidate"
    ),
    "combined_frontier": (
        "the 852 new summaries are combined with the 48 registered v17.13 "
        "summaries so all 900 frozen candidates receive exact oracle outcomes"
    ),
    "selection_order": (
        "corrected feasible path count, actor-floor recovery, reference-"
        "feasible preservation, complete floor seed groups, target fidelity, "
        "reference correction, correction magnitude, then candidate id"
    ),
    "advancement_gate": v17_13.SELECTION_CONTRACT["advancement_gate"],
    "fresh_access_rule": (
        "no fresh path is read; a complete gate only authorizes a separately "
        "frozen closed-loop fresh validation"
    ),
    "failure_rule": (
        "if no candidate passes, close this frozen 900-member linear causal "
        "FIR adapter design and move to state-conditioned policy learning"
    ),
    "claim_boundary": (
        "exhaustive full-grid reused-path development only; no reward, online "
        "learning, fresh-seed, or manuscript performance claim"
    ),
}
