"""Frozen development design for MuJoCo v17.8 causal FIR distillation."""

from __future__ import annotations

from scripts import (  # noqa: F401
    mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4,
)


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_8_causal_fir_distillation_v1"
EVIDENCE_ROLE = (
    "reused_path_grouped_selection_then_fresh_path_validation_not_confirmatory"
)
FROZEN_CORE_REVISION = "a120651572e7a35614527bd2be18bd3b52f0c14f"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "e6819fe80ae428755ffd355dbb8c22eece71a7dff9a7da8c750b8029d4b072c7"
)
SOURCE_RUN_NAME = (
    "mujoco_v17_4_streaming_audit_projection_preflight_20260831_r1"
)
ENVIRONMENTS = v17_4.ENVIRONMENTS
DISTURBANCE_MODES = v17_4.EVALUATION_DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v17_4.EVALUATION_SEEDS

# Derived before fresh-path access with
# derive_seed("mujoco_v17_8_fresh_router_validation_v1", index), index=0..7.
FRESH_VALIDATION_SEEDS = (
    2969266561,
    1060853697,
    1705453152,
    1911126157,
    3726666952,
    2647745800,
    3002649567,
    2889178607,
)
OPTIMIZER_SEED = v17_4.OPTIMIZER_SEEDS[0]
UPPER_RMS_BUDGET = v17_4.UPPER_HF_RMS_BUDGET
LOWER_RMS_BUDGET = v17_4.LOWER_LF_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_4.UPPER_ACTION_SCALE
LOWER_ACTION_LIMIT = v17_4.LOWER_ACTION_SCALE
UPPER_WINDOW = 8
LOWER_WINDOW = 32
POWER_TOLERANCE = 1e-8
RECONSTRUCTION_TOLERANCE = 1e-7
BOUND_TOLERANCE = 1e-7

# One candidate id applies across all environments. Coefficients are fitted
# separately because MuJoCo action dimensions differ, while feature, fitting,
# selection, and validation semantics remain shared.
FIR_WINDOWS = (16, 24, 32, 48, 64)
RIDGE_PENALTIES = (1e-5, 1e-3, 1e-1, 1.0)
OUTPUT_GAINS = (0.80, 0.90, 1.00, 1.10)
FEATURE_SCALE_FLOOR = 1e-8

REUSED_EXPECTED_PATH_COUNT = (
    len(ENVIRONMENTS)
    * len(DISTURBANCE_MODES)
    * len(REUSED_SELECTION_SEEDS)
)
FRESH_EXPECTED_PATH_COUNT = (
    len(ENVIRONMENTS)
    * len(DISTURBANCE_MODES)
    * len(FRESH_VALIDATION_SEEDS)
)

# The reused panel gate is inherited from v17.7 and evaluated only from
# seed-grouped out-of-fold predictions.
REUSED_RECOVERY_GATE_TOTAL = 65
REUSED_EXPECTED_ORACLE_RECOVERABLE_FAILURES = 81
REUSED_RECOVERY_GATE_BY_ENVIRONMENT = {
    "HalfCheetah-v5": 32,
    "Hopper-v5": 24,
    "Walker2d-v5": 6,
}
REUSED_PRESERVE_BASELINE_FEASIBLE_WALKER_GATE = 30
REUSED_EXPECTED_BASELINE_FEASIBLE_WALKER_PATHS = 32

# Fresh validation is conditional on the independently computed full-horizon
# oracle because some total-action paths are physically irreducible by routing.
FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL = 0.75
FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL_BY_ENVIRONMENT = 0.60
FRESH_MINIMUM_BASELINE_FEASIBLE_PRESERVATION = 0.90

SELECTION_CONTRACT = {
    "training_unit": (
        "one environment-specific multivariate causal FIR coefficient tensor "
        "fitted to full-horizon oracle upper actions"
    ),
    "grouping": (
        "eight leave-one-evaluation-seed-out folds; all five disturbance "
        "modes for the held seed remain outside that fold's fit"
    ),
    "shared_candidate": (
        "one window, ridge penalty, and output gain selected jointly for all "
        "three environments"
    ),
    "selection_order": (
        "numerical validity and upper-budget count; worst-environment oracle-"
        "recoverable recall; total recovery; baseline-feasible preservation; "
        "worst-environment mean lower-power ratio; overall mean lower power; "
        "candidate id"
    ),
    "fresh_access_rule": (
        "fit the selected candidate on all reused paths and access the frozen "
        "fresh seeds only if every reused-panel advancement condition passes"
    ),
    "claim_boundary": (
        "fixed-total-action responsibility attribution only; no reward, "
        "closed-loop learning, or manuscript performance claim"
    ),
}
