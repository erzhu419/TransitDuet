"""Frozen design for MuJoCo v17.11 fractional-reservoir FIR routing."""

from __future__ import annotations

from scripts import (  # noqa: F401
    mujoco_v17_10_horizon_reservoir_fir_spec as v17_10,
)


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_11_fractional_reservoir_fir_v1"
EVIDENCE_ROLE = (
    "final_router_only_reused_path_fractional_reservoir_not_confirmatory"
)
FROZEN_CORE_REVISION = "1578e24ecc75bc480f1d41803dc13a19e49b5c5f"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "06557c3f016fc7cbd7cd7f9f4f730f9d6700be7f1dfa07ad0f567ac72e83e8c6"
)
ENVIRONMENTS = v17_10.ENVIRONMENTS
DISTURBANCE_MODES = v17_10.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v17_10.REUSED_SELECTION_SEEDS
FRESH_VALIDATION_SEEDS = v17_10.FRESH_VALIDATION_SEEDS
UPPER_RMS_BUDGET = v17_10.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v17_10.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_10.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v17_10.LOWER_ACTION_LIMIT
UPPER_WINDOW = v17_10.UPPER_WINDOW
LOWER_WINDOW = v17_10.LOWER_WINDOW
POWER_TOLERANCE = v17_10.POWER_TOLERANCE
RECONSTRUCTION_TOLERANCE = v17_10.RECONSTRUCTION_TOLERANCE
BOUND_TOLERANCE = v17_10.BOUND_TOLERANCE
FEATURE_SCALE_FLOOR = v17_10.FEATURE_SCALE_FLOOR
REUSED_EXPECTED_PATH_COUNT = v17_10.REUSED_EXPECTED_PATH_COUNT

FIR_WINDOWS = (48, 64)
RIDGE_PENALTIES = (1e-3,)
OUTPUT_GAIN = 1.0
ENERGY_RESERVE_STEPS = (64, 72, 80, 82)
ENERGY_BORROW_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)

REUSED_RECOVERY_GATE_TOTAL = v17_10.REUSED_RECOVERY_GATE_TOTAL
REUSED_EXPECTED_ORACLE_RECOVERABLE_FAILURES = (
    v17_10.REUSED_EXPECTED_ORACLE_RECOVERABLE_FAILURES
)
REUSED_RECOVERY_GATE_BY_ENVIRONMENT = (
    v17_10.REUSED_RECOVERY_GATE_BY_ENVIRONMENT
)
REUSED_PRESERVE_BASELINE_FEASIBLE_WALKER_GATE = (
    v17_10.REUSED_PRESERVE_BASELINE_FEASIBLE_WALKER_GATE
)
REUSED_EXPECTED_BASELINE_FEASIBLE_WALKER_PATHS = (
    v17_10.REUSED_EXPECTED_BASELINE_FEASIBLE_WALKER_PATHS
)
FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL = (
    v17_10.FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL
)
FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL_BY_ENVIRONMENT = (
    v17_10.FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL_BY_ENVIRONMENT
)
FRESH_MINIMUM_BASELINE_FEASIBLE_PRESERVATION = (
    v17_10.FRESH_MINIMUM_BASELINE_FEASIBLE_PRESERVATION
)

SELECTION_CONTRACT = {
    "router": (
        "gain-one causal multivariate FIR with physical boxes and upper HPF8 "
        "envelope t+1 + rho times max(H_min-t-1, 0)"
    ),
    "repayment": (
        "rho controls the fraction of remaining horizon credit available now; "
        "all borrowed credit vanishes by H_min"
    ),
    "certification": (
        "every path must remain envelope-feasible and reach H_min; endpoint "
        "upper and inherited lower/recovery gates remain unchanged"
    ),
    "stopping_rule": (
        "failure ends router-only filtering for this frozen total-action panel "
        "and redirects development to actor-level total-action feasibility"
    ),
    "claim_boundary": (
        "final reused-path router screen only; no fresh-seed, reward, closed-"
        "loop learning, or manuscript performance claim"
    ),
}
