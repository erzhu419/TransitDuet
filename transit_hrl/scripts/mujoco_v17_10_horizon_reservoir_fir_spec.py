"""Frozen development design for MuJoCo v17.10 horizon-reservoir FIR."""

from __future__ import annotations

from scripts import mujoco_v17_9_prefix_hpf_fir_spec as v17_9  # noqa: F401


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_10_horizon_reservoir_fir_v1"
EVIDENCE_ROLE = (
    "post_v17_9_reused_path_horizon_reservoir_selection_not_confirmatory"
)
FROZEN_CORE_REVISION = "f849d15c0b8c7f8c0f99e0bdf69f9b892d20da36"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "9e78071e94f6bba8c589fb765dc4378ad9be7268662defd91e9bfc231457c892"
)
ENVIRONMENTS = v17_9.ENVIRONMENTS
DISTURBANCE_MODES = v17_9.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v17_9.REUSED_SELECTION_SEEDS
FRESH_VALIDATION_SEEDS = v17_9.FRESH_VALIDATION_SEEDS
UPPER_RMS_BUDGET = v17_9.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v17_9.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_9.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v17_9.LOWER_ACTION_LIMIT
UPPER_WINDOW = v17_9.UPPER_WINDOW
LOWER_WINDOW = v17_9.LOWER_WINDOW
POWER_TOLERANCE = v17_9.POWER_TOLERANCE
RECONSTRUCTION_TOLERANCE = v17_9.RECONSTRUCTION_TOLERANCE
BOUND_TOLERANCE = v17_9.BOUND_TOLERANCE
FEATURE_SCALE_FLOOR = v17_9.FEATURE_SCALE_FLOOR
REUSED_EXPECTED_PATH_COUNT = v17_9.REUSED_EXPECTED_PATH_COUNT

FIR_WINDOWS = (48, 64)
RIDGE_PENALTIES = (1e-5, 1e-3)
OUTPUT_GAIN = 1.0
ENERGY_RESERVE_STEPS = (0, 16, 32, 48, 64, 72, 80, 82)

REUSED_RECOVERY_GATE_TOTAL = v17_9.REUSED_RECOVERY_GATE_TOTAL
REUSED_EXPECTED_ORACLE_RECOVERABLE_FAILURES = (
    v17_9.REUSED_EXPECTED_ORACLE_RECOVERABLE_FAILURES
)
REUSED_RECOVERY_GATE_BY_ENVIRONMENT = (
    v17_9.REUSED_RECOVERY_GATE_BY_ENVIRONMENT
)
REUSED_PRESERVE_BASELINE_FEASIBLE_WALKER_GATE = (
    v17_9.REUSED_PRESERVE_BASELINE_FEASIBLE_WALKER_GATE
)
REUSED_EXPECTED_BASELINE_FEASIBLE_WALKER_PATHS = (
    v17_9.REUSED_EXPECTED_BASELINE_FEASIBLE_WALKER_PATHS
)
FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL = (
    v17_9.FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL
)
FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL_BY_ENVIRONMENT = (
    v17_9.FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL_BY_ENVIRONMENT
)
FRESH_MINIMUM_BASELINE_FEASIBLE_PRESERVATION = (
    v17_9.FRESH_MINIMUM_BASELINE_FEASIBLE_PRESERVATION
)

SELECTION_CONTRACT = {
    "input": (
        "the unchanged v17.8 server-only reused-path panel; no fresh path "
        "access before a complete gate pass"
    ),
    "router": (
        "gain-one causal multivariate FIR with physical boxes and an upper "
        "HPF8 energy envelope max(t+1, H_min) times the per-step budget"
    ),
    "certification": (
        "endpoint upper-budget certification is valid only when realized "
        "trajectory length is at least the selected H_min; shorter paths fail"
    ),
    "grouping": (
        "eight leave-one-seed-out folds with every held-seed disturbance mode "
        "excluded from fitting"
    ),
    "claim_boundary": (
        "post-v17.9 reused-path mechanism development only; no fresh-seed, "
        "reward, closed-loop learning, or manuscript performance claim"
    ),
}
