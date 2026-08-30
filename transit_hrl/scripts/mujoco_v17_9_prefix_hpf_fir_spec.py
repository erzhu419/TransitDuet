"""Frozen development design for MuJoCo v17.9 prefix-HPF FIR routing."""

from __future__ import annotations

from scripts import (  # noqa: F401
    mujoco_v17_8_causal_fir_distillation_spec as v17_8,
)


DEVELOPMENT_PROTOCOL_VERSION = "mujoco_v17_9_prefix_hpf_fir_v1"
EVIDENCE_ROLE = (
    "post_v17_8_reused_path_prefix_hpf_projection_not_confirmatory"
)
FROZEN_CORE_REVISION = "85dc42eaa1518727d6975d8c09faf1345763f28a"
FROZEN_SOURCE_MANIFEST_SHA256 = (
    "24d5649b51b7d2ce30d20c7a4b991f70809ee8a92586c20c30d321a3032a44e2"
)
ENVIRONMENTS = v17_8.ENVIRONMENTS
DISTURBANCE_MODES = v17_8.DISTURBANCE_MODES
REUSED_SELECTION_SEEDS = v17_8.REUSED_SELECTION_SEEDS
FRESH_VALIDATION_SEEDS = v17_8.FRESH_VALIDATION_SEEDS
UPPER_RMS_BUDGET = v17_8.UPPER_RMS_BUDGET
LOWER_RMS_BUDGET = v17_8.LOWER_RMS_BUDGET
UPPER_ACTION_LIMIT = v17_8.UPPER_ACTION_LIMIT
LOWER_ACTION_LIMIT = v17_8.LOWER_ACTION_LIMIT
UPPER_WINDOW = v17_8.UPPER_WINDOW
LOWER_WINDOW = v17_8.LOWER_WINDOW
POWER_TOLERANCE = v17_8.POWER_TOLERANCE
RECONSTRUCTION_TOLERANCE = v17_8.RECONSTRUCTION_TOLERANCE
BOUND_TOLERANCE = v17_8.BOUND_TOLERANCE
FEATURE_SCALE_FLOOR = v17_8.FEATURE_SCALE_FLOOR
REUSED_EXPECTED_PATH_COUNT = v17_8.REUSED_EXPECTED_PATH_COUNT

# The post-v17.8 mechanism keeps gain one and changes only the object being
# constrained. Width/ridge values are the strong v17.8 region plus shorter
# controls; no new trace or fresh seed was accessed before freezing this grid.
FIR_WINDOWS = (24, 32, 48, 64)
RIDGE_PENALTIES = (1e-5, 1e-3)
OUTPUT_GAIN = 1.0

REUSED_RECOVERY_GATE_TOTAL = v17_8.REUSED_RECOVERY_GATE_TOTAL
REUSED_EXPECTED_ORACLE_RECOVERABLE_FAILURES = (
    v17_8.REUSED_EXPECTED_ORACLE_RECOVERABLE_FAILURES
)
REUSED_RECOVERY_GATE_BY_ENVIRONMENT = (
    v17_8.REUSED_RECOVERY_GATE_BY_ENVIRONMENT
)
REUSED_PRESERVE_BASELINE_FEASIBLE_WALKER_GATE = (
    v17_8.REUSED_PRESERVE_BASELINE_FEASIBLE_WALKER_GATE
)
REUSED_EXPECTED_BASELINE_FEASIBLE_WALKER_PATHS = (
    v17_8.REUSED_EXPECTED_BASELINE_FEASIBLE_WALKER_PATHS
)
FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL = (
    v17_8.FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL
)
FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL_BY_ENVIRONMENT = (
    v17_8.FRESH_MINIMUM_ORACLE_FEASIBLE_RECALL_BY_ENVIRONMENT
)
FRESH_MINIMUM_BASELINE_FEASIBLE_PRESERVATION = (
    v17_8.FRESH_MINIMUM_BASELINE_FEASIBLE_PRESERVATION
)

SELECTION_CONTRACT = {
    "input": (
        "the same 120 server-only reused paths and grouped oracle labels used "
        "by v17.8; no fresh path access"
    ),
    "router": (
        "gain-one multivariate causal FIR followed by physical box projection "
        "and causal projection of only the upper HPF8 innovation onto the "
        "registered prefix energy budget"
    ),
    "grouping": (
        "eight leave-one-seed-out folds with all five held-seed disturbance "
        "modes excluded from fitting"
    ),
    "selection": (
        "one shared width and ridge id across environments under the inherited "
        "v17.8 lexicographic order and advancement gate"
    ),
    "claim_boundary": (
        "post-failure reused-path mechanism development only; no fresh-seed, "
        "reward, closed-loop learning, or manuscript performance claim"
    ),
}
