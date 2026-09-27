"""Frozen contextual-value development screen using existing caches only."""

from freq_hrl.experiments.pointmaze_contextual_pair_diagnostic import PROTOCOL_VERSION
from scripts.pointmaze_ridge_error_diagnostic_spec import POLICY, roots, input_results, sampling_options


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_fresh_future_stage21_v1_diagnostic"
RUNNER_SCRIPT = "scripts/run_pointmaze_contextual_pair_diagnostic.py"
CONTINUATION = "stage18_training_labels; stage21_scoring_only; bounded_contextual_value; matched_random_context"
EVIDENCE_ROLE = "cached_contextual_pair_development_only"
