"""Frozen causal decision screen using existing branch and future caches."""

from freq_hrl.experiments.pointmaze_cost_sensitive_decision import PROTOCOL_VERSION
from scripts.pointmaze_ridge_error_diagnostic_spec import POLICY, roots, sampling_options
from scripts.pointmaze_ridge_error_diagnostic_spec import input_results as cached_sources


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_state_noise_stage18_v1_diagnostic"
RUNNER_SCRIPT = "scripts/run_pointmaze_cost_sensitive_decision.py"
CONTINUATION = "frozen_reference; causal_now_wait; cost_weighted_logistic; whole_path_holdout"
EVIDENCE_ROLE = "cached_causal_decision_development_only"


def input_results(root, *, preflight):
    sources = cached_sources(root, preflight=preflight)
    return {k: sources[k] for k in ("source_result", "endpoint_result", "fresh_result")}
