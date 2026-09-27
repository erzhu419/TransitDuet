"""Frozen state/noise diagnostic on the two revealed controller roots."""

from freq_hrl.experiments.pointmaze_state_noise_diagnostic import PROTOCOL_VERSION
from scripts.pointmaze_deployed_pair_diagnostic_spec import POLICY, roots, source_result
from scripts.pointmaze_paired_value_qualification_spec import source_result as endpoint_result


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
RUNNER_SCRIPT = "scripts/run_pointmaze_state_noise_diagnostic.py"
CONTINUATION = "stage12_frozen; equal_shape_history; conditional_future_after_50_steps"
EVIDENCE_ROLE = "state_compression_and_conditional_noise_diagnostic_only"


def input_results(root, *, preflight):
    return {"source_result": source_result(root, preflight=preflight),
            "endpoint_result": endpoint_result(root, preflight=preflight)}


def sampling_options(*, preflight):
    return {"pairs_per_seed": 2 if preflight else 12,
            "noise_pairs_per_seed": 1 if preflight else 2,
            "future_replicates": 4 if preflight else 8}
