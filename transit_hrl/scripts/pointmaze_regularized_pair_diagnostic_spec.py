"""Frozen small-data paired ridge screen with cached neural references."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_regularized_pair_diagnostic import PROTOCOL_VERSION
from scripts.pointmaze_averaged_label_diagnostic_spec import (
    POLICY, SOURCE_PROTOCOL, roots, source_result, endpoint_result, sampling_options,
)


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
RUNNER_SCRIPT = "scripts/run_pointmaze_regularized_pair_diagnostic.py"
CONTINUATION = "cached_stage18_futures; shared_linear_value_difference; mean_mse_plus_unit_l2"
EVIDENCE_ROLE = "cached_regularized_pair_diagnostic_only"


def baseline_result(root, *, preflight):
    if root not in roots(preflight=preflight):
        raise ValueError("Stage-20 root is not registered")
    phase = "preflight" if preflight else "development"
    return (Path(__file__).resolve().parents[1] / "results"
            / f"pointmaze_averaged_label_stage19_v1_{phase}_20260927_r1"
            / "cells" / POLICY / f"replicate_{root}" / "result.json")


def input_results(root, *, preflight):
    return {"source_result": source_result(root, preflight=preflight),
            "endpoint_result": endpoint_result(root, preflight=preflight),
            "baseline_result": baseline_result(root, preflight=preflight)}
