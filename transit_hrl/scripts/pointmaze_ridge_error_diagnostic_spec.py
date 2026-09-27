"""Cached error attribution without modifying the frozen predictor."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_ridge_error_diagnostic import PROTOCOL_VERSION
from scripts.pointmaze_averaged_label_diagnostic_spec import POLICY, roots, source_result, endpoint_result
from scripts.pointmaze_fresh_future_diagnostic_spec import prediction_result


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_fresh_future_stage21_v1_diagnostic"
RUNNER_SCRIPT = "scripts/run_pointmaze_ridge_error_diagnostic.py"
CONTINUATION = "frozen_stage20_ridge; cached_stage18_and_stage21_labels; retrospective_decomposition"
EVIDENCE_ROLE = "retrospective_frozen_ridge_error_decomposition_only"


def input_results(root, *, preflight):
    phase = "preflight" if preflight else "development"
    return {"source_result": source_result(root, preflight=preflight),
            "endpoint_result": endpoint_result(root, preflight=preflight),
            "prediction_result": prediction_result(root, preflight=preflight),
            "fresh_result": Path(__file__).resolve().parents[1] / "results"
            / f"pointmaze_fresh_future_stage21_v1_{phase}_20260927_r1"
            / "cells" / POLICY / f"replicate_{root}" / "result.json"}


def sampling_options(*, preflight):
    return {"pairs_per_seed": 2 if preflight else 12, "noise_pairs_per_seed": 1 if preflight else 2,
            "future_replicates": 4 if preflight else 8, "fresh_future_replicates": 4 if preflight else 64}
