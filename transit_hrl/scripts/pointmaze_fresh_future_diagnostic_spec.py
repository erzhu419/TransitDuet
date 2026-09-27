"""Independent future labels for frozen Stage-20 predictions."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_fresh_future_diagnostic import PROTOCOL_VERSION
from scripts.pointmaze_deployed_pair_diagnostic_spec import POLICY, roots, source_result
from scripts.pointmaze_averaged_label_diagnostic_spec import endpoint_result, source_result as noise_result


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_regularized_pair_stage20_v1_diagnostic"
RUNNER_SCRIPT = "scripts/run_pointmaze_fresh_future_diagnostic.py"
CONTINUATION = "stage12_frozen; stage20_predictions_frozen; independent_conditional_futures"
EVIDENCE_ROLE = "frozen_prediction_independent_future_precision_only"


def prediction_result(root, *, preflight):
    if root not in roots(preflight=preflight):
        raise ValueError("Stage-21 root is not registered")
    phase = "preflight" if preflight else "development"
    return (Path(__file__).resolve().parents[1] / "results"
            / f"pointmaze_regularized_pair_stage20_v1_{phase}_20260927_r1"
            / "cells" / POLICY / f"replicate_{root}" / "result.json")


def input_results(root, *, preflight):
    return {"source_result": source_result(root, preflight=preflight),
            "endpoint_result": endpoint_result(root, preflight=preflight),
            "noise_result": noise_result(root, preflight=preflight),
            "prediction_result": prediction_result(root, preflight=preflight)}


def sampling_options(*, preflight):
    return {"pairs_per_seed": 2 if preflight else 12, "noise_pairs_per_seed": 1 if preflight else 2,
            "cached_future_replicates": 4 if preflight else 8, "future_replicates": 4 if preflight else 64,
            "workers": 2 if preflight else 16}


def task_resources(*, preflight):
    return {"cpu": 2 if preflight else 16, "ram_mb": 3072 if preflight else 12288,
            "cpu_training_justification": "One exact controller reconstruction followed by independent single-thread future-replay workers."}
