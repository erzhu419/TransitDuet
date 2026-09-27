"""Split-future decision diagnostic using three existing result caches."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_tail_decision_diagnostic import PROTOCOL_VERSION
from scripts.pointmaze_ridge_error_diagnostic_spec import POLICY, roots, input_results as value_sources


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_contextual_pair_stage23_v1_diagnostic"
RUNNER_SCRIPT = "scripts/run_pointmaze_tail_decision_diagnostic.py"
CONTINUATION = "frozen_reference; cached_short_window; disjoint_future_halves; frozen_critics"
EVIDENCE_ROLE = "retrospective_split_future_tail_decision_only"


def input_results(root, *, preflight):
    sources = value_sources(root, preflight=preflight)
    phase = "preflight" if preflight else "development"
    return {"endpoint_result": sources["endpoint_result"], "fresh_result": sources["fresh_result"],
            "prediction_result": Path(__file__).resolve().parents[1] / "results"
            / f"pointmaze_contextual_pair_stage23_v1_{phase}_20260927_r1"
            / "cells" / POLICY / f"replicate_{root}" / "result.json"}


def sampling_options(*, preflight):
    return {"opportunities_per_path": 1 if preflight else 2, "future_replicates": 4 if preflight else 64}
