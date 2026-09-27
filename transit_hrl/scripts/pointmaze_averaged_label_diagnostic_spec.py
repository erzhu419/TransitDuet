"""Frozen cached-label comparison on the existing Stage-18 opportunities."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_averaged_label_diagnostic import PROTOCOL_VERSION, SOURCE_PROTOCOL
from scripts.pointmaze_state_noise_diagnostic_spec import POLICY, roots, endpoint_result, sampling_options


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
RUNNER_SCRIPT = "scripts/run_pointmaze_averaged_label_diagnostic.py"
CONTINUATION = "cached_stage18_futures; matched_compact_critic; single_draw_vs_mean; no_environment_sampling"
EVIDENCE_ROLE = "cached_label_averaging_diagnostic_only"


def source_result(root, *, preflight):
    if root not in roots(preflight=preflight):
        raise ValueError("Stage-19 root is not registered")
    phase = "preflight" if preflight else "development"
    return (Path(__file__).resolve().parents[1] / "results"
            / f"pointmaze_state_noise_stage18_v1_{phase}_20260927_r1"
            / "cells" / POLICY / f"replicate_{root}" / "result.json")


def input_results(root, *, preflight):
    return {"source_result": source_result(root, preflight=preflight),
            "endpoint_result": endpoint_result(root, preflight=preflight)}
