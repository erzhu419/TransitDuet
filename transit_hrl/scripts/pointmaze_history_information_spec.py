"""Remote-only Stage-26 cache reuse for a bounded information diagnostic."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_history_information import PROTOCOL_VERSION
from scripts.pointmaze_temporal_plan_spec import POLICY, roots
from scripts.pointmaze_deployed_pair_diagnostic_spec import source_result as controller_result


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_temporal_plan_stage26_v1_development"
RUNNER_SCRIPT = "scripts/run_pointmaze_history_information.py"
CONTINUATION = "stage26_remote_cache; lags1_10_25_50; horizons10_25_50; ridge_sum_loss_alpha1"
EVIDENCE_ROLE = "reused_temporal_cache_information_diagnostic_only"


def input_results(root, *, preflight):
    if root not in roots(preflight=preflight):
        raise ValueError("history-information root is not registered")
    phase = "preflight" if preflight else "development"
    source = (Path(__file__).resolve().parents[1] / "results"
              / f"pointmaze_temporal_plan_stage26_v1_{phase}_20260928_r1"
              / "cells" / POLICY / f"replicate_{root}" / "result.json")
    return {"source_result": source, "controller_result": controller_result(root, preflight=preflight)}


def sampling_options(*, preflight):
    return {}


def task_resources(*, preflight):
    return {"cpu": 1, "ram_mb": 1536}
