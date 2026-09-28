"""Independent motion inference with fixed causal lags and fresh query paths."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_separate_motion import PROTOCOL_VERSION
from scripts.pointmaze_history_information_spec import POLICY, roots
from scripts.pointmaze_deployed_pair_diagnostic_spec import source_result as controller_result

EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_state_response_stage29_v1_development"
RUNNER_SCRIPT = "scripts/run_pointmaze_separate_motion.py"
CONTINUATION = "stage29_cached_fit; external_only_unit_ridge; lags1_10_25_50; fresh_motion_paths"
EVIDENCE_ROLE = "fresh_exogenous_motion_development_only"


def input_results(root, *, preflight):
    if root not in roots(preflight=preflight):
        raise ValueError("separate-motion root is not registered")
    phase = "preflight" if preflight else "development"
    source = (Path(__file__).resolve().parents[1]/"results"
              /f"pointmaze_state_response_stage29_v1_{phase}_20260928_r1"
              /"cells"/POLICY/f"replicate_{root}"/"result.json")
    return {"source_result":source, "controller_result":controller_result(root, preflight=preflight)}


def sampling_options(*, preflight):
    return {}


def task_resources(*, preflight):
    return {"cpu":1, "ram_mb":1536}
