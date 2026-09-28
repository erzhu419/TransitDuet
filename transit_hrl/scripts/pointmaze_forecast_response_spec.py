"""Frozen motion-to-plan-response bridge with fresh equal-call evaluation."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_forecast_response import PROTOCOL_VERSION
from scripts.pointmaze_history_information_spec import POLICY,roots,input_results as controller_inputs

EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_plan_hold_stage28_v1_development"
RUNNER_SCRIPT = "scripts/run_pointmaze_forecast_response.py"
CONTINUATION = "stage28_fit; stage30_frozen_motion; causal_candidate_plan; hold100_equal_call150; unit_ridge"
EVIDENCE_ROLE = "fresh_path_frozen_forecast_plan_response_development_only"


def input_results(root, *, preflight):
    phase = "preflight" if preflight else "development"
    base = Path(__file__).resolve().parents[1]/"results"
    cell = Path("cells")/POLICY/f"replicate_{root}"/"result.json"
    return {**controller_inputs(root,preflight=preflight),
            "hold_result":base/f"pointmaze_plan_hold_stage28_v1_{phase}_20260928_r1"/cell,
            "motion_result":base/f"pointmaze_separate_motion_stage30_v1_{phase}_20260928_r1"/cell}


def sampling_options(*, preflight):
    return {"pairs_per_path":1 if preflight else 15,"workers":1 if preflight else 16}


def task_resources(*, preflight):
    return {"cpu":2 if preflight else 17,"ram_mb":3072 if preflight else 24576}
