"""Fixed path-held-out residual response qualification on fresh paths."""

from pathlib import Path

from freq_hrl.experiments.pointmaze_crossfit_response import PROTOCOL_VERSION
from scripts.pointmaze_forecast_response_spec import POLICY, roots, input_results as response_inputs

EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_forecast_response_stage31_v1_development"
RUNNER_SCRIPT = "scripts/run_pointmaze_crossfit_response.py"
CONTINUATION = "stage31_fit_designs_only; path_crossfit; centered_rbf_width_squared_columns; ridge1; hold100_equal_call150"
EVIDENCE_ROLE = "fresh_path_crossfit_plan_response_development_only"


def input_results(root, *, preflight):
    phase = "preflight" if preflight else "development"
    base = Path(__file__).resolve().parents[1]/"results"
    cell = Path("cells")/POLICY/f"replicate_{root}"/"result.json"
    return {**response_inputs(root, preflight=preflight),
            "response_result":base/f"pointmaze_forecast_response_stage31_v1_{phase}_20260928_r1"/cell}


def sampling_options(*, preflight):
    return {"pairs_per_path":1 if preflight else 15, "workers":1 if preflight else 16}


def task_resources(*, preflight):
    return {"cpu":2 if preflight else 17, "ram_mb":3072 if preflight else 24576}
