"""Frozen predictive-state factorial and actuator intervention protocol."""

from freq_hrl.experiments.pointmaze_state_response import PROTOCOL_VERSION
from scripts.pointmaze_history_information_spec import POLICY, roots, input_results

EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_temporal_plan_stage26_v1_development"
RUNNER_SCRIPT = "scripts/run_pointmaze_state_response.py"
CONTINUATION = "cached_controller; action_excitation025; state_gaussian_gru64_latent16; factorial_history_action"
EVIDENCE_ROLE = "fresh_path_action_conditioned_predictive_state_development_only"


def sampling_options(*, preflight):
    return {"model_epochs":4 if preflight else 64, "workers":1 if preflight else 4}


def task_resources(*, preflight):
    return {"cpu":2 if preflight else 5, "ram_mb":3072 if preflight else 8192}
