"""Fresh-path hold/renew supervision using remote cached controllers."""

from freq_hrl.experiments.pointmaze_plan_hold import PROTOCOL_VERSION
from scripts.pointmaze_history_information_spec import POLICY, roots, input_results


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_temporal_plan_stage26_v1_development"
RUNNER_SCRIPT = "scripts/run_pointmaze_plan_hold.py"
CONTINUATION = "cached_controller; old_plan_hold100; executed_equal_call_settlement150; ridge_alpha1"
EVIDENCE_ROLE = "fresh_path_true_plan_hold_development_only"


def sampling_options(*, preflight):
    return {"pairs_per_path": 2 if preflight else 20, "workers": 1 if preflight else 16}


def task_resources(*, preflight):
    return {"cpu": 2 if preflight else 17, "ram_mb": 3072 if preflight else 24576}
