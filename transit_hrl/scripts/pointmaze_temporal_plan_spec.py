"""New path-disjoint temporal supervision with remote-only raw caches."""

from freq_hrl.experiments.pointmaze_temporal_plan import PROTOCOL_VERSION
from scripts.pointmaze_deployed_pair_diagnostic_spec import POLICY, roots, source_result


EXPERIMENT_PROTOCOL = PROTOCOL_VERSION
SOURCE_PROTOCOL = "pointmaze_timing_pair_stage12_v1_development"
RUNNER_SCRIPT = "scripts/run_pointmaze_temporal_plan.py"
CONTINUATION = "balanced_jitter; matched_one_check_pair; 10_25_50_step_cost_curve"
EVIDENCE_ROLE = "fresh_path_temporal_plan_development_only"


def sampling_options(*, preflight):
    return {"pairs_per_path": 2 if preflight else 20, "curve_epochs": 4 if preflight else 128,
            "workers": 1 if preflight else 16}


def task_resources(*, preflight):
    return {"cpu": 2 if preflight else 17, "ram_mb": 3072 if preflight else 24576}
