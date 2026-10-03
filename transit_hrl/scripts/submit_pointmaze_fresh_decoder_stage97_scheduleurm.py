#!/usr/bin/env python3
"""Schedule new-cohort decoder calibration dynamically on node001-006."""

from scripts import pointmaze_fresh_decoder_stage97_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task, qualification_task as previous_qualification


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight, protocol_spec=spec)
    task.update(resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'preflight' if preflight else 'full'}",
        cpu_training_justification="Offline new-BC-label command calibration and small native probes; no optimizer or trace writes.")
    return task


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)
