#!/usr/bin/env python3
"""Schedule read-only crossed execution dynamically on node001-006."""
from scripts import pointmaze_crossed_advice_stage108_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task, qualification_task as previous_qualification


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight, protocol_spec=spec)
    task.update(cpu=3 if preflight else 5, ram_mb=3072 if preflight else 4096,
        cpu_training_justification="Native evaluation workers only; all policy weights fixed; no checkpoint writes.")
    return task


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)
