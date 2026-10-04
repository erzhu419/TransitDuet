#!/usr/bin/env python3
"""Dynamically schedule native local-credit pairs on node001-006."""
from scripts import pointmaze_option_residual_stage111_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task, qualification_task as previous_qualification


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight, protocol_spec=spec)
    task.update(cpu=3 if preflight else 5, ram_mb=3072 if preflight else 4096,
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/local_credit/{'preflight' if preflight else 'full'}",
        cpu_training_justification="Paired native suffix Q diagnostics; frozen feedback base; no local native training or checkpoint pull.")
    return task


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)
