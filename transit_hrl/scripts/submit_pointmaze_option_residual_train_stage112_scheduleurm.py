#!/usr/bin/env python3
"""Schedule branch-only native learning dynamically on node001-006."""
from scripts import pointmaze_option_residual_train_stage112_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task, qualification_task as previous_qualification


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight, protocol_spec=spec)
    task.update(cpu=5 if not preflight else 3, ram_mb=4096 if not preflight else 3072,
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/branch_only/{'preflight' if preflight else 'full'}",
        cpu_training_justification="Native branch-only MC mean updates with frozen base, critic and upper; final branch weights server-only.")
    return task


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)
