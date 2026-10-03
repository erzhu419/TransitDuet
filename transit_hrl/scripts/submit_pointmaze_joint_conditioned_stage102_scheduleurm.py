#!/usr/bin/env python3
"""Dispatch separate-credit joint MC learning dynamically on node001-006."""

from scripts import pointmaze_joint_conditioned_stage102_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task, qualification_task as previous_qualification


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight, protocol_spec=spec)
    task.update(resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'preflight' if preflight else 'full'}",
        cpu_training_justification="Native joint mean updates with separate upper/lower credit rollouts; only final weights remain server-side.")
    return task


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)
