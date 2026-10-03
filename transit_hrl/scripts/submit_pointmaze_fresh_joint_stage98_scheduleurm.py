#!/usr/bin/env python3
"""Submit new-cohort native mean training dynamically on node001-006."""

from scripts import pointmaze_fresh_joint_stage98_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task, qualification_task as previous_qualification


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight, protocol_spec=spec)
    task.update(resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{'preflight' if preflight else 'full'}",
        cpu_training_justification="Same Stage88 native MC mean learner and sample budget, new teacher sources; final weights stay server-only.")
    return task


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)
