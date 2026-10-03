#!/usr/bin/env python3
"""Schedule plan-baseline comparisons dynamically across node001-006."""
from scripts import pointmaze_plan_baselines_stage106_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task, qualification_task as previous_qualification


def task_specification(run_name, root, *, preflight):
    return previous_task(run_name, root, preflight=preflight, protocol_spec=spec)


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)
