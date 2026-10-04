#!/usr/bin/env python3
"""Dynamically schedule small frozen-response probes on node001-006."""
from scripts import pointmaze_control_response_stage109_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task, qualification_task as previous_qualification


def task_specification(run_name, root, *, preflight):
    task = previous_task(run_name, root, preflight=preflight, protocol_spec=spec)
    task.update(cpu=3 if preflight else 5, ram_mb=3072 if preflight else 4096,
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/frozen_response/{'preflight' if preflight else 'full'}",
        cpu_training_justification="Native forecast-only trajectories and batched fixed-policy response; no training or checkpoints.")
    return task


def qualification_task(run_name, *, preflight):
    return previous_qualification(run_name, preflight=preflight, protocol_spec=spec)
