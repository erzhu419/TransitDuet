"""Schedule the compact Stage112 branch diagnostic on node001-006."""
import shlex

from scripts import pointmaze_option_residual_diagnostic_spec as spec
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification as previous_task
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk


def task_specification(run_name, root):
    task = previous_task(run_name, root, preflight=True)
    output = spec.ROOT / "results" / run_name / "cells" / f"replicate_{root}" / "result.json"
    command = [DEFAULT_LINUX_PYTHON, "-u", spec.RUNNER_SCRIPT, "--optimizer-seed", str(root),
        "--output", str(output)]
    task.update(project=spec.EXPERIMENT_PROTOCOL,
        description=f"Freq-HRL {spec.EXPERIMENT_PROTOCOL} root{root}",
        signature=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/{run_name}/{root}",
        resource_family=f"Freq-HRL/{spec.EXPERIMENT_PROTOCOL}/diagnostic",
        cpu=3, ram_mb=3072,
        cpu_training_justification="Frozen Stage112 branch output diagnostic; no training and no checkpoint write.",
        cmd="PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= " + shlex.join(command))
    return task


def main():
    run_name = "pointmaze_option_residual_diagnostic_stage112_full_20261004_r1"
    tasks = [task_specification(run_name, root) for root in spec.roots()]
    execute_bulk(tasks, dry_run=False, intent_label=spec.EXPERIMENT_PROTOCOL + ":" + run_name)


if __name__ == "__main__":
    main()
