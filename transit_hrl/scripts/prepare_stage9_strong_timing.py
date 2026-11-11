"""Reserve a bounded PointMaze-only strong-timing study and its worker jobs."""

import json
from copy import deepcopy
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_development import load_plan
from freq_hrl.experiments.pointmaze_goal_validation import _training_seed
from scripts import pointmaze_budgeted_trigger_stage9_spec as stage9
from scripts.prepare_plan_validity_boundary_conditioning_confirmation import historical_seeds
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON

RUN = "pointmaze_stage9_strong_timing_20261111"
MANIFEST = Path("protocols/plan_validity_stage9_strong_timing_2026-11-11.json")
SOURCE = Path("protocols/plan_validity_local_episode_diagnostic_2026-11-10.json")


def training_seeds(root, seeds, iterations):
    return [_training_seed(optimizer_seed=root, rollout_root=s, iteration=i)
            for s in seeds for i in range(iterations)]


def prepare():
    source = json.loads((ROOT / SOURCE).read_text())
    template = json.loads((ROOT / "protocols/plan_validity_boundary_conditioning_freeze_2026-11-11.json").read_text())["jobs"][0]
    protocols = ROOT / "protocols"
    forbidden = historical_seeds(protocols, load_plan(protocols / "plan_validity_preregistration_2026-10-22.json"))
    for root in (*stage9.OPTIMIZER_SEEDS, *stage9.CONFIRMATION_OPTIMIZER_SEEDS, *stage9.PREFLIGHT_OPTIMIZER_SEEDS):
        roles = stage9.seed_roles(root)
        forbidden.add(root)
        forbidden.update(s for seeds in roles.values() for s in seeds)
        forbidden.update(training_seeds(root, roles["train"], stage9.ITERATIONS))
    forbidden_count = len(forbidden)

    def reserve(seeds):
        if len(set(seeds)) != len(seeds) or forbidden.intersection(seeds):
            raise ValueError("new seed role overlaps an existing role or effective training seed")
        forbidden.update(seeds)

    cells, jobs = [], {"preflight": [], "audit": []}
    for i, cell in enumerate(c for c in source["cells"] if c["task"] == "pointmaze"):
        base = 1_981_010_000 + 1000 * i
        preflight, audit = base + 1, list(range(base + 100, base + 132))
        reserve([preflight, *audit])
        directory = f"results/{RUN}/development/{cell['optimizer_seed']}"
        cells.append({**cell, "preflight_seed": preflight, "audit_seeds": audit, "directory": directory})
        for phase in jobs:
            job = deepcopy(template)
            output = str(ROOT / directory / phase / "public")
            command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_stage9_strong_timing.py",
                       "--manifest", str(MANIFEST), "--phase", phase, "--cell", str(i)]
            job.update(project="Freq-HRL-Stage9-StrongTiming",
                description=f"PointMaze Stage9 {phase} {cell['optimizer_seed']}",
                signature=f"Freq-HRL/stage9-strong-timing/{RUN}/{phase}/{cell['optimizer_seed']}",
                cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command),
                result_dir=output, local_result_dir=output, wait_for_files=[str(ROOT / MANIFEST)])
            jobs[phase].append(job)

    confirmation = []
    for i in range(8):
        root = 1_981_400_000 + i * 10000
        roles = {name: list(range(root + start, root + start + count)) for name, start, count in (
            ("train_seeds", 1000, 8), ("selection_seeds", 2000, 8), ("branch_fit_seeds", 3000, 8),
            ("timing_selection_seeds", 4000, 8), ("trigger_eval_seeds", 5000, 128))}
        reserve([root, *(s for seeds in roles.values() for s in seeds),
                 *training_seeds(root, roles["train_seeds"], stage9.ITERATIONS)])
        confirmation.append({"optimizer_seed": root, **roles})
    return {"run_name": RUN, "source_manifest": str(SOURCE), "hardware_profile": source["source_hardware_profile"],
        "original_stage9_revision": stage9.ALGORITHM_REVISION, "scope": "pointmaze_ID_shared_controller_timing_only",
        "development_cells": cells, "confirmation_reservation": confirmation, "jobs": jobs,
        "historical_seed_exclusion_count": forbidden_count, "minimum_ise_gain": 0.02,
        "budget": {"cpu_seconds": 10000, "offline_steps": 6500000, "p99_seconds": 0.01, "deployments": 100},
        "development_gate": "all_budgets_and_termination_competence_selection_and_audit_every_root;"
            "at_least_2pct_pooled_episode_ISE_gain_against_threshold_termination_best_clock;"
            "positive_each_root_and_both_pooled_path_halves_for_each_comparison",
        "preflight_steps_per_root": 88800, "audit_steps_per_root": 153600,
        "confirmation_gate": "after_development_gate_only; freeze_all_8_eligible_budget_admitted_builds_before_test;"
            "each_paired_root_t95_interval_for_ISE_advantage_strictly_positive_vs_threshold_and_termination;"
            "at_least_2pct_pooled_gain_and_both_path_halves_favor_candidate_vs_both;"
            "best_clock_reported_and_at_least_2pct_pooled_gain_against_it; risk_and_tail_descriptive_only",
        "confirmation_jobs_released": False, "original_two_task_confirmation_released": False}


if __name__ == "__main__":
    if (ROOT / MANIFEST).exists():
        raise FileExistsError(MANIFEST)
    plan = prepare()
    with (ROOT / MANIFEST).open("x") as out:
        json.dump(plan, out, indent=2)
        out.write("\n")
