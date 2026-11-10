"""Freeze prefix/continuation diagnostics; four prechecks precede eight shards."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_development import load_plan
from freq_hrl.experiments.plan_validity_preregistration import seed_cells
from freq_hrl.experiments.pointmaze_goal_validation import _training_seed
from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk


def prepare(source, plan, run_name, manifest_path):
    cells = deepcopy(source["cells"])
    registered = list(seed_cells(plan))
    forbidden = {s for c in registered for seeds in c["seed_roles"].values() for s in seeds}
    forbidden.update(c["optimizer_seed"] for c in registered)
    for c in cells:
        forbidden.update(c["fit_prefix_seeds"] + c["held_prefix_seeds"] + c["exposed_held_prefix_seeds"])
    forbidden.update(_training_seed(optimizer_seed=c["optimizer_seed"], rollout_root=s, iteration=i)
        for c in registered for s in c["seed_roles"]["train_seeds"]
        for i in range(2 if c["phase"] == "preflight" else plan["contract"]["primary_iterations"]))
    jobs = {"precheck": [], "run": []}
    for index, cell in enumerate(cells):
        root = cell["optimizer_seed"]
        cell["phase_cache_directory"] = cell["output_directory"] + "/run/private"
        cell["phase_public_result"] = cell["output_directory"] + "/run/public/result.json"
        cell["previous_evaluation_seeds"] = cell["held_prefix_seeds"]
        cell["evaluation_seeds"] = list(range(root + 20000, root + 20008))
        if set(cell["evaluation_seeds"]) & forbidden:
            raise ValueError("diagnostic paths overlap exposed/reserved/effective seeds")
        forbidden.update(cell["evaluation_seeds"])
        cell["output_directory"] = f"results/{run_name}/{cell['task']}/{root}"
        for phase, shards in (("precheck", [-1]), ("run", [0, 1])):
            for shard in shards:
                directory = ROOT / cell["output_directory"] / ("precheck" if shard < 0 else f"run/shard{shard}")
                command = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_continuation.py", "--manifest", str(manifest_path),
                    "--cell-index", str(index), "--phase", phase, "--shard-index", str(shard)]
                job = deepcopy(source["jobs"][phase][index])
                job.update(project="Freq-HRL-Continuation", description=f"Frozen prefix/continuation {phase} {cell['task']} {root} shard{shard}",
                    signature=f"Freq-HRL/continuation/{run_name}/{phase}/{cell['task']}/{root}/shard{shard}",
                    cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(command),
                    result_dir=str(directory / "public"), local_result_dir=str(directory / "public"),
                    wait_for_files=[str(ROOT / manifest_path)])
                jobs[phase].append(job)
    return {"protocol_version": "frozen_prefix_continuation_diagnostic_v1", "run_name": run_name,
        "source_hardware_profile": source["source_hardware_profile"], "cells": cells, "jobs": jobs,
        "policies": {"reference": "original_history", "candidate": "phase_checks_clock"},
        "controller_and_policy_fitting": "none", "probes_per_path": 3, "conditional_replicates": 8,
        "shards_per_root": 2, "paths_per_shard": 4,
        "sampling": "960_bins_961_common_reference_candidate_optional_check_uniform_per_bin",
        "future_seed_recipe": "SeedSequence([optimizer_seed,path_seed,step,replica_index,962])",
        "continuation_switch": "first_primitive_step_after_forced_action_not_before_it",
        "support_role": "paired_clock_intersection_only_exclusive_checks_counted_not_forced",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "protocols/plan_validity_preregistration_2026-10-22.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit-precheck", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen continuation manifest")
    result = prepare(json.loads(args.source_manifest.read_text()), load_plan(args.preregistration), args.run_name,
                     args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit_precheck:
        execute_bulk(result["jobs"]["precheck"], dry_run=False, intent_label=args.run_name + "_precheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
