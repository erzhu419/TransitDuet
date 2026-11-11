"""Release only the 32 training-acquisition shards from the frozen design."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.submit_hyperparameter_pilot_scheduleurm import DEFAULT_LINUX_PYTHON, execute_bulk


def prepare(design, design_path, manifest_path):
    shards, jobs = [], []
    for i, cell in enumerate(design["cells"]):
        for method in ("dense", "wide"):
            seeds = cell["development_prefix_seeds"] if method == "dense" else cell["wide_training_prefix_seeds"]
            for j, steps in enumerate(design[method]["planned_shard_steps"]):
                directory = str(Path(cell["output_directory"]) / "sample" / method / f"shard{j}")
                prefixes = seeds[4 * j:4 * (j + 1)]
                pairs = steps // (2 * design["horizon"]) if method == "dense" else 4 * 3 * 4 * 2
                entry = {"cell_index": i, "method": method, "shard_index": j, "prefix_seeds": prefixes,
                    "environment_steps": steps, "new_pairs": pairs, "output_directory": directory}
                cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_boundary_information_sample.py",
                    "--manifest", str(manifest_path), "--shard-index", str(len(shards))]
                job = deepcopy(design["jobs"][i])
                job.update(project="Freq-HRL-BoundaryInformation-Sample", description=f"Training A {method} {cell['task']} {cell['optimizer_seed']} shard{j}",
                    allow_cpu_training=True,
                    signature=f"Freq-HRL/boundary-information/{design['run_name']}/sample/{method}/{cell['task']}/{cell['optimizer_seed']}/shard{j}",
                    cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
                    result_dir=str(ROOT / directory / "public"), local_result_dir=str(ROOT / directory / "public"),
                    wait_for_files=[str(ROOT / design_path), str(ROOT / manifest_path)])
                shards.append(entry)
                jobs.append(job)
    if len(shards) != 32 or sum(s["environment_steps"] for s in shards) != design["formal_training_total_steps"]:
        raise ValueError("release does not match the frozen 32-shard training budget")
    return {"protocol_version": "boundary_information_training_acquisition_release_v1", "run_name": design["run_name"],
        "design_manifest": str(design_path), "released_environment_steps": design["formal_training_total_steps"],
        "shards": shards, "jobs": jobs, "new_fits": 0, "readout_released": False,
        "current_release": "training_A_acquisition_only", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen sampling release")
    design = json.loads(args.design.read_text())
    receipt = json.loads((ROOT / "results" / design["run_name"] / "precheck_summary.json").read_text())
    if (receipt["run_name"], receipt["stage"], receipt["environment_steps"], receipt["fits"]) != (
        design["run_name"], "four_mechanical_prechecks_complete", design["precheck_total_steps"], 0):
        raise ValueError("completed matching precheck receipt is required")
    result = prepare(design, args.design.resolve().relative_to(ROOT), args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit:
        execute_bulk(result["jobs"], dry_run=False, intent_label=design["run_name"] + "_sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
