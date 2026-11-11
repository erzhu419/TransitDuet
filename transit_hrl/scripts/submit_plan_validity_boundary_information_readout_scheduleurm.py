"""Release the reserved eight-prefix E readout, without fitting or cut search."""

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


def prepare(design, design_path, fit_path, manifest_path):
    shards, jobs = [], []
    for i, cell in enumerate(design["cells"]):
        seeds = cell["readout_prefix_seeds"]
        training = {*cell["development_prefix_seeds"], *cell["wide_training_prefix_seeds"], cell["precheck_seed"]}
        if len(seeds) != 8 or len(set(seeds)) != 8 or set(seeds) & training or design["readout"]["suffix_indices"] != list(range(20, 28)):
            raise ValueError("reserved readout roles differ or overlap training")
        for j in range(2):
            prefixes = seeds[j * 4:(j + 1) * 4]
            pairs = len(prefixes) * design["checks_per_prefix"] * design["windows_per_state"] * len(design["readout"]["suffix_indices"])
            steps = design["horizon"] * (len(prefixes) + 2 * pairs)
            directory = str(Path(cell["output_directory"]) / "readout" / f"shard{j}")
            shard = {"cell_index": i, "shard_index": j, "prefix_seeds": prefixes,
                "environment_steps": steps, "new_pairs": pairs, "output_directory": directory}
            cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_boundary_information_readout.py",
                "--manifest", str(manifest_path), "--shard-index", str(len(shards))]
            job = deepcopy(design["jobs"][i])
            job.update(project="Freq-HRL-BoundaryInformation-Readout", description=f"Frozen E readout {cell['task']} {cell['optimizer_seed']} shard{j}",
                allow_cpu_training=True, protect_from_auto_eviction=True,
                signature=f"Freq-HRL/boundary-information/{design['run_name']}/readout/{cell['task']}/{cell['optimizer_seed']}/shard{j}",
                cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
                result_dir=str(ROOT / directory / "public"), local_result_dir=str(ROOT / directory / "public"),
                wait_for_files=[str(ROOT / design_path), str(ROOT / fit_path), str(ROOT / manifest_path)])
            shards.append(shard)
            jobs.append(job)
    if len(shards) != 8 or sum(s["environment_steps"] for s in shards) != design["formal_readout_total_steps"]:
        raise ValueError("E release differs from the frozen budget")
    return {"protocol_version": "boundary_information_frozen_E_release_v1", "run_name": design["run_name"],
        "design_manifest": str(design_path), "fit_manifest": str(fit_path), "shards": shards, "jobs": jobs,
        "released_environment_steps": design["formal_readout_total_steps"], "new_fits": 0,
        "current_release": "reserved_new_prefix_E_prediction_only_all_frozen_models",
        "statistics": design["readout_statistics"], "halves": design["readout"]["halves"],
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen E release")
    design = json.loads(args.design.read_text())
    receipt = json.loads((ROOT / "results" / design["run_name"] / "fit_summary.json").read_text())
    if (receipt["run_name"], receipt["status"], receipt["stage"], receipt["all_models_frozen"],
        receipt["new_ridge_fits"], receipt["new_environment_steps"], receipt["evaluation_prefixes_accessed"]) != (
        design["run_name"], "complete", "boundary_information_fit_complete", True, 8, 0, False):
        raise ValueError("matching completed global fit receipt is required")
    result = prepare(design, args.design.resolve().relative_to(ROOT), args.fit_manifest.resolve().relative_to(ROOT),
        args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit:
        execute_bulk(result["jobs"], dry_run=False, intent_label=design["run_name"] + "_readout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
