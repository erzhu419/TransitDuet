"""Release exactly the reserved 16-shard frozen conditioning readout."""

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


def prepare(plan, freeze, design, plan_path, manifest_path):
    shards, jobs = [], []
    for i, cell in enumerate(plan["cells"]):
        source = design["cells"][i]
        if (cell["task"], cell["optimizer_seed"]) != (source["task"], source["optimizer_seed"]):
            raise ValueError("plan and frozen C0 roots differ")
        seeds = cell["readout_prefix_seeds"]
        old = {*source["development_prefix_seeds"], *source["wide_training_prefix_seeds"], *source["readout_prefix_seeds"], source["precheck_seed"]}
        if len(seeds) != 16 or len(set(seeds)) != 16 or set(seeds) & old or plan["suffix_indices"] != list(range(40, 48)):
            raise ValueError("reserved fresh roles differ or overlap exposed prefixes")
        for j in range(4):
            prefixes = seeds[j * 4:(j + 1) * 4]
            pairs = len(prefixes) * plan["checks_per_prefix"] * plan["windows_per_state"] * len(plan["suffix_indices"])
            steps = plan["horizon"] * (len(prefixes) + 2 * pairs)
            directory = f"results/{plan['run_name']}/{cell['task']}/{cell['optimizer_seed']}/shard{j}"
            cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_boundary_conditioning_confirmation.py",
                "--manifest", str(manifest_path), "--shard-index", str(len(shards))]
            job = deepcopy(freeze["jobs"][0])
            job.update(project="Freq-HRL-Conditioning-FreshConfirmation",
                description=f"Frozen predictor confirmation {cell['task']} {cell['optimizer_seed']} shard{j}",
                signature=f"Freq-HRL/boundary-conditioning-confirmation/{plan['run_name']}/{cell['task']}/{cell['optimizer_seed']}/shard{j}",
                cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
                result_dir=str(ROOT / directory / "public"), local_result_dir=str(ROOT / directory / "public"),
                wait_for_files=[str(ROOT / p) for p in (plan_path, plan["freeze_manifest"], freeze["design_manifest"], manifest_path)],
                allow_cpu_training=True, protect_from_auto_eviction=True)
            shards.append({"cell_index": i, "shard_index": j, "prefix_seeds": prefixes,
                "environment_steps": steps, "new_pairs": pairs, "output_directory": directory})
            jobs.append(job)
    if len(shards) != plan["proposed_shards"] or sum(s["environment_steps"] for s in shards) != plan["proposed_environment_steps"]:
        raise ValueError("release differs from the frozen confirmation budget")
    return {"protocol_version": "conditioning_frozen_confirmation_release_v1", "run_name": plan["run_name"],
        "plan_manifest": str(plan_path), "design_manifest": freeze["design_manifest"], "shards": shards, "jobs": jobs,
        "released_environment_steps": plan["proposed_environment_steps"], "native_sampling_released": True, "new_fits": 0,
        "current_release": "fresh_prefix_frozen_boundary_prediction_only_no_policy_confirmation",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "performance_claim_supported": False, "lifecycle_costs_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the fixed confirmation release")
    plan = json.loads(args.plan.read_text())
    freeze = json.loads((ROOT / plan["freeze_manifest"]).read_text())
    design = json.loads((ROOT / freeze["design_manifest"]).read_text())
    receipt = json.loads((ROOT / freeze["output_directory"] / "summary.json").read_text())
    r, check = receipt["result"], receipt["bundle_check"]
    if (r["run_name"], r["status"], r["stage"], r["all_models_frozen"], r["new_ridge_fits"], r["new_environment_steps"],
        check["all_eight_models_present"], check["all_old_controls_preserved"]) != (
        freeze["run_name"], "complete", "boundary_conditioning_full_A_freeze_complete", True, 8, 0, True, True):
        raise ValueError("completed verified global model freeze is required")
    release = prepare(plan, freeze, design, args.plan.resolve().relative_to(ROOT), args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(release, indent=2) + "\n")
    if args.submit:
        execute_bulk(release["jobs"], dry_run=False, intent_label=plan["run_name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
