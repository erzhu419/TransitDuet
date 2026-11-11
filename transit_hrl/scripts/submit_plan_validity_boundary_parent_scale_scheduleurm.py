"""Freeze a single geometry change; no baseline refits or new data."""

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


def prepare(previous, run_name, manifest_path):
    manifest = deepcopy(previous)
    manifest.update(protocol_version="cache_only_parent_scaled_boundary_tail_v1", run_name=run_name,
        prior_probe_run_name=previous["run_name"],
        boundary_model="parent_standardization_then_products_then_difference_direct_zero_intercept_SVD_ridge",
        parent_scaling="training_now_and_wait_arm_states_only_39_parents_center_std_existing_1e-8_constant_rule",
        contrast_scaling="none", baseline_fits="none_reuse_prior_predictions_and_reproduce_saved_losses",
        ridge_fits_per_root=9, constant_fits_per_root=0, total_ridge_fits=36, total_constant_fits=0,
        followup="none_no_additional_geometry_search_distillation_simulation_or_policy_release")
    for i, (cell, job) in enumerate(zip(manifest["cells"], manifest["jobs"])):
        cell["prior_probe_directory"] = cell["output_directory"]
        cell["output_directory"] = f"results/{run_name}/{cell['task']}/{cell['optimizer_seed']}"
        directory = ROOT / cell["output_directory"]
        cmd = [DEFAULT_LINUX_PYTHON, "-u", "scripts/run_plan_validity_boundary_parent_scale.py", "--manifest", str(manifest_path), "--cell-index", str(i)]
        job.update(project="Freq-HRL-BoundaryParentScale", description=f"Parent-scaled tail {cell['task']} {cell['optimizer_seed']}",
            signature=f"Freq-HRL/boundary-parent-scale/{run_name}/{cell['task']}/{cell['optimizer_seed']}",
            cmd=job["cmd"].split(DEFAULT_LINUX_PYTHON)[0] + shlex.join(cmd),
            result_dir=str(directory / "public"), local_result_dir=str(directory / "public"),
            wait_for_files=[str(ROOT / manifest_path)])
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest.exists():
        raise FileExistsError("preserve the frozen geometry-repair manifest")
    result = prepare(json.loads(args.prior_manifest.read_text()), args.run_name, args.manifest.resolve().relative_to(ROOT))
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    if args.submit:
        execute_bulk(result["jobs"], dry_run=False, intent_label=args.run_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
