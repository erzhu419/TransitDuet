"""Attribute Reacher1's cached A-prefix gap, with no new fit or simulation."""

import argparse
import json
from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.compute_budget import ComputeLedger, measured_phase, single_thread_runtime
from freq_hrl.experiments.plan_validity_boundary_wide_influence import attribute
from freq_hrl.experiments.pointmaze_goal_validation import _json_ready
from scripts.run_plan_validity_boundary_wide_transfer import load_wide_inputs


def run_attribution(manifest):
    source = json.loads((ROOT / manifest["source_manifest"]).read_text())
    with measured_phase("load_only_target_A_and_saved_folds"):
        design, prepared = load_wide_inputs(source, cell_indices=(manifest["cell_index"],))
        item, = prepared
        cell = item["cell"]
        directory = ROOT / source["output_directory"]
        previous = json.loads((directory / "public/result.json").read_text())
        if (previous["run_name"], previous["status"], previous["stage"], previous["new_ridge_fits"],
            previous["new_environment_steps"], previous["evaluation_prefixes_accessed"], previous["frozen_models_modified"]) != (
            source["run_name"], "complete", "boundary_Wide_A_transfer_complete", 256, 0, False, False):
            raise ValueError("completed matching A-only diagnostic is required")
        prior, = [c["diagnostic"] for c in previous["cells"] if (c["task"], c["optimizer_seed"]) == (cell["task"], cell["optimizer_seed"])]
        private = json.loads((directory / "private" / f"{cell['task']}_{cell['optimizer_seed']}.json").read_text())
    with measured_phase("fixed_model_coordinates_label_response_and_support"):
        public, detail = attribute(item["data"], item["frozen_model"], private, prior)
    return {"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
        "source_run_name": source["run_name"], "attribution": public}, detail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    directory = ROOT / manifest["output_directory"]
    if directory.exists():
        raise FileExistsError("preserve completed fixed-fold attribution attempts")
    for part in ("public", "private"):
        (directory / part).mkdir(parents=True)
    ledger = ComputeLedger()
    result = {"run_name": manifest["run_name"], "status": "running", "candidate_recipe_decision": "PILOT_RECIPE_HOLD",
        "confirmation_released": False, "performance_claim_supported": False, "lifecycle_costs_complete": False}
    try:
        design = json.loads((ROOT / manifest["design_manifest"]).read_text())
        with ledger.recording("boundary_Wide_fixed_influence"), single_thread_runtime() as profile:
            if profile != design["source_hardware_profile"]:
                raise ValueError("runtime differs from the frozen source")
            result["hardware_profile"] = profile
            public, private = run_attribution(manifest)
            (directory / "private/attribution.json").write_text(json.dumps(_json_ready(private)) + "\n")
        if ledger.summary()["counts"] != manifest["expected_counts"]:
            raise RuntimeError("attribution exceeded its fixed scoring/geometry and zero-fit budget")
        result.update(public, status="complete", stage="boundary_Wide_fixed_influence_complete",
            new_fits=0, new_environment_steps=0, evaluation_prefixes_accessed=False, frozen_models_modified=False)
    except Exception as exc:
        result.update(status="failed", stage="failed", error_type=type(exc).__name__, error=str(exc))
    result.update(measured_cost=ledger.summary(), peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (directory / "public/result.json").write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    print(result.get("stage", result["status"]), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
