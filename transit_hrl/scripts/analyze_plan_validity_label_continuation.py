"""Read paired-label phases; preserve root counts and fixed-continuation scope."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_label_continuation import CONTINUATIONS
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval


def paired_effect(rows):
    return {"path_interval": paired_mean_interval([r["mean"] for r in rows]),
        "conditional_MC_standard_error": float(np.sqrt(sum(r["conditional_MC_standard_error"] ** 2 for r in rows)) / len(rows))}


def aggregate_paths(paths):
    output = {}
    for block in ("A", "B", "all"):
        output[block] = {}
        for continuation in CONTINUATIONS:
            rows = [p["blocks"][block][continuation] for p in paths]
            heads = {name: {"W_mse": paired_mean_interval([r["heads"][name]["W_mse"] for r in rows]),
                "noise_corrected_W_mse": paired_mean_interval([r["heads"][name]["noise_corrected_W_mse"] for r in rows]),
                "gains_against_source_C": {k: paired_effect([r["heads"][name]["gains_against_source_C"][k] for r in rows])
                    for k in ("W", "tail", "G")}} for name in rows[0]["heads"]}
            contrast = [r["candidate_minus_reference_label"] for r in rows]
            output[block][continuation] = {"heads": heads, "candidate_minus_reference_label": {
                "W_mse_difference": paired_mean_interval([r["W_mse_difference"] for r in contrast]),
                "gains": {k: paired_effect([r["gains"][k] for r in contrast]) for k in ("W", "tail", "G")}}}
    return output


def summarize(manifest, *, root=ROOT):
    roots, missing, failed, costs = [], [], [], []
    for cell in manifest["cells"]:
        records = {}
        for phase, shards in (("precheck", [-1]), ("sample", [0, 1]), ("fit", [-1]), ("probe", [0, 1])):
            for shard in shards:
                directory = root / cell["output_directory"] / phase
                if shard >= 0:
                    directory /= f"shard{shard}"
                path = directory / "public/result.json"
                key = [cell["task"], cell["optimizer_seed"], phase, shard]
                if not path.exists():
                    missing.append(key)
                    continue
                row = json.loads(path.read_text())
                if any(row[k] != value for k, value in (("run_name", manifest["run_name"]), ("task", cell["task"]),
                    ("optimizer_seed", cell["optimizer_seed"]), ("phase", phase), ("shard_index", shard))):
                    raise ValueError("result differs from its frozen paired-label phase")
                costs.append(row["measured_cost"])
                if row["status"] != "complete":
                    failed.append(key)
                    continue
                if phase in ("sample", "probe"):
                    roster, count = (cell["fit_prefix_seeds"], 16) if phase == "sample" else (cell["evaluation_seeds"], 4)
                    if row["sampled_seeds"] != roster[count * shard:count * (shard + 1)]:
                        raise ValueError("paired-label shard changes its path roster")
                    if phase == "probe" and [p["seed"] for p in row["paths"]] != row["sampled_seeds"]:
                        raise ValueError("paired-label probe paths are missing or duplicated")
                records[(phase, shard)] = row
        result = {"task": cell["task"], "optimizer_seed": cell["optimizer_seed"],
            "sampling_complete": all(("sample", i) in records for i in (0, 1)),
            "fit_complete": ("fit", -1) in records, "probe_complete": all(("probe", i) in records for i in (0, 1))}
        if result["fit_complete"]:
            result["training_readout"] = records[("fit", -1)]["training_readout"]
        if result["probe_complete"]:
            paths = [p for i in (0, 1) for p in records[("probe", i)]["paths"]]
            result.update(path_count=len(paths), blocks=aggregate_paths(paths))
        roots.append(result)
    return {"run_name": manifest["run_name"], "status": "complete" if not missing and not failed else "incomplete_or_failed",
        "missing": missing, "failed": failed, "root_readouts": roots,
        "trained_roots_per_task": {task: sum(c["task"] == task for c in manifest["cells"]) for task in sorted(set(c["task"] for c in manifest["cells"]))},
        "known_cpu_seconds": sum(c["cpu_seconds"] for c in costs),
        "statistical_scope": "eight_path_CIs_conditional_on_frozen_root_not_new_trained_roots",
        "control_scope": "one_forced_timing_action_with_fixed_prefix_and_continuation_not_head_closed_loop",
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False,
        "lifecycle_costs_complete": False, "performance_claim_supported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "missing", "failed", "candidate_recipe_decision")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
