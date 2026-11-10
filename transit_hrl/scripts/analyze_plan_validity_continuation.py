"""Merge continuation shards without treating shards or futures as trained roots."""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_continuation import CONTEXTS, CONTRASTS
from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval


def aggregate_paths(paths):
    output = {}
    for block in ("A", "B", "all"):
        contexts = {c: {k: float(np.mean([p["blocks"][block]["contexts"][c][k] for p in paths]))
            for k in paths[0]["blocks"][block]["contexts"][c]} for c in CONTEXTS}
        contrasts = {}
        for name in CONTRASTS:
            contrasts[name] = {}
            for key in ("W", "tail", "G"):
                values = [p["blocks"][block]["contrasts"][name][key] for p in paths]
                contrasts[name][key] = {"path_interval": paired_mean_interval([v["mean"] for v in values]),
                    "conditional_MC_standard_error": float(np.sqrt(sum(v["conditional_MC_standard_error"] ** 2 for v in values)) / len(values))}
        output[block] = {"contexts": contexts, "contrasts": contrasts}
    return output


def summarize(manifest, *, root=ROOT):
    roots, missing, failed, costs = [], [], [], []
    for cell in manifest["cells"]:
        shards = []
        for index in range(manifest["shards_per_root"]):
            path = root / cell["output_directory"] / f"run/shard{index}/public/result.json"
            identity = [cell["task"], cell["optimizer_seed"], index]
            if not path.exists():
                missing.append(identity)
                continue
            row = json.loads(path.read_text())
            if (row["run_name"] != manifest["run_name"] or row["task"] != cell["task"]
                or row["optimizer_seed"] != cell["optimizer_seed"] or row["phase"] != "run" or row["shard_index"] != index):
                raise ValueError("result differs from the frozen continuation shard")
            costs.append(row["measured_cost"])
            if row["status"] != "complete":
                failed.append(identity)
                continue
            expected = cell["evaluation_seeds"][index * 4:(index + 1) * 4]
            if row["evaluation_seeds"] != expected or [p["seed"] for p in row["paths"]] != expected:
                raise ValueError("continuation shard changes or duplicates its path roster")
            shards.append(row)
        if len(shards) != manifest["shards_per_root"]:
            roots.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "status": "incomplete"})
            continue
        paths = [p for s in shards for p in s["paths"]]
        roots.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], "status": "complete",
            "path_count": len(paths), "paired_clock_probes": sum(p["probes"] for p in paths),
            "blocks": aggregate_paths(paths),
            "episodes": {panel: paired_mean_interval([p["episodes"]["candidate"] - p["episodes"]["reference"] for p in subset])
                for panel, subset in (("all", paths), ("first_half", paths[:4]), ("second_half", paths[4:]))},
            "support_counts": {k: sum(p["support_counts"][k] for p in paths) for k in paths[0]["support_counts"]},
            "occupancy_shards": [s["occupancy"] for s in shards],
            "termination_development_eligible": all(s["termination_development_eligible"] for s in shards)})
    return {"run_name": manifest["run_name"], "status": "complete" if not missing and not failed else "incomplete_or_failed",
        "missing": missing, "failed": failed, "root_readouts": roots,
        "trained_roots_per_task": {task: sum(c["task"] == task for c in manifest["cells"]) for task in sorted(set(c["task"] for c in manifest["cells"]))},
        "known_main_cpu_seconds": sum(c["cpu_seconds"] for c in costs),
        "statistical_scope": "path_CIs_conditional_on_frozen_root_MC_SE_not_additional_root_replication",
        "interpretation": "common_clock_diagnostic_not_additive_policy_improvement_or_new_learned_policy",
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
