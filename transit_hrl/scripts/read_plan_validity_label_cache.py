"""Read frozen scores and common-target errors without fitting or native replay."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.plan_validity_label_continuation import CONTINUATIONS, HEADS, validate_banks
from freq_hrl.experiments.plan_validity_phase_support import predict
from freq_hrl.experiments.plan_validity_window_coverage import error
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.analyze_plan_validity_label_continuation import summarize


def margin_readout(rows, rules):
    scores = {name: predict(rules[name], rows) for name in HEADS}
    left, right = (scores[name] for name in HEADS)
    delta = right - left
    cuts = [rules[name].threshold for name in HEADS]
    margins = [score - cut for score, cut in zip((left, right), cuts)]
    changed = (margins[0] >= 0) != (margins[1] >= 0)
    return {"states": len(rows), "cut_reference": cuts[0], "cut_candidate": cuts[1],
        "cut_difference": cuts[1] - cuts[0], "mean_score_difference": float(delta.mean()),
        "state_dependent_score_difference_SD": float(delta.std()),
        "margin_difference_RMS": float(np.sqrt(np.mean((margins[1] - margins[0]) ** 2))),
        "chosen_now": {name: int(np.count_nonzero(margin >= 0)) for name, margin in zip(HEADS, margins)},
        "action_disagreements": int(changed.sum()), "changed_states": [
            {"seed": row["seed"], "step": row["opportunity_step"]} for row, change in zip(rows, changed) if change]}


def read_cache(manifest, *, root=ROOT):
    public = summarize(manifest, root=root)
    if public["status"] != "complete":
        raise ValueError("cache readout requires all frozen experiment phases complete")
    roots = []
    for cell, saved in zip(manifest["cells"], public["root_readouts"]):
        directory = root / cell["output_directory"]
        rules = {k: TimingRule(**v) for k, v in json.loads((directory / "fit/private/frozen_rules.json").read_text()).items()}
        panels = {}
        for phase in ("sample", "probe"):
            banks = {c: [] for c in CONTINUATIONS}
            for shard in (0, 1):
                rows = json.loads((directory / f"{phase}/shard{shard}/private/label_banks.json").read_text())
                for c in CONTINUATIONS:
                    banks[c].extend(rows[c])
            seeds = cell["fit_prefix_seeds"] if phase == "sample" else cell["evaluation_seeds"]
            a, b = (manifest["fitting_futures"], manifest["independent_B_futures"]) if phase == "sample" else (
                manifest["evaluation_A_futures"], manifest["evaluation_B_futures"])
            validate_banks(banks, seeds=seeds, probes=manifest["probes_per_path"], a=a, b=b)
            target = banks["candidate"]
            errors = {name: {block: error(predict(rule, target), target, target="W", block=block, independent=phase == "probe" or block == "B")
                for block in ("A", "B")} for name, rule in rules.items()}
            for name in HEADS:
                for block in ("A", "B"):
                    expected = (saved["training_readout"][name]["cross_target_B"]["candidate"]["full_fit"]["mse"] if phase == "sample" and block == "B"
                        else saved["blocks"][block]["candidate"]["heads"][name]["W_mse"]["mean"] if phase == "probe" else None)
                    if expected is not None and not np.isclose(errors[name][block]["mse"], expected, rtol=1e-10, atol=1e-10):
                        raise ValueError("cached frozen predictions do not reproduce public errors")
            panels[phase] = {"common_C_target_errors": errors, "score_and_action_changes": margin_readout(target, rules)}
        roots.append({"task": cell["task"], "optimizer_seed": cell["optimizer_seed"], **panels})
    return {"run_name": manifest["run_name"], "status": "complete", "role": "posthoc_frozen_cache_readout",
        "root_readouts": roots, "new_fits": 0, "new_environment_steps": 0,
        "candidate_recipe_decision": "PILOT_RECIPE_HOLD", "confirmation_released": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    start_cpu = time.process_time()
    result = read_cache(json.loads(args.manifest.read_text()))
    result["analysis_cpu_seconds"] = time.process_time() - start_cpu
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "new_fits", "new_environment_steps", "analysis_cpu_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
