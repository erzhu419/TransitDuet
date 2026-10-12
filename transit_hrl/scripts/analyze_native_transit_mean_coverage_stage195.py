#!/usr/bin/env python3
"""Report matched single-target versus complete-group mean-target values."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_mean_coverage_stage195 as spec
from scripts.analyze_native_transit_expected_value_stage192 import qualify
from scripts.analyze_native_transit_coverage_value_stage188 import validation_summary, FIELDS
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json


def summarize(cells, registered, prepared):
    if set(cells) != {(m, f["split"], f["index"]) for m in spec.MODES for f in registered["folds"]}:
        raise ValueError("Missing complete-group mean-target fits")
    if not (prepared["protocol"] == spec.EXPERIMENT_PROTOCOL and prepared["contract"] == spec.contract()
            and prepared["software_qualified"] and prepared["input_states"] == 600
            and prepared["mean_states"] == 240 and prepared["prefix_groups"] == 40
            and prepared["new_draws"] == 1680 and prepared["reused_draws"] == 240
            and prepared["new_native_episodes"] == 13440):
        raise ValueError("Report changed target coverage or native collection budget")
    for mode in spec.MODES:
        for fold in registered["folds"]:
            qualify(cells[mode, fold["split"], fold["index"]], mode, fold, registered, protocol_spec=spec)
    variants = {}
    for variant in spec.VARIANTS:
        for policy in spec.source.value.POLICIES:
            evaluations = {}
            for label in ("original", "mean8"):
                evaluations[label] = {}
                for split in spec.SPLITS:
                    rows = [r for fold in registered["folds"] if fold["split"] == split
                        for r in cells["tail_summary", split, fold["index"]]["predictions"][variant][label][policy]]
                    if len(rows) != 240 or len({spec.expected.state_key(r) for r in rows}) != 240:
                        raise ValueError("Report lost a complete-group test state")
                    summary = validation_summary(rows)
                    summary["by_regime"] = {s: validation_summary([r for r in rows if r["scenario"] == s])
                        for s in spec.SCENARIOS}
                    evaluations[label][split] = summary
            gates = {"cost": all(v["mean_actual_cost_delta"] < 0 for s in evaluations["mean8"].values() for v in s["by_root"].values()),
                "wait": all(v["mean_wait_delta_min"] <= 0 for s in evaluations["mean8"].values() for v in s["by_root"].values())}
            variants[f"tail_summary:{variant}:{policy}"] = {"evaluation": evaluations, "expected_label_gates": gates}
    effects = {}
    for policy in spec.source.value.POLICIES:
        previous = variants[f"tail_summary:cached_original:{policy}"]["evaluation"]["mean8"]
        current = variants[f"tail_summary:mean8_overlay:{policy}"]["evaluation"]["mean8"]
        effects[policy] = {split: {"overall": {f: current[split]["overall"][f]-previous[split]["overall"][f] for f in FIELDS},
            "false_feasible_choices_delta": current[split]["false_feasible_choices"]-previous[split]["false_feasible_choices"]}
            for split in spec.SPLITS}
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "new_fits": len(cells), "reused_fits": len(cells), "critic_updates": len(cells)*spec.STEPS,
        "actor_updates": 0, "evaluation_states": 240, "evaluation_prefix_scene_groups": 40,
        "new_native_episodes": 13440, "reused_native_episodes": 1920, "mean_noise": prepared["mean_noise"],
        "variants": variants, "mean_target_effects": effects,
        "fit_wall_s_sum": sum(c["fit_wall_s"] for c in cells.values()),
        "expected_target_development_gate": all(variants[spec.contract()["primary"]]["expected_label_gates"].values()),
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    registered = json.loads((directory/"inputs/manifest.json").read_text())
    prepared = json.loads((directory/"prepare/result.json").read_text())
    cells = {(m, f["split"], f["index"]): json.loads((raw_directory(spec.result_path(args.run_name, m, f))/"result.json").read_text())
        for m in spec.MODES for f in registered["folds"]}
    result = summarize(cells, registered, prepared)
    write_json(args.output, result)
    print(json.dumps({"fits": result["new_fits"], "gates": {k: v["expected_label_gates"] for k, v in result["variants"].items()}}, indent=2))
    print("NATIVE_MEAN_COVERAGE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
