#!/usr/bin/env python3
"""Full-cohort sensitivity of the matched-root optimal-action diagnosis."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_native_transit_best_action_stage218 as source
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_best_action_full_stage219_v1"
SOURCE_RUN = "native_transit_best_action_stage218_diagnostic_20261014_r1"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "data": "all_original240_mean8_states_40_complete_prefixes_both_roots_no_new_data",
        "role": "sample_growth_reference_not_equal_size_single_vs_mixed_comparison",
        "constraints_and_features": source.contract(),
        "budget": {"views": 1, "max_linear_programs": 8, "optimizer_updates": 0,
            "actor_updates": 0, "new_native_episodes": 0},
        "boundary": "finite_full_training_class_diagnosis_not_generalization_or_population_CI"}


def analyze(bank, panels, registered, save_references):
    views = source.matched_views(bank["training_data"], registered)
    rows = [*views["root_397"], *views["root_401"]]
    if set(panels) != {source.state_key(r) for r in rows}:
        raise ValueError("Full-cohort panels changed the original240 states")
    result = source.diagnose_view(rows, panels, save_references)
    bounds = [b for p in result["partitions"].values() for b in p["bounds"].values()]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "full_cohort": result, "linear_programs": sum(b["linear_programs"] for b in bounds),
        "unresolved_bounds": sum(b["feasible"] is None for b in bounds),
        "linear_bounds_qualified": all(b["feasible"] is not None for b in bounds),
        "optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0,
        "third_scene_screen_eligible": False, "native_deployment_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != source.EXPERIMENT_PROTOCOL or prior["contract"] != source.contract() or not prior["linear_bounds_qualified"]:
        raise ValueError("Complete qualified Stage218 before full-cohort sensitivity")
    bank, _, registered, _ = source.source.load_data()
    result = analyze(bank, source.source.wait_source.source.diagnostic.load_panels(), registered,
        lambda p, r: write_json(raw_directory(args.output)/f"{p}.json", r))
    write_json(args.output, result)
    print(json.dumps({p: {k: b["feasible"] for k, b in v["bounds"].items()}
        for p, v in result["full_cohort"]["partitions"].items()}), flush=True)
    print("NATIVE_BEST_ACTION_FULL_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
