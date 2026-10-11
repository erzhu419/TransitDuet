#!/usr/bin/env python3
"""Complete opposite-direction native labels on the registered training scenes."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_data_stage174 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
shared, records, validate_continuations = source.shared, source.records, source.validate_continuations
source_record = source.source_record
EXPERIMENT_PROTOCOL = "native_transit_value_direction_stage175_v1"
SOURCE_RUN = "native_transit_value_data_stage174_frozen_20261011_r1"
PROBES = tuple(i+1 for i in source.PROBES)


def contract():
    return {**source.contract(), "paired_source_run": SOURCE_RUN, "paired_source_protocol": source.EXPERIMENT_PROTOCOL,
        "probes": list(PROBES), "paired_probes": list(source.PROBES),
        "coverage": "adjacent_opposite_direction_decisions_same_ten_training_scenes_no_outcome_selection",
        "boundary": "60_correlated_bidirectional_training_states_360_labels_after_merge_not_policy_performance_validation"}


def paired_reference(root, scenario, index, summary):
    if not (summary["protocol"] == source.EXPERIMENT_PROTOCOL and summary["contract"] == source.contract()
            and summary["software_qualified"] and summary["training_updates"] == 0
            and summary["native_steps"] == 210*61380 and summary["source_control_reproductions"] == 60
            and summary["window_ledger_qualified_episodes"] == 210 and summary["training_states"] == 30):
        raise ValueError("Opposite-direction labels require qualified Stage174 training data")
    matches = [r for r in summary["training_data"]
        if (r["root"], r["scenario"], r["probe_index"]) == (root, scenario, index-1)]
    if index not in PROBES or len(matches) != 1:
        raise ValueError("Missing unique registered adjacent training probe")
    return matches[0]


def validate_pair(row, paired):
    probe = row["observer"]["probe"]
    if not (row["seed"] == paired["root"] and row["scenario"] == paired["scenario"]
            and row["probe_index"] == paired["probe_index"]+1 and row["probe_index"] in PROBES
            and row["observer"]["scene_seed"] == paired["scene_seed"]
            and probe["index"] == row["probe_index"] and probe["direction"] != paired["direction"]
            and probe["time_s"] >= paired["time_s"]):
        raise ValueError("Native training probes are not adjacent opposite-direction pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--probe", type=int, choices=PROBES, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--paired-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paired = paired_reference(args.seed, args.scenario, args.probe, json.loads(args.paired_data.read_text()))
    registered = source_record(args.seed, args.scenario)
    validate_pair({"seed": args.seed, "scenario": args.scenario, "probe_index": args.probe,
        "observer": {"scene_seed": registered["scene_seed"],
            "probe": {**registered["supply"]["decisions"][args.probe], "index": args.probe}}}, paired)
    result = source.collect(args.seed, args.scenario, args.probe,
        json.loads(args.models.read_text())[str(args.seed)], args.output)
    validate_pair(result, paired)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), **result})
    print("NATIVE_BIDIRECTIONAL_VALUE_DATA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
