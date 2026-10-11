#!/usr/bin/env python3
"""Join qualified full timetable inputs without changing old cost/wait labels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_timetable_context_stage185 as spec
from freq_hrl.domains.transit.native_timetable_context import event_features
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, rows, sources):
    expected = {(r, s, i) for r in spec.ROOTS for s in spec.SCENARIOS for i in (0, 1)}
    if set(cells) != expected or set(sources) != expected or len(rows) != 120:
        raise ValueError("Missing full timetable development scenes or original label states")
    data = []
    for (root, scenario, index), cell in sorted(cells.items()):
        registered = sources[root, scenario, index]
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and (cell["seed"], cell["scenario"], cell["scene_index"], cell["scene_seed"])
                == (root, scenario, index, registered["scene_seed"])
                and cell["training_updates"] == 0 and cell["native_steps"] == 61380
                and cell["observer"]["source_control_reproduced"] and cell["observer"]["window_ledger_qualified"]):
            raise ValueError("Changed passive full timetable scene, policy or episode budget")
        for k in spec.OBSERVER_KEYS:
            if k not in ("source_control_reproduced", "window_ledger_qualified") and cell["observer"][k] != registered[k]:
                raise ValueError("Full timetable replay changed a source physical outcome")
        if cell["policy_trace"] != spec.shared.policy_trace(registered["supply"]):
            raise ValueError("Full timetable replay changed a source selector trace")
        group = [r for r in rows if (r["root"], r["scenario"], r["scene_seed"]) == (root, scenario, registered["scene_seed"])]
        spec.qualify_contexts(cell["contexts"], group, registered)
        by_index = {c["index"]: c for c in cell["contexts"]}
        for row in group:
            data.append({**row, "timetable_context": by_index[row["probe_index"]]["timetable_context"]})
    if len(data) != 120 or len({(r["root"], r["scene_seed"], r["probe_index"]) for r in data}) != 120:
        raise ValueError("Full timetable join lost or duplicated an original label state")
    lengths = [event_features(r).shape[1] for r in data]
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "new_native_steps": 20*61380, "new_native_episodes": 20,
        "source_control_reproductions": 20, "training_states": 120, "paired_labels_reused": 720,
        "event_count_range": [min(lengths), max(lengths)], "event_feature_count": len(spec.EVENT_NAMES),
        "training_data": sorted(data, key=lambda r: (r["root"], r["scene_seed"], r["probe_index"])),
        "next_step": "same_label_grouped_full_timetable_encoder_test_before_expanding_development_scenes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    rows = json.loads((directory/"inputs/training.json").read_text())
    prior = ROOT/"results"/spec.SOURCE_RUN/"inputs"
    original = spec.prior.source.source.source.training_rows(*(json.loads((prior/f"{n}.json").read_text())
        for n in ("first", "labels", "second", "baselines")))
    models = json.loads((directory/"inputs/models.json").read_text())
    source_models = json.loads((ROOT/"results"/spec.MODEL_SOURCE_RUN/"inputs/models.json").read_text())
    if rows != original or models != source_models:
        raise ValueError("Full timetable inputs changed source labels or duration models")
    keys = [(r, s, i) for r in spec.ROOTS for s in spec.SCENARIOS for i in (0, 1)]
    result = summarize({(r, s, i): json.loads((directory/"cells"/s/f"seed_{r}"/f"scene_{i}/result.json").read_text())
        for r, s, i in keys}, rows, {(r, s, i): spec.labels.source_record(r, s, scene_index=i) for r, s, i in keys})
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("contract", "training_data")}, indent=2))


if __name__ == "__main__":
    main()
