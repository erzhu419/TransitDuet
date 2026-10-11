#!/usr/bin/env python3
"""Qualify frozen second-scene value transfer without fitting or deployment."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_transfer_stage179 as spec
from scripts.analyze_native_transit_value_data_stage174 import summarize as qualify_labels
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, sources, models, source_models, value_input, primary_cell):
    if value_input != spec.model_input(primary_cell):
        raise ValueError("Transfer changed frozen Stage178 weights or normalization")
    result = qualify_labels(cells, sources, models, source_models, protocol_spec=spec)
    rows = result.pop("training_data")
    for row in rows:
        key = row["root"], row["scenario"], row["probe_index"]
        cell = cells[key]
        if [row["root"], row["scene_seed"]] in value_input["model"]["training_groups"]:
            raise ValueError("Transfer uses a critic training scene")
        spec.context.qualify_contexts([cell["context"]], [row], sources[key[:2]])
        row["occupancy_context"] = {k: cell["context"][k] for k in spec.OCCUPANCY_KEYS}
        expected = spec.frozen_prediction(row, value_input["model"])
        if not (all(cell["prediction"][k] == expected[k] for k in ("time_s", "selected"))
                and np.allclose(cell["prediction"]["predicted_heads"], expected["predicted_heads"], rtol=1e-10, atol=1e-12)):
            raise ValueError("Sealed preaction prediction differs from frozen inference")
    evaluated = spec.paired.evaluate_action_value(rows, value_input["model"], predictor=spec.value.predict_value)
    groups = lambda key: {str(g): spec.paired.summarize_choices([r for r in evaluated if r[key] == g])
        for g in sorted({r[key] for r in evaluated})}
    by_root = groups("root")
    states = result.pop("training_states")
    return {**result, "evaluation_states": states, "critic_updates": 0, "actor_updates": 0,
        "evaluation_groups": [{"root": r, "scenario": s, "scene_seed": sources[r, s]["scene_seed"],
            "probes": list(spec.PROBES)} for r in spec.ROOTS for s in spec.SCENARIOS],
        "evaluation_data": rows, "predictions": evaluated, "overall": spec.paired.summarize_choices(evaluated),
        "by_root": by_root, "by_regime": groups("scenario"), "by_direction": groups("direction"),
        "frozen_transfer_improves_both_roots": all(v["mean_actual_cost_delta"] < 0 for v in by_root.values()),
        "native_screen_eligible": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s, i): json.loads((directory/"cells"/s/f"seed_{r}/probe_{i}/result.json").read_text())
        for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES}
    for key, cell in cells.items():
        prediction = json.loads((directory/"cells"/key[1]/f"seed_{key[0]}/probe_{key[2]}/prediction.json").read_text())
        if prediction != cell["prediction"]:
            raise ValueError("Prediction saved before branches differs from the completed result")
    result = summarize(cells, {(r, s): spec.source_record(r, s) for r in spec.ROOTS for s in spec.SCENARIOS},
        json.loads((directory/"inputs/models.json").read_text()),
        json.loads((ROOT/"results"/spec.labels.SOURCE_RUN/"inputs/duration_models.json").read_text()),
        json.loads((directory/"inputs/value_model.json").read_text()),
        json.loads((ROOT/"results"/spec.SOURCE_RUN/"cells"/spec.fitted.PRIMARY/"result.json").read_text()))
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "evaluation_data", "predictions"}}, indent=2))


if __name__ == "__main__":
    main()
