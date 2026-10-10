#!/usr/bin/env python3
"""Separate physical service-allocation authority from learned upper claims."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_plan_stage156 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells):
    contract = spec.contract()
    if set(cells) != set(spec.ROOTS):
        raise ValueError("Incomplete service-plan roots")
    metrics = (*spec.source_spec.authority.routing.METRICS, "lower_action_mean")
    means, panels = {}, []
    for root, cell in cells.items():
        required = {(condition, scenario, scene) for condition in spec.CONDITIONS
            for scenario in contract["scenarios"]
            for scene in spec.source_spec.scene_seeds(root, scenario, preflight=False)}
        if not (cell["software_qualified"] and cell["baseline_reproduced"] and cell["neutral_reproduced"]
                and cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == contract
                and cell["method"] == spec.METHOD and cell["seed"] == root and cell["training_updates"] == 0
                and cell["native_steps"] == len(required) * contract["training_clock_s"]):
            raise ValueError("Unqualified frozen service-plan cell")
        rows = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in cell["evaluation"]}
        if set(rows) != required or len(rows) != len(cell["evaluation"]):
            raise ValueError("Missing or duplicate service-plan scenes")
        for (condition, scenario, scene), row in rows.items():
            reference = rows["source_baseline", scenario, scene]
            if (any(row[k] != reference[k] for k in ("N_fleet", "passengers_generated", "simulation_end_time_s", "scheduled_trips"))
                    or row["N_fleet"] != 12 or row["simulation_end_time_s"] != contract["training_clock_s"]
                    or not all(np.isfinite(row[k]) for k in metrics)):
                raise ValueError("Unpaired/nonfinite service-plan outcomes")
            if condition == "source_baseline":
                continue
            ledger = row["service_plan"]
            ids = []
            for block in row["blocks"]:
                nominal, planned = np.asarray(block["nominal_s"]), np.asarray(block["planned_s"])
                targets = np.asarray(block["target_headways_s"])
                ids.extend(block["trip_ids"])
                if (len(nominal) != len(planned) or len(targets) != len(planned)
                        or len(planned) != len(block["trip_ids"]) or not 1 <= len(planned) <= 6
                        or planned[0] != nominal[0] or planned[-1] != nominal[-1]
                        or block["decision_time_s"] > planned[0]
                        or np.any(np.diff(planned) < 239) or np.any(np.diff(planned) > 481)
                        or not np.array_equal(targets[1:], np.diff(planned))):
                    raise ValueError("Plan changed the time budget or lost executable lower goals")
            if (not row["blocks"] or len(ids) != len(set(ids)) or len(ids) != ledger["queries"]
                    or ledger["queries"] != row["scheduled_trips"]
                    or ledger["blocks"] != len(row["blocks"]) or ledger["endpoint_error_max_s"] != 0
                    or not all(np.isfinite(v) for v in ledger.values())):
                raise ValueError("Incomplete service-allocation execution ledger")
            if condition == "nominal_plan" and ledger["planned_shift_max_abs_s"] != 0:
                raise ValueError("Nominal allocation moved a departure")
            if condition in {"frontload", "backload"} and ledger["changed_headway_count"] == 0:
                raise ValueError("Fixed allocation failed to change any service intervals")
        for condition in spec.CONDITIONS:
            for scenario in contract["scenarios"]:
                values = {k: float(np.mean([r[k] for (c, s, _), r in rows.items()
                    if c == condition and s == scenario])) for k in metrics}
                panels.append({"root": root, "condition": condition, "scenario": scenario, **values})
            means[root, condition] = {k: float(np.mean([r[k] for r in panels
                if r["root"] == root and r["condition"] == condition])) for k in metrics}
    contrast = lambda a, b: [{"root": root, **{k: means[root, a][k] - means[root, b][k]
        for k in metrics}} for root in spec.ROOTS]
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": contract, "software_qualified": True,
        "stage": "frozen_service_allocation_authority_not_learned_upper_evidence", "training_updates": 0,
        "native_steps": sum(c["native_steps"] for c in cells.values()),
        "plan_minus_nominal": {c: contrast(c, "nominal_plan") for c in spec.PLAN_CONDITIONS if c != "nominal_plan"},
        "forecast_minus_reversed": contrast("causal_forecast", "forecast_reversed"),
        "regime_root_means": panels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT / "results" / args.run_name
    cells = {root: json.loads((directory / "cells" / spec.METHOD / f"seed_{root}/result.json").read_text())
             for root in spec.ROOTS}
    result = summarize(cells)
    write_json(directory / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_means"}}, indent=2))


if __name__ == "__main__":
    main()
