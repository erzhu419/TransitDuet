#!/usr/bin/env python3
"""Qualify incumbent reproduction and augmented-pool physical performance."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_incumbent_window_stage169 as spec
from scripts.analyze_native_transit_supply_allocation_stage163 import contrast
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from freq_hrl.experiments.pointmaze_root_response import write_json


def summarize(cells, sources, models):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required or set(models) != set(spec.ROOTS):
        raise ValueError("Incomplete incumbent-window matrix")
    for root, model in models.items():
        spec.source.distributional.validate_model(model)
        if model != spec.source.distributional.fit_model(root):
            raise ValueError("Incumbent experiment changed the fit-only model")
    metrics = spec.native.source.source_spec.authority.routing.METRICS
    all_rows, panels = {r: [] for r in spec.ROOTS}, []
    for (root, scenario), cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["scenario"] == scenario and cell["software_qualified"]
                and cell["training_updates"] == 0 and cell["native_steps"] == 16*61380):
            raise ValueError("Changed incumbent-window contract or budget")
        scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
        rows = {(r["condition"], r["scene_seed"]): r for r in cell["evaluation"]}
        if len(rows) != len(cell["evaluation"]) or set(rows) != {(c, s) for c in spec.CONDITIONS for s in scenes}:
            raise ValueError("Incomplete or duplicate incumbent-window episodes")
        for (condition, scene), row in rows.items():
            ref = next(r for r in sources[root, scenario]["evaluation"] if r["scene_seed"] == scene and r["condition"] == (
                "window_forecast" if condition in {"window_forecast", "incumbent_only"} else "prefix_wait_full_fleet"))
            reproduced, incumbent = condition in spec.CONTROLS, condition == "incumbent_only"
            if not (row["scenario"] == scenario and row["source_control_reproduced"] == reproduced
                    and row["incumbent_control_reproduced"] == incumbent and row["N_fleet"] == 12
                    and row["simulation_end_time_s"] == 61380 and np.all(np.isfinite([row[k] for k in metrics]))):
                raise ValueError("Changed incumbent-window physical controls")
            spec.source.distributional.source.check_episode(row, ref, baseline=reproduced or incumbent)
            validate_schedule(row["schedule"], ref["schedule"])
            if (reproduced or incumbent) and row["schedule"] != ref["schedule"]:
                raise ValueError("Changed exact incumbent/source schedule")
            if reproduced and row["supply"] != ref["supply"]:
                raise ValueError("Changed source candidate scores or choices")
            if condition == "window_forecast":
                if row["supply"] is not None:
                    raise ValueError("Window control must not select supply candidates")
            else:
                spec.validate_candidates(row, models[root])
                if reproduced and row["window_panels"] != ref["window_panels"]:
                    raise ValueError("Changed source window accounting")
            all_rows[root].append(row)
        panels.extend({"root": root, "scenario": scenario, **contrast(list(rows.values()), "rolling_plus_incumbent", ref, metrics)}
            for ref in spec.CONTROLS)
    diagnostics = []
    for root, rows in all_rows.items():
        decisions = [d for r in rows if r["condition"] == "rolling_plus_incumbent" for d in r["supply"]["decisions"]]
        diagnostics.append({"root": root, "decisions": len(decisions),
            "choice_counts": [sum(d["selected"] == i for d in decisions) for i in range(6)],
            "incumbent_choices": sum(d["selected"] == 5 for d in decisions),
            "mean_selected_proxy_gain_over_incumbent": float(np.mean([
                d["candidates"][5]["score"]-d["candidates"][d["selected"]]["score"] for d in decisions]))})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 160*61380, "source_control_reproductions": 80,
        "incumbent_control_reproductions": 40, "window_ledger_qualified_episodes": 120,
        "paired_contrasts": [{"root": r, **contrast(rows, "rolling_plus_incumbent", ref, metrics)}
            for r, rows in all_rows.items() for ref in spec.CONTROLS],
        "regime_root_contrasts": panels, "candidate_diagnostics": diagnostics, "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    models = {int(r): m for r, m in json.loads((directory/"inputs/duration_models.json").read_text()).items()}
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells}, models)
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_contrasts"}}, indent=2))


if __name__ == "__main__":
    main()
