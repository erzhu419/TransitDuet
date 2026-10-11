#!/usr/bin/env python3
"""Qualify correction controls and compare actual native service outcomes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_remaining_supply_stage172 as spec
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from scripts.analyze_native_transit_supply_allocation_stage163 import contrast
from freq_hrl.experiments.pointmaze_root_response import write_json


def ranking_diagnostic(original, corrected):
    pairs = list(zip(original["supply"]["decisions"], corrected["supply"]["decisions"]))
    changed = [i for i, (a, b) in enumerate(pairs) if a["selected"] != b["selected"]]
    first = None
    if changed:
        i = changed[0]
        a, b = pairs[i]
        old, new = a["selected"], b["selected"]
        ac, bc = a["candidates"], b["candidates"]
        samples = corrected["remaining_corrections"][i]["active"]
        first = {"index": i, "time_s": a["time_s"], "direction": a["direction"],
            "original_choice": old, "corrected_choice": new,
            "same_candidate_prefixes": [c["commitment"]["prefix_planned_s"] for c in ac]
                == [c["commitment"]["prefix_planned_s"] for c in bc],
            "same_candidate_wait_costs": [c["commitment"]["prefix_wait_cost"] for c in ac]
                == [c["commitment"]["prefix_wait_cost"] for c in bc],
            "original_choice_score_advantage": ac[new]["score"]-ac[old]["score"],
            "corrected_choice_score_advantage": bc[old]["score"]-bc[new]["score"],
            "relative_fleet_shift_against_original_choice":
                (bc[old]["fleet_cost"]-ac[old]["fleet_cost"])-(bc[new]["fleet_cost"]-ac[new]["fleet_cost"]),
            "active_samples": len(samples),
            **{kind+"_active_sample_mean_mae_s": float(np.mean([
                abs(s[kind+"_sample_mean_s"]-s["actual_remaining_s"]) for s in samples]))
                if samples else None for kind in ("original", "corrected")}}
    return {"scene_seed": corrected["scene_seed"], "choice_changes": len(changed),
        "actual_peak_delta": corrected["peak_fleet"]-original["peak_fleet"],
        "actual_cost_delta": corrected["service_cost_restricted"]-original["service_cost_restricted"],
        "actual_wait_delta_min": corrected["restricted_wait_horizon_min"]-original["restricted_wait_horizon_min"],
        "first_choice_divergence": first}


def summarize(cells, sources, pack, source_summary, source_duration_models):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required:
        raise ValueError("Incomplete remaining-supply matrix")
    if pack != {"duration_models": source_duration_models, "remaining_models": source_summary["models"]}:
        raise ValueError("Changed registered fit-only duration or correction models")
    metrics = spec.native.source.source_spec.authority.routing.METRICS
    all_rows, panels, diagnostics, rankings = {r: [] for r in spec.ROOTS}, [], [], []
    for (root, scenario), cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["seed"] == root and cell["scenario"] == scenario and cell["software_qualified"]
                and cell["training_updates"] == 0 and cell["native_steps"] == 4*61380
                and [r["condition"] for r in cell["evaluation"]] == list(spec.CONDITIONS)):
            raise ValueError("Changed remaining-supply contract or native budget")
        duration, remaining = pack["duration_models"][str(root)], pack["remaining_models"][str(root)]
        rows = cell["evaluation"]
        window = next(r for r in sources[root, scenario]["evaluation"] if r["condition"] == "check_control")
        for row in rows:
            ref = rows[1] if row["condition"] == "zero_remaining_correction" else window
            reproduced = row["condition"] in spec.CONTROLS
            if not (row["scenario"] == scenario and row["scene_seed"] == spec.source.check_scene(root, scenario)
                    and row["source_control_reproduced"] == reproduced and row["N_fleet"] == 12
                    and row["simulation_end_time_s"] == 61380 and np.all(np.isfinite([row[k] for k in metrics]))):
                raise ValueError("Changed remaining-supply scenes or physical controls")
            spec.incumbent.source.distributional.source.check_episode(row, ref, baseline=reproduced)
            validate_schedule(row["schedule"], ref["schedule"])
            if reproduced and row["schedule"] != ref["schedule"]:
                raise ValueError("Changed exact remaining-supply control schedule")
            if row["condition"] == "window_forecast":
                if row["supply"] is not None or row["remaining_corrections"]:
                    raise ValueError("Window control executed a supply correction")
            else:
                spec.validate_selector(row, duration, remaining, window["completed_trips"])
                if row["condition"] == "zero_remaining_correction" and (
                        row["supply"] != ref["supply"] or row["window_panels"] != ref["window_panels"]):
                    raise ValueError("Zero correction changed native scores or window outcomes")
        all_rows[root].extend(rows)
        panels.extend({"root": root, "scenario": scenario, **contrast(rows, "corrected_remaining", ref, metrics)}
            for ref in spec.CONDITIONS[:2])
        rankings.append({"root": root, "scenario": scenario, **ranking_diagnostic(rows[1], rows[3])})
    for root, rows in all_rows.items():
        changed = [r for r in rows if r["condition"] == "corrected_remaining"]
        samples = [s for r in changed for p in r["remaining_corrections"] for s in p["active"]]
        diagnostics.append({"root": root, "active_samples": len(samples),
            "choice_counts": np.sum([r["supply"]["choice_counts"] for r in changed], axis=0).tolist(),
            "mean_correction_s": float(np.mean([s["correction_s"] for s in samples])),
            **{kind+"_sample_mean_mae_s": float(np.mean([abs(s[kind+"_sample_mean_s"]-s["actual_remaining_s"])
                for s in samples])) for kind in ("original", "corrected")}})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 40*61380, "source_control_reproductions": 20,
        "window_ledger_qualified_episodes": 30,
        "paired_contrasts": [{"root": r, **contrast(rows, "corrected_remaining", ref, metrics)}
            for r, rows in all_rows.items() for ref in spec.CONDITIONS[:2]],
        "regime_root_contrasts": panels, "active_diagnostics": diagnostics,
        "ranking_diagnostics": rankings, "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory, source = ROOT/"results"/args.run_name, ROOT/"results"/spec.SOURCE_RUN
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells},
        json.loads((directory/"inputs/models.json").read_text()), json.loads((source/"summary.json").read_text()),
        json.loads((source/"inputs/duration_models.json").read_text()))
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_contrasts", "ranking_diagnostics"}}, indent=2))


if __name__ == "__main__":
    main()
