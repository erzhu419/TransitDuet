#!/usr/bin/env python3
"""Analyze physical value of the cold-start prior, with exact source controls."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_calibrated_supply_stage165 as spec
from scripts.analyze_native_transit_supply_allocation_stage163 import validate_supply, contrast
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from freq_hrl.experiments.pointmaze_root_response import write_json

CONTRASTS = (("calibrated_supply", "legacy_supply"), ("calibrated_supply", "calibrated_demand"),
             ("calibrated_supply", "window_forecast"))


def validate_prior(supply, prior):
    if not (supply["initial_duration_prior_s"] == prior
            and supply["duration_model"] == "fit_only_prior_until_first_completion_then_original_EWMA"
            and supply["decisions"][0]["observed_completions"] == 0):
        raise ValueError("Calibrated selector did not use the registered training-only prior")
    previous, peak = {"False", "True"}, 0
    for d in supply["decisions"]:
        in_use, estimates = set(d["prior_directions_in_use"]), d["duration_estimates_s"]
        if not (in_use <= previous and set(estimates) == {"False", "True"}
                and all(np.isfinite(v) and v > 0 for v in estimates.values())
                and all(estimates[k] == prior[k] for k in in_use)
                and d["historical_peak"] >= peak
                and all(c["predicted_peak"] >= d["historical_peak"] for c in d["candidates"])):
            raise ValueError("Invalid causal prior-to-completion transition or historical peak")
        previous, peak = in_use, d["historical_peak"]
    first = supply["decisions"][0]
    if set(first["prior_directions_in_use"]) != {"False", "True"} or first["duration_estimates_s"] != prior:
        raise ValueError("Cold-start estimate ignored the fitted prior")


def summarize(cells, sources, priors):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required or set(priors) != set(spec.ROOTS):
        raise ValueError("Incomplete calibrated supply matrix or prior roots")
    metrics = spec.native.source.source_spec.authority.routing.METRICS
    all_rows, panels, diagnostics = {r: [] for r in spec.ROOTS}, [], []
    for (root, scenario), cell in cells.items():
        prior = {str(d): v for d, v in priors[root].items()}
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
                and cell["training_updates"] == 0 and cell["native_steps"] == 16*61380
                and cell["duration_prior_s"] == prior):
            raise ValueError("Invalid calibrated supply contract, prior or budget")
        source = sources[root, scenario]
        if not (source["protocol"] == spec.source.EXPERIMENT_PROTOCOL and source["contract"] == spec.source.contract()
                and source["software_qualified"] and source["seed"] == root and source["scenario"] == scenario
                and source["training_updates"] == 0 and source["native_steps"] == 20*61380):
            raise ValueError("Invalid calibrated supply source")
        scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
        rows = {(r["condition"], r["scene_seed"]): r for r in cell["evaluation"]}
        if len(rows) != len(cell["evaluation"]) or set(rows) != {(c, s) for c in spec.CONDITIONS for s in scenes}:
            raise ValueError("Incomplete or duplicate calibrated supply episodes")
        for (condition, scene), row in rows.items():
            ref = next(r for r in source["evaluation"] if r["condition"] == spec.SOURCE_CONDITIONS[condition] and r["scene_seed"] == scene)
            reproduced = condition in spec.CONTROLS
            if not (row["scenario"] == scenario and row["source_control_reproduced"] == reproduced
                    and row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and np.all(np.isfinite([row[k] for k in metrics]))):
                raise ValueError("Changed calibrated supply physical controls")
            spec.check_episode(row, ref, baseline=reproduced)
            validate_schedule(row["schedule"], ref["schedule"])
            if reproduced and row["schedule"] != ref["schedule"]:
                raise ValueError("Changed calibrated supply source-control schedule")
            if condition == "legacy_supply" and row["supply"] != ref["supply"]:
                raise ValueError("Changed legacy supply selector")
            validate_supply({**row, "condition": spec.SOURCE_CONDITIONS[condition]})
            if condition.startswith("calibrated_"):
                validate_prior(row["supply"], prior)
            all_rows[root].append(row)
        panels.extend({"root": root, "scenario": scenario, **contrast(list(rows.values()), c, ref, metrics)} for c, ref in CONTRASTS)
    for root, rows in all_rows.items():
        for condition in ("legacy_supply", "calibrated_demand", "calibrated_supply"):
            selected = [r for r in rows if r["condition"] == condition]
            decisions = [d for r in selected for d in r["supply"]["decisions"]]
            counts = np.sum([r["supply"]["choice_counts"] for r in selected], axis=0)
            entry = {"root": root, "condition": condition, "decisions": len(decisions),
                "choice_counts": counts.tolist(), "selected_mean_strength": float(counts@np.asarray(spec.STRENGTHS)/counts.sum()),
                "candidate_fleet_cost_distinguishes": int(sum(np.ptp([c["fleet_cost"] for c in d["candidates"]]) > 0 for d in decisions)),
                "fleet_term_changes_wait_only_argmin": sum(d["selected"] != min(range(5), key=lambda i: d["candidates"][i]["wait_cost"]) for d in decisions),
                "first_decision_candidate_peak_range": [min(c["predicted_peak"] for r in selected for c in r["supply"]["decisions"][0]["candidates"]),
                    max(c["predicted_peak"] for r in selected for c in r["supply"]["decisions"][0]["candidates"])]}
            if condition.startswith("calibrated_"):
                entry.update(prior_in_use_decisions=sum(bool(d["prior_directions_in_use"]) for d in decisions),
                    historical_peak_plateau_decisions=sum(max(c["predicted_peak"] for c in d["candidates"]) == d["historical_peak"] for d in decisions))
            diagnostics.append(entry)
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 160*61380, "source_control_reproductions": 120,
        "duration_prior_s": {str(r): {str(d): v for d, v in p.items()} for r, p in priors.items()},
        "paired_contrasts": [{"root": root, **contrast(rows, c, ref, metrics)} for root, rows in all_rows.items() for c, ref in CONTRASTS],
        "regime_root_contrasts": panels, "supply_diagnostics": diagnostics, "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells},
        {r: spec.duration_prior(r, directory/"inputs/calibration_summary.json") for r in spec.ROOTS})
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_contrasts"}}, indent=2))


if __name__ == "__main__":
    main()
