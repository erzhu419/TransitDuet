#!/usr/bin/env python3
"""Qualify executed score horizons and decompose matched physical outcomes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_commitment_scoring_stage168 as spec
from scripts.analyze_native_transit_calibrated_supply_stage165 import validate_prior
from scripts.analyze_native_transit_distributional_supply_stage166 import validate_distributional
from scripts.analyze_native_transit_commitment_audit_stage167 import validate_audits
from scripts.analyze_native_transit_rolling_authority_stage161 import validate_schedule
from scripts.analyze_native_transit_supply_allocation_stage163 import contrast
from freq_hrl.domains.transit.native_commitment_audit import fleet_cost
from freq_hrl.experiments.pointmaze_root_response import write_json

CONTRASTS = [(c, ref) for c in spec.HORIZONS for ref in spec.CONTROLS]


def validate_scoring(row, model, *, strengths=spec.distributional.STRENGTHS):
    supply = row["supply"]
    if row["condition"] == "full_wait_full_fleet":
        validate_distributional(supply, model)
    else:
        validate_prior(supply, model["duration_prior_s"])
        if not (supply["condition"] == "supply_aware" and supply["fleet_model"] == spec.distributional.FLEET_MODEL
                and supply["ensemble_members"] == 64 and supply["ensemble_seed"] == model["ensemble_seed"]
                and supply["fit_scene_seeds"] == model["fit_scene_seeds"] and len(supply["decisions"]) == 44):
            raise ValueError("Changed commitment scoring model or decision budget")
        prefix_wait, prefix_fleet = spec.HORIZONS[row["condition"]]
        choices, times, completions = [0]*len(strengths), [], []
        for d in supply["decisions"]:
            if [c["strength"] for c in d["candidates"]] != list(strengths):
                raise ValueError("Changed commitment scoring candidates")
            for c in d["candidates"]:
                p = c["commitment"]
                for counts in (c["ensemble_peak_counts"], p["prefix_peak_counts"]):
                    if not (sum(counts.values()) == 64 and all(isinstance(n, int) and n > 0
                            and int(peak) >= d["historical_peak"] for peak, n in counts.items())):
                        raise ValueError("Invalid commitment member counts or historical floor")
                score = (p["prefix_wait_cost"] if prefix_wait else c["wait_cost"]) + (
                    p["prefix_fleet_cost"] if prefix_fleet else c["fleet_cost"])
                if not (np.all(np.isfinite([c[k] for k in ("wait_cost", "fleet_cost", "score", "predicted_peak", "headway_cost")]))
                        and np.isfinite(p["prefix_wait_cost"]) and p["window_end_s"] > d["time_s"]
                        and np.isclose(c["fleet_cost"], fleet_cost(c["ensemble_peak_counts"]), rtol=0, atol=1e-10)
                        and p["full_fleet_cost"] == c["fleet_cost"]
                        and np.isclose(p["prefix_fleet_cost"], fleet_cost(p["prefix_peak_counts"]), rtol=0, atol=1e-10)
                        and np.isclose(p["prefix_score"], p["prefix_wait_cost"]+p["prefix_fleet_cost"], rtol=0, atol=1e-10)
                        and np.isclose(c["score"], score, rtol=0, atol=1e-10)):
                    raise ValueError("Changed full/prefix score terms or horizon")
            if d["selected"] != min(range(len(strengths)), key=lambda i: d["candidates"][i]["score"]):
                raise ValueError("Commitment selector did not execute its score minimum")
            choices[d["selected"]] += 1
            times.append(d["time_s"])
            completions.append(d["observed_completions"])
        if not (choices == supply["choice_counts"] and np.all(np.diff(times) >= 0)
                and np.all(np.diff(completions) >= 0) and completions[-1] == supply["completed_trip_count"]):
            raise ValueError("Inconsistent commitment choices or observations")
    decisions, panels = supply["decisions"], row["window_panels"]
    if not row["window_ledger_qualified"] or len(panels) != len(decisions):
        raise ValueError("Missing selected-window native ledger qualification")
    for i, (d, p) in enumerate(zip(decisions, panels)):
        future = next((n for n in decisions[i+1:] if n["direction"] == d["direction"]), None)
        end = future["time_s"] if future is not None else row["simulation_end_time_s"]
        peak = future["historical_peak"] if future is not None else row["peak_fleet"]
        if not (all(d[k] == p[k] for k in ("time_s", "direction", "selected"))
                and p["window_end_s"] == end and p["actual_window_peak"] == peak
                and np.isclose(p["actual_window_fleet_cost"], max(peak-12, 0)**2/12, rtol=0, atol=1e-10)):
            raise ValueError("Selected-window accounting differs from executed native callbacks")
        if row["condition"] in spec.HORIZONS:
            chosen = d["candidates"][d["selected"]]["commitment"]
            if not (chosen["window_end_s"] == end and chosen["prefix_fleet_cost"] == p["selected_prefix_fleet_cost"]):
                raise ValueError("Selected prefix score does not match the executed window")


def factorial_effects(rows, metrics):
    effects = {k: [] for k in ("prefix_wait_main_effect", "prefix_fleet_main_effect", "interaction")}
    for reference in (r for r in rows if r["condition"] == "full_wait_full_fleet"):
        group = {r["condition"]: r for r in rows if r["scenario"] == reference["scenario"]
            and r["scene_seed"] == reference["scene_seed"]}
        a, b, c, d = [group[k] for k in spec.CONDITIONS[1:]]
        for name, coefficients in zip(effects, ((-.5, -.5, .5, .5), (-.5, .5, -.5, .5), (1, -1, -1, 1))):
            effects[name].append({m: float(sum(w*r[m] for w, r in zip(coefficients, (a, b, c, d)))) for m in metrics})
    return {name: {m: float(np.mean([r[m] for r in records])) for m in metrics} for name, records in effects.items()}


def initial_window_candidate_coverage(cells):
    coverage = []
    for root in spec.ROOTS:
        available, total = 0, 0
        for (r, _), cell in cells.items():
            if r != root:
                continue
            for row in (v for v in cell["evaluation"] if v["condition"] == "prefix_wait_full_fleet"):
                ref = next(v for v in cell["evaluation"] if v["condition"] == "window_forecast" and v["scene_seed"] == row["scene_seed"])
                for direction in (False, True):
                    first = next(d for d in row["supply"]["decisions"] if d["direction"] == direction)
                    baseline = next(v for v in ref["schedule"]["schedules"] if v["direction"] == direction)["planned_s"][:6]
                    available += any(c["commitment"]["prefix_planned_s"][:6] == baseline for c in first["candidates"])
                    total += 1
        coverage.append({"root": root, "initial_direction_decisions": total, "window_prefix_available": available})
    return coverage


def summarize(cells, sources, window_sources, models):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required or set(window_sources) != required or set(models) != set(spec.ROOTS):
        raise ValueError("Incomplete commitment scoring matrix")
    metrics = spec.native.source.source_spec.authority.routing.METRICS
    all_rows, panels = {r: [] for r in spec.ROOTS}, []
    for root, model in models.items():
        spec.distributional.validate_model(model)
        if model != spec.distributional.fit_model(root):
            raise ValueError("Commitment scoring changed the fit-only model")
    for (root, scenario), cell in cells.items():
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
                and cell["training_updates"] == 0 and cell["native_steps"] == 20*61380):
            raise ValueError("Changed commitment scoring contract or budget")
        scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
        rows = {(r["condition"], r["scene_seed"]): r for r in cell["evaluation"]}
        if len(rows) != len(cell["evaluation"]) or set(rows) != {(c, s) for c in spec.CONDITIONS for s in scenes}:
            raise ValueError("Incomplete or duplicate commitment scoring episodes")
        for (condition, scene), row in rows.items():
            ref = next(r for r in (window_sources[root, scenario] if condition == "window_forecast" else sources[root, scenario])["evaluation"]
                if r["scene_seed"] == scene and r["condition"] == (
                    "window_forecast" if condition == "window_forecast" else "distributional_supply"))
            reproduced = condition in spec.CONTROLS
            if not (row["scenario"] == scenario and row["source_control_reproduced"] == reproduced
                    and row["N_fleet"] == 12 and row["simulation_end_time_s"] == 61380
                    and np.all(np.isfinite([row[k] for k in metrics]))):
                raise ValueError("Changed commitment scoring physical controls")
            spec.distributional.source.check_episode(row, ref, baseline=reproduced)
            validate_schedule(row["schedule"], ref["schedule"])
            if reproduced and (row["schedule"] != ref["schedule"] or row["supply"] != ref["supply"]):
                raise ValueError("Changed commitment scoring source trajectory")
            if condition == "window_forecast":
                if row["supply"] is not None:
                    raise ValueError("Window control must not use a supply selector")
            else:
                validate_scoring(row, models[root])
                if reproduced and row["window_panels"] != validate_audits(ref):
                    raise ValueError("Changed full-score control window accounting")
            all_rows[root].append(row)
        panels.extend({"root": root, "scenario": scenario, **contrast(list(rows.values()), c, ref, metrics)} for c, ref in CONTRASTS)
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 200*61380, "source_control_reproductions": 80,
        "window_ledger_qualified_episodes": 160,
        "paired_contrasts": [{"root": r, **contrast(rows, c, ref, metrics)} for r, rows in all_rows.items() for c, ref in CONTRASTS],
        "regime_root_contrasts": panels,
        "factorial_effects": [{"root": r, **factorial_effects(rows, metrics)} for r, rows in all_rows.items()],
        "posthoc_initial_window_candidate_coverage": initial_window_candidate_coverage(cells),
        "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    models = {int(r): m for r, m in json.loads((directory/"inputs/duration_models.json").read_text()).items()}
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells},
        {(r, s): spec.source.source_cell(r, s) for r, s in cells}, models)
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "regime_root_contrasts"}}, indent=2))


if __name__ == "__main__":
    main()
