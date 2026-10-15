#!/usr/bin/env python3
"""Compare proxy ranks with single-action native terminal continuations."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_candidate_value_stage173 as spec
from freq_hrl.experiments.pointmaze_root_response import write_json


def ranking_pairs(predicted, actual):
    counts = {"concordant": 0, "discordant": 0, "prediction_ties": 0, "outcome_ties": 0}
    for i in range(6):
        for j in range(i):
            p, a = np.sign(predicted[i]-predicted[j]), np.sign(actual[i]-actual[j])
            if a == 0:
                counts["outcome_ties"] += 1
            elif p == 0:
                counts["prediction_ties"] += 1
            else:
                counts["concordant" if p == a else "discordant"] += 1
    return counts


def validate_continuations(baseline, branches, *, candidates=spec.CANDIDATES):
    base = baseline["probe"]
    index = base["index"]
    if [r["probe"]["selected"] for r in branches] != list(candidates):
        raise ValueError("Incomplete or duplicate native candidate continuations")
    for candidate, row in zip(candidates, branches):
        probe = row["probe"]
        same = {k: v for k, v in probe.items() if k not in {"selected", "physical_state_matched", "rng_matched"}}
        ref = {k: v for k, v in base.items() if k not in {"selected", "physical_state_matched", "rng_matched"}}
        reproduced = candidate == base["greedy"]
        spec.source.incumbent.source.distributional.source.check_episode(row, baseline, baseline=reproduced)
        spec.validate_schedule(row["schedule"], baseline["schedule"])
        trace, windows = row["policy_trace"], row["window_panels"]
        if not (row["condition"] == f"candidate_{candidate}" and row["scenario"] == baseline["scenario"]
                and row["scene_seed"] == baseline["scene_seed"] and same == ref
                and probe["physical_state_matched"] and probe["rng_matched"]
                and row["source_control_reproduced"] == reproduced and row["window_ledger_qualified"]
                and len(trace) == len(windows) == 44 and row["N_fleet"] == 12
                and row["simulation_end_time_s"] == 61380 and np.all(np.isfinite([row[k] for k in spec.METRICS]))):
            raise ValueError("Changed matched native candidate state, outcome or control")
        if trace[:index] != baseline["policy_trace"][:index]:
            raise ValueError("Native candidate changed its pre-intervention action trace")
        for i, (decision, window) in enumerate(zip(trace, windows)):
            future = next((d for d in trace[i+1:] if d["direction"] == decision["direction"]), None)
            end = future["time_s"] if future is not None else 61380
            if not (decision["selected"] == (candidate if i == index else decision["greedy"])
                    and decision["selected"] in spec.CANDIDATES
                    and all(decision[k] == window[k] for k in ("time_s", "direction", "selected"))
                    and window["window_end_s"] == end
                    and np.isclose(window["actual_window_fleet_cost"], max(window["actual_window_peak"]-12, 0)**2/12,
                        rtol=0, atol=1e-10)):
                raise ValueError("Native candidate changed its intervention, continuation or window accounting")
        if trace[index] != {k: probe[k] for k in ("time_s", "direction", "selected", "greedy")}:
            raise ValueError("Native candidate probe differs from the executed action")
        if reproduced and any(row[k] != baseline[k] for k in ("schedule", "policy_trace", "window_panels")):
            raise ValueError("Unchanged native action failed exact identity reproduction")


def panel(cell, registered, model):
    index, baseline = spec.probe_index(registered), cell["observer"]
    original, corrected = registered["evaluation"][1], registered["evaluation"][3]
    if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
            and cell["software_qualified"] and cell["training_updates"] == 0 and cell["native_steps"] == 7*61380
            and cell["seed"] == registered["seed"] and cell["scenario"] == registered["scenario"]
            and cell["probe_index"] == index and cell["corrected_probe"] == corrected["supply"]["decisions"][index]):
        raise ValueError("Changed candidate-value contract, source probe or budget")
    spec.qualify(baseline, {**original, "probe": baseline["probe"]})
    spec.validate_scoring({**baseline, "condition": "prefix_wait_full_fleet"}, model,
        strengths=spec.source.incumbent.CANDIDATES["rolling_plus_incumbent"])
    base = baseline["probe"]
    expected = original["supply"]["decisions"][index]
    if not (base["index"] == index and base["candidates"] == expected["candidates"]
            and base["time_s"] == expected["time_s"] and base["direction"] == expected["direction"]
            and base["selected"] == base["greedy"] == expected["selected"]
            and base["prefix_decisions_qualified"] == index+1 and np.isfinite(base["cost_before"])
            and baseline["policy_trace"] == spec.policy_trace(baseline["supply"])):
        raise ValueError("Changed candidate-value source state or selector")
    branches = cell["branches"]
    validate_continuations(baseline, branches)
    costs = [r["service_cost_restricted"] for r in branches]
    fleets = [max(r["peak_fleet"]-12, 0)**2/12 for r in branches]
    old, new = base["greedy"], cell["corrected_probe"]["selected"]
    best = min(spec.CANDIDATES, key=lambda i: costs[i])
    return {"root": cell["seed"], "scenario": cell["scenario"], "scene_seed": baseline["scene_seed"],
        "probe_index": index, "time_s": base["time_s"], "original_choice": old, "corrected_choice": new,
        "realized_best_candidate": best, "original_realized_regret": costs[old]-costs[best],
        "corrected_realized_regret": costs[new]-costs[best], "actual_costs": costs,
        "actual_waits_min": [r["restricted_wait_horizon_min"] for r in branches],
        "actual_peaks": [r["peak_fleet"] for r in branches],
        "original_score_ranks": ranking_pairs([c["score"] for c in base["candidates"]], costs),
        "corrected_score_ranks": ranking_pairs([c["score"] for c in cell["corrected_probe"]["candidates"]], costs),
        "original_fleet_ranks": ranking_pairs([c["fleet_cost"] for c in base["candidates"]], fleets),
        "corrected_fleet_ranks": ranking_pairs([c["fleet_cost"] for c in cell["corrected_probe"]["candidates"]], fleets),
        "corrected_choice_minus_original": {**{k: branches[new][k]-branches[old][k] for k in spec.METRICS},
            "fleet_cost_delta": fleets[new]-fleets[old],
            "nonfleet_cost_delta": costs[new]-costs[old]-(fleets[new]-fleets[old])}}


def summarize(cells, sources, pack, source_pack):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required or pack != source_pack:
        raise ValueError("Incomplete candidate-value matrix or changed fitted inputs")
    panels = [panel(cells[k], sources[k], pack["duration_models"][str(k[0])]) for k in sorted(required)]
    summaries = []
    for root in spec.ROOTS:
        group = [p for p in panels if p["root"] == root]
        summaries.append({"root": root, "states": len(group),
            "states_with_peak_discrimination": sum(len(set(p["actual_peaks"])) > 1 for p in group),
            "original_realized_regret_mean": float(np.mean([p["original_realized_regret"] for p in group])),
            "corrected_realized_regret_mean": float(np.mean([p["corrected_realized_regret"] for p in group])),
            "corrected_choice_minus_original": {k: float(np.mean([p["corrected_choice_minus_original"][k] for p in group]))
                for k in (*spec.METRICS, "fleet_cost_delta", "nonfleet_cost_delta")}})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 70*61380, "source_control_reproductions": 20,
        "window_ledger_qualified_episodes": 70, "paired_interventions": 60,
        "root_diagnostics": summaries, "state_panels": panels, "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory, source = ROOT/"results"/args.run_name, ROOT/"results"/spec.SOURCE_RUN
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells, {k: spec.source_cell(*k) for k in cells},
        json.loads((directory/"inputs/models.json").read_text()), json.loads((source/"inputs/models.json").read_text()))
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "state_panels"}}, indent=2))


if __name__ == "__main__":
    main()
