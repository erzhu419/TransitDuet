#!/usr/bin/env python3
"""Measure committed-window forecast errors without changing the policy."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_commitment_audit_stage167 as spec
from freq_hrl.domains.transit.native_commitment_audit import fleet_cost, observed_window_peak
from freq_hrl.experiments.pointmaze_root_response import write_json


def validate_audits(row):
    audits, decisions, trips = row["audits"], row["supply"]["decisions"], row["completed_trips"]
    if len(audits) != len(decisions) or len({t["trip_id"] for t in trips}) != len(trips):
        raise ValueError("Incomplete audit or duplicate native trip ledger")
    if not all(np.all(np.isfinite([t["launch_time_s"], t["completion_time_s"]]))
               and 0 <= t["launch_time_s"] < t["completion_time_s"] <= row["simulation_end_time_s"] for t in trips):
        raise ValueError("Invalid completed native trip times")
    scheduled = {}
    for direction in (False, True):
        group = sorted((t for t in trips if bool(t["direction"]) == direction), key=lambda t: t["trip_id"])
        schedule = next(s for s in row["schedule"]["schedules"] if s["direction"] == direction)
        if [t["launch_time_s"] for t in group] != schedule["actual_s"]:
            raise ValueError("Audit ledger does not match native actual launches")
        scheduled.update(zip([t["trip_id"] for t in group], schedule["planned_s"]))
    if len(scheduled) != row["schedule"]["scheduled_trips"] or observed_window_peak(trips, 0, row["simulation_end_time_s"], 0) != row["peak_fleet"]:
        raise ValueError("Audit ledger does not reproduce full native trip/peak accounting")
    panels = []
    for i, (audit, d) in enumerate(zip(audits, decisions)):
        if any(audit[k] != d[k] for k in ("time_s", "direction", "historical_peak", "selected")) or len(audit["candidates"]) != 5:
            raise ValueError("Passive audit differs from the executed original decision")
        ids, proposal = audit["proposed_trip_ids"], audit["selected_full_planned_s"]
        if len(ids) != len(proposal) or any(scheduled[t] != p for t, p in zip(ids[:7], proposal[:7])):
            raise ValueError("Committed prefix or tentative next-block anchor changed")
        for candidate, full in zip(audit["candidates"], d["candidates"]):
            h = candidate["prefix_peak_counts"]
            if not (sum(h.values()) == spec.source.MEMBERS and all(isinstance(n, int) and n > 0
                    and int(p) >= d["historical_peak"] for p, n in h.items())
                    and candidate["window_end_s"] > d["time_s"]
                    and candidate["full_fleet_cost"] == full["fleet_cost"]
                    and np.isclose(candidate["prefix_fleet_cost"], fleet_cost(h), rtol=0, atol=1e-10)
                    and np.isfinite(candidate["prefix_wait_cost"])
                    and np.isclose(candidate["prefix_score"], candidate["prefix_wait_cost"]+fleet_cost(h), rtol=0, atol=1e-10)):
                raise ValueError("Invalid passive prefix score, histogram or historical floor")
        chosen = audit["candidates"][d["selected"]]
        future = next((a for a in audits[i+1:] if a["direction"] == d["direction"]), None)
        end_s = future["time_s"] if future is not None else row["simulation_end_time_s"]
        next_id = future["proposed_trip_ids"][0] if future is not None else None
        if chosen["window_end_s"] != end_s or chosen["next_trip_id"] != next_id:
            raise ValueError("Predicted replan window does not match native callback/anchor")
        observed = observed_window_peak(trips, d["time_s"], end_s, d["historical_peak"])
        if observed != (future["historical_peak"] if future is not None else row["peak_fleet"]):
            raise ValueError("Window trip ledger does not match causal historical peak updates")
        actual = max(observed-12, 0)**2/12
        tail = [abs(scheduled[t]-p) for t, p in zip(ids[7:], proposal[7:])]
        panels.append({"time_s": d["time_s"], "direction": d["direction"], "window_end_s": end_s,
            "selected": d["selected"], "prefix_proxy_argmin": min(range(5), key=lambda k: audit["candidates"][k]["prefix_score"]),
            "full_fleet_distinguishes": bool(np.ptp([c["fleet_cost"] for c in d["candidates"]]) > 0),
            "prefix_fleet_distinguishes": bool(np.ptp([c["prefix_fleet_cost"] for c in audit["candidates"]]) > 0),
            "actual_window_peak": observed, "actual_window_fleet_cost": actual,
            "selected_prefix_fleet_cost": chosen["prefix_fleet_cost"], "selected_full_fleet_cost": chosen["full_fleet_cost"],
            "prefix_abs_cost_error": abs(actual-chosen["prefix_fleet_cost"]),
            "full_abs_window_cost_error": abs(actual-chosen["full_fleet_cost"]),
            "tail_proposed_trips": len(tail), "tail_rewritten_trips": sum(t > 0 for t in tail),
            "tail_shift_abs_sum_s": float(sum(tail))})
    return panels


def summarize(cells, sources):
    required = {(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}
    if set(cells) != required or set(sources) != required:
        raise ValueError("Incomplete commitment audit matrix")
    panels = []
    for (root, scenario), cell in cells.items():
        source = sources[root, scenario]
        if not (cell["protocol"] == spec.EXPERIMENT_PROTOCOL and cell["contract"] == spec.contract()
                and cell["software_qualified"] and cell["seed"] == root and cell["scenario"] == scenario
                and cell["training_updates"] == 0 and cell["native_steps"] == 4*61380):
            raise ValueError("Changed passive audit contract or budget")
        scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
        rows = {r["scene_seed"]: r for r in cell["evaluation"]}
        if len(rows) != len(cell["evaluation"]) or set(rows) != set(scenes):
            raise ValueError("Incomplete or duplicate passive audit scenes")
        for scene, row in rows.items():
            ref = next(r for r in source["evaluation"] if r["condition"] == "distributional_supply" and r["scene_seed"] == scene)
            spec.source.source.check_episode(row, ref, baseline=True)
            if not (row["scenario"] == scenario and row["condition"] == "distributional_supply"
                    and row["source_control_reproduced"] and row["schedule"] == ref["schedule"] and row["supply"] == ref["supply"]):
                raise ValueError("Passive audit changed native source physics, schedules or scores")
            panels.extend({"root": root, "scenario": scenario, "scene_seed": scene, **p} for p in validate_audits(row))
    summary = []
    for root in spec.ROOTS:
        selected = [p for p in panels if p["root"] == root]
        summary.append({"root": root, "decisions": len(selected),
            "full_only_fleet_discrimination": sum(p["full_fleet_distinguishes"] and not p["prefix_fleet_distinguishes"] for p in selected),
            "prefix_argmin_disagreements": sum(p["selected"] != p["prefix_proxy_argmin"] for p in selected),
            "tail_proposed_trips": sum(p["tail_proposed_trips"] for p in selected),
            "tail_rewritten_trips": sum(p["tail_rewritten_trips"] for p in selected),
            "mean_prefix_abs_cost_error": float(np.mean([p["prefix_abs_cost_error"] for p in selected])),
            "mean_full_abs_window_cost_error": float(np.mean([p["full_abs_window_cost_error"] for p in selected]))})
    return {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
        "training_updates": 0, "native_steps": 40*61380, "source_control_reproductions": 40,
        "audit_summary": summary, "decision_panels": panels, "stage": spec.contract()["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    directory = ROOT/"results"/args.run_name
    cells = {(r, s): json.loads((directory/"cells"/s/f"seed_{r}/result.json").read_text()) for r in spec.ROOTS for s in spec.SCENARIOS}
    result = summarize(cells, {(r, s): spec.source_cell(r, s) for r, s in cells})
    write_json(directory/"summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"contract", "decision_panels"}}, indent=2))


if __name__ == "__main__":
    main()
