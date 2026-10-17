#!/usr/bin/env python3
"""Read existing V86 paths to locate costs without changing their controller."""

import argparse
import json
from pathlib import Path

import run_freqduet_v86_feedback_timing as timing
from run_freqduet_v85_recovery_review import count_stats


WINDOWS = ("0-900", "900-1800", "1800-3600", "3600-None")
METRICS = ("passenger_person_s", *timing.EXPOSURES, "holding_vehicle_s",
           "holding_person_s", "readiness_delay_sum_s", "launched_trips", "cumulative_cv_squared")


def paired_timing(control, recovery):
    origin = control["target_time_s"]
    samples = list(zip(control["samples"], recovery["samples"]))
    if [a["time_s"] for a, _ in samples] != [b["time_s"] for _, b in samples]:
        raise AssertionError("paired temporal grids differ")
    chains = [timing.vehicle_chain(row) for row in (control, recovery)]
    returned = [c["returned"] for c in chains]
    first_return = min(r["physical_time_s"] for r in returned) if all(returned) else None
    first_change = {}
    for label, keys in (("passenger", timing.EXPOSURES[:2]), ("holding", ("holding_vehicle_s",))):
        first_change[label] = next((b["time_s"] - origin for a, b in samples
            if any(a[k] != b[k] for k in keys)), None)
    before = [(a, b) for a, b in samples if first_return is not None and a["time_s"] <= first_return]
    pre_return = (sum(before[-1][1][k] - before[-1][0][k] for k in timing.EXPOSURES[:2]) if before else None)
    launches = [{r["trip_id"]: r for r in row["launches"]} for row in (control, recovery)]
    common = launches[0].keys() & launches[1].keys()
    shifted = [min(launches[0][k]["time_s"], launches[1][k]["time_s"]) - origin for k in common
               if launches[0][k]["time_s"] != launches[1][k]["time_s"]]
    return dict(first_change_s=first_change, first_target_return_s=None if first_return is None else first_return - origin,
        passenger_change_observed_before_target_return=bool(first_return is not None and first_change["passenger"] is not None
            and origin + first_change["passenger"] <= first_return),
        passenger_at_last_pre_return_grid_s=pre_return,
        matched_trips=len(common), shifted_matched_dispatches=len(shifted),
        first_matched_dispatch_shift_s=min(shifted) if shifted else None,
        reassigned_matched_trips=sum(launches[0][k]["bus_id"] != launches[1][k]["bus_id"] for k in common),
        unmatched_launches=len(launches[0].keys() ^ launches[1].keys()))


def review(root):
    old = json.loads((root / "summary/status.json").read_text())
    rows = {(ri, mode): json.loads((root / "arms" / f"a{i}.json").read_text())
            for i, (ri, mode) in enumerate(timing.ITEMS)}
    counts, contexts = {}, []
    for control in timing.recovery.propagation.MODES:
        counts[control] = {window: {m: count_stats([c["effects"][control][window][m] for c in old["contexts"]])
                          for m in METRICS} for window in WINDOWS}
    for context in old["contexts"]:
        ci = context["context_index"]
        refs = [ri for ri, (c, _) in enumerate(timing.repeated.REFERENCES) if c == ci]
        effects = {}
        for control in timing.recovery.propagation.MODES:
            pairs = [paired_timing(rows[ri, control], rows[ri, "recover_900s"]) for ri in refs]
            effects[control] = dict(
                windows={w: {m: [s["mean"], s["ci95"]] for m, s in context["effects"][control][w].items()
                             if m in METRICS} for w in WINDOWS},
                vehicle_chain=context["target_vehicle"][control],
                pre_return_passenger=timing.repeated.describe([p["passenger_at_last_pre_return_grid_s"] for p in pairs])
                    if all(p["passenger_at_last_pre_return_grid_s"] is not None for p in pairs) else None,
                passenger_change_before_return=sum(p["passenger_change_observed_before_target_return"] for p in pairs),
                paths_with_matched_dispatch_shift=sum(p["shifted_matched_dispatches"] > 0 for p in pairs),
                paths_with_vehicle_reassignment=sum(p["reassigned_matched_trips"] > 0 for p in pairs),
                unmatched_launches=sum(p["unmatched_launches"] for p in pairs),
                first_passenger_change_s=[p["first_change_s"]["passenger"] for p in pairs],
                first_matched_dispatch_shift_s=[p["first_matched_dispatch_shift_s"] for p in pairs],
                first_target_return_s=[p["first_target_return_s"] for p in pairs])
        contexts.append({**{k: context[k] for k in ("context_index", "seed", "day", "stratum")}, "effects": effects})
    result = dict(status="v86_timing_review_complete", counts=counts, contexts=contexts,
        scope="conditional eight-future intervals on consumed prefixes; timing, not mediator identification",
        actors_updated=False, main_policy_changed=False)
    timing.pilot.write_json(root / "review/compact.json", result)
    for control, windows in counts.items():
        for window, metrics in windows.items():
            print("V86_WINDOW " + json.dumps(dict(control=control, window=window,
                metrics={m: [s["mean"], s["ci_negative"], s["ci_positive"]] for m, s in metrics.items()})), flush=True)
    for c in contexts:
        print("V86_CHAIN " + json.dumps({**{k: c[k] for k in ("context_index", "seed", "day", "stratum")},
            "effects": {mode: {k: v for k, v in e.items() if k != "windows"} for mode, e in c["effects"].items()}}), flush=True)
    timing.pilot.write_json(root / "review_ready/done.json", dict(status="v86_timing_review_complete"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    review(parser.parse_args().out_root)


if __name__ == "__main__":
    main()
