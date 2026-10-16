#!/usr/bin/env python3
"""Read-only physical fleet and downstream-command accounting of V71 traces."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v71_mechanism_trace as audit


def indexed(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"physical events are not unique by {key}")
    return result


def fleet_effects(reference, candidate, action):
    first, bus, trip, _, _ = reference["identity"]
    old = indexed(reference["launches"], "trip_id")
    new = indexed(candidate["launches"], "trip_id")
    common = sorted(old.keys() & new.keys())
    shifts = np.asarray([new[t]["time_s"] - old[t]["time_s"] for t in common])
    ready = np.asarray([new[t]["readiness_s"] - old[t]["readiness_s"] for t in common])
    plan = np.asarray([new[t]["scheduled_s"] - old[t]["scheduled_s"] for t in common])
    reassigned = np.asarray([new[t]["bus_id"] != old[t]["bus_id"] for t in common])
    returned = audit.trip_return(reference, trip)
    next_trip = min((r for r in old.values() if r["bus_id"] == bus and r["time_s"] >= returned),
                    key=lambda r: r["time_s"])["trip_id"]
    ref_returns = indexed(reference["returns"], "trip_id")
    new_returns = indexed(candidate["returns"], "trip_id")
    other_returns = (ref_returns.keys() & new_returns.keys()) - {trip}

    def commands(data):
        remaining = [h for h in data["holds"] if h["time_s"] >= first]
        return np.asarray([sum(h["action_s"] for h in remaining if h["trip_id"] == trip),
                           sum(h["action_s"] for h in remaining if h["trip_id"] != trip and h["bus_id"] == bus),
                           sum(h["action_s"] for h in remaining if h["bus_id"] != bus)], dtype=float)

    command_delta = commands(candidate) - commands(reference)
    command_delta[0] -= action - reference["native_action_s"]
    return dict(actual_launch_times_changed=int((shifts != 0).sum()),
                actual_launches_advanced=int((shifts < 0).sum()),
                actual_launches_delayed=int((shifts > 0).sum()),
                actual_launch_time_shift_sum_s=float(shifts.sum()),
                readiness_only_launch_changes=int(((shifts == 0) & (ready != 0) & ~reassigned).sum()),
                published_plan_changes=int((plan != 0).sum()),
                published_plan_only_changes=int(((plan != 0) & (shifts == 0) & (ready == 0) & ~reassigned).sum()),
                vehicles_reassigned=int(reassigned.sum()),
                launches_only_reference=len(old.keys() - new.keys()), launches_only_candidate=len(new.keys() - old.keys()),
                target_return_delta_s=audit.trip_return(candidate, trip) - returned,
                reference_next_trip_launch_delta_s=new[next_trip]["time_s"] - old[next_trip]["time_s"],
                other_trip_returns_changed=sum(ref_returns[t]["physical_time_s"] != new_returns[t]["physical_time_s"]
                                               for t in other_returns),
                target_trip_other_command_s=float(command_delta[0]),
                target_vehicle_later_trip_command_s=float(command_delta[1]),
                other_vehicle_command_s=float(command_delta[2]))


def cleared_tail(data):
    timeline, holding = data["timeline"], data["holding"]
    last_dt = timeline[-1, 0] - timeline[-2, 0]
    physical_dt = holding[1, 0] - holding[0, 0]
    extension = last_dt > physical_dt
    if extension:
        np.testing.assert_equal(timeline[-2, 0], holding[-2, 0])
        np.testing.assert_array_equal(holding[-1, 1:], holding[-2, 1:])
    exposure = timeline[-1, 1:] - timeline[-2, 1:]
    return dict(physical_end_s=float(timeline[-2 if extension else -1, 0]),
                unserved=float(exposure[0] / last_dt),
                passenger_person_s=float(exposure[:2].sum()) if extension else 0.)


def censor_effects(reference, candidate):
    old, new = cleared_tail(reference), cleared_tail(candidate)
    common_end = max(old["physical_end_s"], new["physical_end_s"])
    total_delta = candidate["timeline"][-1, 1:3].sum() - reference["timeline"][-1, 1:3].sum()
    at_clear = audit.cumulative(candidate["timeline"], common_end) - audit.cumulative(reference["timeline"], common_end)
    return dict(reference_service_end_s=old["physical_end_s"], service_end_delta_s=new["physical_end_s"] - old["physical_end_s"],
                reference_unserved=old["unserved"], candidate_unserved=new["unserved"], unserved_delta=new["unserved"] - old["unserved"],
                own_cleared_tail_delta_person_s=new["passenger_person_s"] - old["passenger_person_s"],
                post_both_cleared_passenger_person_s=float(total_delta - at_clear[:2].sum()),
                common_cleared_tail_duration_s=float(reference["timeline"][-1, 0] - common_end))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                    str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v71_fleet.py", "-v"], check=True)
    root = args.out_root
    panels = []
    for context, (native, actions) in audit.PLAN.items():
        effects = {a: [] for a in actions}
        for future in audit.source.FUTURES:
            group = json.loads(audit.group_path(root, audit.ITEMS.index((context, future))).read_text())
            reference = torch.load(root / "traces" / f"c{context}_f{future}_native.pt", map_location="cpu", weights_only=False)
            for action in actions:
                candidate = torch.load(root / "traces" / f"c{context}_f{future}_a{action}.pt", map_location="cpu", weights_only=False)
                row = next(r for r in group["rows"] if r["action_s"] == action)
                physical = fleet_effects(reference, candidate, action)
                np.testing.assert_allclose(sum(physical[k] for k in ("target_trip_other_command_s",
                    "target_vehicle_later_trip_command_s", "other_vehicle_command_s")),
                    row["phases"]["day_end"]["other_command_holding_s"], atol=0, rtol=0)
                np.testing.assert_allclose(physical["target_return_delta_s"], row["target_return_delta_s"], atol=0, rtol=0)
                phases = row["phases"]
                effects[action].append(dict(physical=physical, censor=censor_effects(reference, candidate),
                    elapsed={cut: v["time_s"] - reference["identity"][0] for cut, v in phases.items() if cut != "before"},
                    post_return_passenger_person_s=phases["day_end"]["passenger_person_s"] - phases["reference_return"]["passenger_person_s"],
                    post_30min_passenger_person_s=phases["day_end"]["passenger_person_s"] - phases["min30"]["passenger_person_s"],
                    post_return_executed_holding_s=phases["day_end"]["executed_holding_s"] - phases["reference_return"]["executed_holding_s"],
                    sign_pairs={cut: (v["passenger_person_s"], row["passenger_person_s"])
                                for cut, v in phases.items() if cut in ("min15", "min30", "reference_return")}))
        for action, rows in effects.items():
            panels.append(dict(context_index=context, native_action_s=native, action_s=action,
                physical={k: audit.future_interval([r["physical"][k] for r in rows]) for k in rows[0]["physical"]},
                censor={k: audit.future_interval([r["censor"][k] for r in rows]) for k in rows[0]["censor"]},
                elapsed_seconds={cut: dict(mean=float(np.mean([r["elapsed"][cut] for r in rows])),
                                          min=min(r["elapsed"][cut] for r in rows), max=max(r["elapsed"][cut] for r in rows))
                                 for cut in ("reference_return", "reference_next_launch", "day_end")},
                tail={k: audit.future_interval([r[k] for r in rows]) for k in
                      ("post_return_passenger_person_s", "post_30min_passenger_person_s", "post_return_executed_holding_s")},
                sign_pairs={cut: dict(short_harm_full_benefit=sum(r["sign_pairs"][cut][0] > 0 and r["sign_pairs"][cut][1] < 0 for r in rows),
                                     short_benefit_full_harm=sum(r["sign_pairs"][cut][0] < 0 and r["sign_pairs"][cut][1] > 0 for r in rows),
                                     short_tie=sum(r["sign_pairs"][cut][0] == 0 for r in rows),
                                     full_tie=sum(r["sign_pairs"][cut][1] == 0 for r in rows))
                            for cut in ("min15", "min30", "reference_return")}))
    result = dict(status="fixed_V71_physical_fleet_and_censor_accounting_complete", panels=panels,
                  interval_unit="32 consumed continuations conditional on each fixed development prefix",
                  rollouts_added=0, actor_updates=0, dual_updates=0, value_updates=0, main_policy_changed=False,
                  causal_mediation_identified=False)
    output = root / "fleet_tail_summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V71_FLEET_COMPLETE " + json.dumps(dict(panels=len(panels), rollouts_added=0)), flush=True)


if __name__ == "__main__":
    main()
