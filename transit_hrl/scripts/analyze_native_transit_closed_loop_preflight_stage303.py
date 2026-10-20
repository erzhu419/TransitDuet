#!/usr/bin/env python3
"""Qualify saved Stage303 histories and independent menus without outcome selection."""

import argparse
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_closed_loop_preflight_stage303 as source
from freq_hrl.experiments.pointmaze_root_response import write_json

SOURCE = "native_transit_closed_loop_preflight_stage303_20261021_r1"
PROTOCOL = "native_transit_closed_loop_preflight_audit_stage303_v1"


def validate_panel(records, row, root, scenario, probe, draws):
    if [r["draw"] for r in records] != draws: raise ValueError("Missing P/S draws")
    for r in records:
        if r["future_seed"] != source.future_seed(root, scenario, probe, r["draw"]) or sorted(map(int, r["values"])) != list(range(6)):
            raise ValueError("Wrong inner future or action menu")
        v = np.asarray([r["values"][str(a)] for a in range(6)])
        if not np.isfinite(v).all() or np.any(v[row["original_choice"]] != 0) or np.max(abs(v[:, 0]-v[:, 1]-v[:, 2])) > 1e-12:
            raise ValueError("Identity or native cost decomposition failed")
    return np.asarray([[[r["values"][str(a)] for a in range(6)] for r in records]])


def run_audit(run):
    from freq_hrl.domains.transit.native_timetable_context import qualify_timetable
    episodes, draw_records, seed_ids, feedback, groups, changed = 0, 0, set(), [], [], dict.fromkeys(source.ARMS, 0)
    witnesses = 0
    for root in source.source.ROOTS:
        source.parent.native.native_inputs(root)
        for scenario in source.source.SCENARIOS:
            saved = {}
            for arm in source.ARMS:
                path = source.directory(SOURCE)/"cells"/str(root)/scenario/arm
                cell = source.source.read(path/"result.json"); saved[arm] = cell
                if (cell["contract"] != source.contract() or not cell["software_qualified"] or cell["performance_claim"]
                        or cell["profile"] != 0 or cell["root"] != root or cell["scenario"] != scenario or cell["arm"] != arm
                        or cell["native_episodes"] != (5 if arm == "raw" else 227)
                        or [d["probe"] for d in cell["decisions"]] != list(source.PROBES)):
                    raise ValueError("Incomplete native pilot receipt")
                episodes += cell["native_episodes"]
                if arm == "raw":
                    witness = cell["nontrivial_history_witness"]
                    row = cell["decisions"][0]["row"]
                    curves = [c["commitment"]["prefix_planned_s"] for c in row["candidates"]]
                    if not witness["exact_identity"] or curves[witness["forced_candidate"]] == curves[row["original_choice"]]:
                        raise ValueError("Forced-history witness was trivial or failed")
                    witnesses += 1
                history = {}
                for d in cell["decisions"]:
                    row, probe = d["row"], d["probe"]
                    if d["history_before"] != history or row["probe_index"] != probe or row["scene_seed"] != source.scene_seed(root, scenario):
                        raise ValueError("Second decision does not use its controller's prior action")
                    qualify_timetable({"timetable_context": row["timetable_context"]}, row)
                    panels = []
                    if arm != "raw":
                        for draws in (list(range(8)), list(range(8, 16))):
                            records = source.source.read(path/f"probe{probe}/draws{draws[0]}to{draws[-1]}.json")
                            panels.append(validate_panel(records, row, root, scenario, probe, draws))
                            draw_records += len(records); seed_ids.update(r["future_seed"] for r in records)
                    if source.choose(SOURCE, arm, row, *(panels or (None, None))) != d["selected"]:
                        raise ValueError("Saved action differs from frozen controller")
                    history[str(probe)] = d["selected"]
            rows = {a: saved[a]["decisions"] for a in source.ARMS}
            if any(rows[a][0]["row"] != rows["raw"][0]["row"] for a in source.ARMS):
                raise ValueError("Controllers did not start from the same causal state")
            for a in source.ARMS:
                changed[a] += sum(x["selected"] != y["selected"] for x, y in zip(rows[a], rows["raw"]))
            for a, b in combinations(source.ARMS, 2):
                different = rows[a][0]["selected"] != rows[b][0]["selected"]
                state_changed = rows[a][1]["row"] != rows[b][1]["row"]
                if not different and state_changed: raise ValueError("Identical past actions produced mismatched second states")
                feedback.append({"root": root, "scenario": scenario, "pair": [a, b],
                    "first_action_differs": different, "second_context_differs": state_changed})
            groups.append([root, scenario])
    if episodes != 4590 or draw_records != 640 or len(seed_ids) != 320 or witnesses != 10:
        raise ValueError("Native preflight physical or menu budget differs")
    write_json(ROOT/"results"/run/"report/result.json", {"protocol": PROTOCOL, "source": SOURCE,
        "all_pilots_qualified": True, "qualified_cells": 30, "native_episodes_qualified_at_execution": episodes,
        "P8S8_draw_records_rechecked": draw_records, "distinct_inner_seeds": len(seed_ids),
        "identity_menus_rechecked": draw_records, "nontrivial_history_witnesses": witnesses,
        "groups": groups, "selected_index_differences_vs_raw": changed, "feedback": feedback,
        "pilot_metrics_used_for_selection": False, "new_native_episodes": 0,
        "raw_reference_fits": 0, "risk_fits": 0, "actor_updates": 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--run-name", required=True)
    run_audit(p.parse_args().run_name)
