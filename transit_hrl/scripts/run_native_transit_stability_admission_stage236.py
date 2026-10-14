#!/usr/bin/env python3
"""Score future-only, prefix-only and joint admission without new training."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_prefix_stability_stage235 as previous
from freq_hrl.domains.transit import native_stability_admission as admission
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

diagnostic, joint, training = admission.diagnostic, admission.joint, previous.training
EXPERIMENT_PROTOCOL = "native_transit_stability_admission_stage236_v1"
SOURCE_RUN = "native_transit_prefix_stability_stage235_development_20261015_r1"
OUTER_RUN = previous.SOURCE_RUN
VIEWS, PARTITIONS, KEYS, SCOPES = previous.VIEWS, previous.PARTITIONS, previous.KEYS, previous.SCOPES


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "outer_run": OUTER_RUN, "outer_protocol": previous.previous.EXPERIMENT_PROTOCOL,
        "cohort": parent["cohort"], "outer": parent["outer"], "partitions": parent["partitions"], "inner": parent["inner"],
        "candidates": parent["candidates"], "anchors": list(joint.ANCHORS), "modes": list(admission.MODES),
        "filters": list(admission.FILTERS), "variants": list(admission.VARIANTS),
        "pooled": "same_fixed_candidate_strict_pooled_cost_wait_improvement_each_source_safe_also_nonincreased_unsafe",
        "future_only": "fixed_candidate_nonharm_each_source_on_both_inner_future_halves",
        "prefix_only": "fixed_candidate_nonharm_each_source_after_each_validation_prefix_deletion",
        "joint": "intersection_of_future_only_and_prefix_only",
        "nonharm_tolerance": admission.NONHARM_TOL, "risk_tolerance": 0,
        "inference": "one_training_only_admission_per_cell_anchor_mode_no_reselection_no_outer_label_access",
        "actions": "accepted_equals_unchanged_Stage234_action_rejected_equals_unchanged_raw_action",
        "controls": "all27_Stage234_controls_and_Stage235_unfiltered_outer_deltas_exact",
        "budget": {"report_tasks": 1, "read_stage234_cells": 32, "read_stage235_cells": 32, "admission_decisions": 128,
            "source_choice_comparisons": 207360, "source_score_comparisons": 414720,
            "new_raw_reference_fits": 0, "new_residual_fits": 0, "network_updates": 0,
            "new_native_episodes": 0, "new_outer_prediction_calls": 0, "rl_actor_updates": 0},
        "boundary": parent["boundary"], "storage": parent["storage"]}


def aggregate(items):
    cells = [c for _, c in items]
    scopes = {s: [r for c in cells for r in c["scopes"][s]] for s in SCOPES}
    roots = sorted({r["root"] for r in scopes["held_other"]})
    counts = {a: {m: {"pooled_candidates": sum(c["admission"][a][m]["pooled_improvement"] for c in cells),
        "strategies": {f: {"accepted_cells": sum(c["admission"][a][m]["accepted"][f] for c in cells),
            "effectful_held_cells": sum(any(r["decisions"][admission.variant_name(a, m, f)]["selected"] != r["decisions"]["raw"]["selected"]
                for r in c["scopes"]["held_other"]) for c in cells)} for f in admission.FILTERS}}
        for m in admission.MODES} for a in joint.ANCHORS}
    return {"cells": len(cells), "admission_counts": counts, "scopes": {s: admission.metrics(rows) for s, rows in scopes.items()},
        "held_other_by_root": {str(root): admission.metrics([r for r in scopes["held_other"] if r["root"] == root]) for root in roots},
        "by_future_direction": {f"{p}/{d}": admission.metrics([r for k, c in items if k[2] == p and k[3] == d
            for r in c["scopes"]["held_other"]]) for p in PARTITIONS for d in (0, 1)}}


def summarize(cells):
    if set(cells) != set(KEYS): raise ValueError("Complete all32 matched admission cells")
    for c in cells.values():
        if set(c["scopes"]) != set(SCOPES) or any(len(rs) != 60 for rs in c["scopes"].values()):
            raise ValueError("Dropped admission scope/state")
        for rows in c["scopes"].values():
            if len({joint.binding.learned.state_key(r) for r in rows}) != 60 or any(set(r["decisions"]) != set(admission.VARIANTS) for r in rows):
                raise ValueError("Dropped admission controller/state")
    views = {v: aggregate([(k, cells[k]) for k in KEYS if k[0] == v]) for v in VIEWS}
    if any(x["prediction_cases"] != 480 or x["unique_states"] != 120 for v in views.values() for x in v["scopes"].values()):
        raise ValueError("Changed admission view coverage")
    mixed = aggregate([(k, cells[k]) for k in KEYS if k[0] in ("mixed_a", "mixed_b")])
    reproduction = {f: sum(c["source_reproduction"][f] for c in cells.values()) for f in (
        "candidate_comparisons", "choice_comparisons", "score_comparisons", "changed_choices", "outer_metric_records")}
    reproduction["max_abs_difference"] = max(c["source_reproduction"]["max_abs_difference"] for c in cells.values())
    if (reproduction["candidate_comparisons"] != 128 or reproduction["choice_comparisons"] != 207360
            or reproduction["score_comparisons"] != 414720 or reproduction["changed_choices"] or reproduction["max_abs_difference"]):
        raise ValueError("Incomplete source admission reproduction")
    per_cell = []
    for key in KEYS:
        c = cells[key]
        outer = {a: {m: {f: {scope: {str(root): diagnostic.outer_stats([r for r in rows if r["root"] == root],
                admission.control_name(a, m) if f == "unfiltered" else admission.variant_name(a, m, f))
                for root in sorted({r["root"] for r in rows})} for scope, rows in c["scopes"].items()}
                for f in ("unfiltered", *admission.FILTERS)} for m in admission.MODES} for a in joint.ANCHORS}
        per_cell.append({"key": list(key), "admission": c["admission"], "outer": outer})
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "views": views, "mixed_pooled": mixed, "cells": per_cell, "source_reproduction": reproduction,
        **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    for run, spec in ((SOURCE_RUN, previous), (OUTER_RUN, previous.previous)):
        report = json.loads((ROOT/"results"/run/"report/result.json").read_text())
        if report["protocol"] != spec.EXPERIMENT_PROTOCOL or report["contract"] != spec.contract() or not report["coverage_qualified"]:
            raise ValueError("Complete source results before stability admission")
    bank, _, registered, _ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"], registered)
    cells = {}
    for key in KEYS:
        old = json.loads((raw_directory(previous.previous.result_path(OUTER_RUN, *key))/"result.json").read_text())
        stable = json.loads((raw_directory(previous.result_path(SOURCE_RUN, *key))/"result.json").read_text())
        previous.previous.qualify(old, key, views)
        previous.qualify(stable, key, views)
        cells[key] = admission.admit_cell(old["cell"], stable["cell"])
    write_json(ROOT/"results"/args.run_name/"report/result.json", summarize(cells))
    print("NATIVE_STABILITY_ADMISSION_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
