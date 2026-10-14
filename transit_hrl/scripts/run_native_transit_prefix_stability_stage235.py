#!/usr/bin/env python3
"""Replay prefix contributions and block-deletion selection on Stage234's grid."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_cost_strength_stage234 as previous
from freq_hrl.domains.transit import native_prefix_stability as diagnostic
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

strength, joint, training = diagnostic.strength, diagnostic.joint, previous.training
EXPERIMENT_PROTOCOL = "native_transit_prefix_stability_stage235_v1"
SOURCE_RUN = "native_transit_cost_strength_stage234_development_20261014_r1"
VIEWS, PARTITIONS, KEYS, SCOPES = previous.VIEWS, previous.PARTITIONS, previous.KEYS, previous.SCOPES


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": parent["cohort"], "outer": parent["outer"], "partitions": parent["partitions"],
        "inner": parent["inner"], "candidates": strength.CANDIDATES, "anchors": list(joint.ANCHORS),
        "selectors": diagnostic.FIELDS, "normalization": parent["normalization"],
        "reproduction": "all64 Stage234 inner selections and clean Stage223 raw OOF references exactly",
        "deletion": "remove_both_validation_directions_of_one_prefix_rescore_existing_OOF_no_refit",
        "future": "rescore_one_inner_future_half_with_fixed_full_candidate_and_separate_reselection",
        "concentration": "signed_prefix_sums_and_top1_top3_shares_of_positive_benefit_by_source",
        "outer_scores": "unchanged_Stage234_actions_all4_scopes_by_source",
        "budget": {**parent["budget"], "full_selection_replays": 64,
            "validation_prefix_deletions": 640, "future_half_rescores": 128,
            "selector_deletion_comparisons": 1920, "new_outer_prediction_calls": 0},
        "boundary": parent["boundary"], "storage": parent["storage"],
        "limitation": "block_deletion_influence_not_independent_refit_or_unconfounded_reference_transfer"}


def result_path(run_name, *key): return previous.result_path(run_name, *key)


def fit_cell(train, panels, source, reused_oof):
    curves, ledger, reproduction = joint.collect_paths(train, panels, source["training_draws"], reused_oof,
        path_fitter=strength.fit_path)
    anchors = {a: diagnostic.diagnose(curves[a], ledger, a, source["selection"][a]) for a in joint.ANCHORS}
    outer = {a: {mode: {scope: {str(root): diagnostic.outer_stats([r for r in rows if r["root"] == root],
            f"cost_strength_{a}_{mode}") for root in sorted({r["root"] for r in rows})}
            for scope, rows in source["scopes"].items()} for mode in diagnostic.FIELDS} for a in joint.ANCHORS}
    return {"partition": source["partition"], "direction": source["direction"],
        "training_draws": source["training_draws"], "evaluation_draws": source["evaluation_draws"],
        "training_groups": source["training_groups"], "held_groups": source["held_groups"],
        "blocks": [{k: f[k] for k in ("held_group", "direction", "training_draws", "validation_draws", "held_states")} for f in ledger],
        "selection_reproduction": {"anchor_selections": 2, "differences": 0},
        "inner_reference_reproduction": reproduction, "anchors": anchors, "outer": outer}


def qualify(result, key, views):
    c = result["cell"]
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    groups = lambda rows: {(r["root"], r["scene_seed"]) for r in rows}
    a, b = PARTITIONS[key[2]][key[3]], PARTITIONS[key[2]][1-key[3]]
    if (result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract() or result["key"] != list(key)
            or c["partition"] != key[2] or c["direction"] != key[3]
            or c["training_draws"] != list(a) or c["evaluation_draws"] != list(b)
            or set(map(tuple, c["training_groups"])) != groups(train)
            or set(map(tuple, c["held_groups"])) != groups(held)):
        raise ValueError("Changed prefix diagnostic cohort")
    expected = {(g, d) for g in groups(train) for d in (0, 1)}
    blocks = c["blocks"]
    if len(blocks) != 20 or {(tuple(f["held_group"]), f["direction"]) for f in blocks} != expected:
        raise ValueError("Dropped validation prefix/direction")
    for f in blocks:
        halves = (list(a[:2]), list(a[2:]))
        if f["held_states"] != 6 or f["training_draws"] != halves[f["direction"]] or f["validation_draws"] != halves[1-f["direction"]]:
            raise ValueError("Changed inner future half")
    if c["selection_reproduction"] != {"anchor_selections": 2, "differences": 0}:
        raise ValueError("Missing selection reproduction")
    if c["inner_reference_reproduction"]["max_abs_difference"] > 1e-12 or c["inner_reference_reproduction"]["changed_choices"]:
        raise ValueError("Raw reference changed")
    for kind in joint.ANCHORS:
        d = c["anchors"][kind]
        if (len(d["prefixes"]) != 10 or {tuple(x["group"]) for x in d["prefixes"]} != groups(train)
                or len(d["deletions"]) != 10 or {tuple(x["removed_group"]) for x in d["deletions"]} != groups(train)
                or any(x["removed_blocks"] != 2 for x in d["deletions"])
                or len(d["future_halves"]) != 2 or {x["direction"] for x in d["future_halves"]} != {0, 1}):
            raise ValueError("Incomplete prefix influence")
        for mode in diagnostic.FIELDS:
            if sum(x["modes"][mode]["cases"] for x in d["prefixes"]) != 120:
                raise ValueError("Changed prefix validation coverage")
            if set(c["outer"][kind][mode]) != set(SCOPES) or any(sum(x["cases"] for x in s.values()) != 60
                    for s in c["outer"][kind][mode].values()):
                raise ValueError("Incomplete outer scopes")


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 prefix diagnostics")
    for key, r in results.items(): qualify(r, key, views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for key, r in results.items() if key[0] == view]
        output[view] = {"keys": [list(k) for k in KEYS if k[0] == view],
            "anchors": {a: diagnostic.summarize_anchor(cells, a) for a in joint.ANCHORS}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "views": output,
        "selection_reproduction": {"anchor_selections": 64, "differences": 0},
        "inner_reference_max_abs_difference": max(r["cell"]["inner_reference_reproduction"]["max_abs_difference"] for r in results.values()),
        **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("fit", "report"), required=True)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != previous.EXPERIMENT_PROTOCOL or prior["contract"] != previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage234 before prefix diagnostics")
    bank, _, registered, _ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"], registered)
    if args.operation == "fit":
        key = args.view, args.fold, args.partition, args.direction
        old = json.loads((raw_directory(previous.result_path(SOURCE_RUN, *key))/"result.json").read_text())
        previous.qualify(old, key, views)
        oof_path = raw_directory(training.precision_spec.result_path(training.binding_spec.SOURCE_RUN, *key))/"inner_oof.json"
        oof = {k: np.asarray(v) for k, v in json.loads(oof_path.read_text())["c2v4"].items()}
        train, _ = training.cohort.split_rows(views[args.view], args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell = fit_cell(train, panels, old["cell"], oof)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "cell": cell}
        qualify(result, key, views)
        path = result_path(args.run_name, *key)
        write_json(raw_directory(path)/"result.json", result)
        write_json(path, {k: v for k, v in result.items() if k != "cell"})
        print("NATIVE_PREFIX_STABILITY_FIT_COMPLETE", flush=True)
    else:
        report = summarize({k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS}, views)
        write_json(ROOT/"results"/args.run_name/"report/result.json", report)
        print("NATIVE_PREFIX_STABILITY_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
