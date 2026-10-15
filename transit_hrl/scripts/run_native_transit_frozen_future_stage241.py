#!/usr/bin/env python3
"""Freeze existing OOF actions, then collect independent conditional futures."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_refit_consistency_stage240 as prior
from scripts import run_native_transit_mean_coverage_stage195 as coverage
from freq_hrl.domains.transit import native_frozen_future as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_frozen_future_stage241_v1"
PREVIOUS_RUN = "native_transit_refit_consistency_stage240_development_20261015_r2"
SOURCE_RUN, REFIT_RUN = prior.SOURCE_RUN, prior.REFIT_RUN
KEYS, VIEWS, DRAWS = prior.KEYS, prior.VIEWS, tuple(range(8))
ROOTS, SCENARIOS = coverage.ROOTS, coverage.SCENARIOS


def cells(*, preflight=False):
    return [(*k, d) for k in coverage.states() for d in DRAWS
        if not preflight or (k[2] == 1 and k[3] == 9 and d == 0)]


def continuation_seed(root, scenario, replica, probe, draw):
    return 1241000000+ROOTS.index(root)*100000+SCENARIOS.index(scenario)*10000+replica*1000+probe*10+draw


def draw_path(run_name, *key):
    return coverage.draw_path(run_name, *key)


def sealed_path(run_name):
    return ROOT/"results"/run_name/"seal_raw/choices.json"


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": prior.previous.EXPERIMENT_PROTOCOL,
        "previous_run": PREVIOUS_RUN, "refit_run": REFIT_RUN,
        "cohort": "same240_development_states_40_prefixes_roots397_401_five_regimes_replicas0to3_six_probes",
        "selection": "all20_Stage239_candidates_and_640_inner_models_fixed_before_new_futures_no_reselection_or_refits",
        "actions": "seal_all21_action_vectors_on6_state_inner_validation_batches_reproduce12800_selected_OOF_records",
        "futures": "eight_new_postprobe_ScenarioTape_seeds_shared_across_all_six_actions_and_all32_diagnostic_cells",
        "future_seed": "1241000000+root_index*100000+regime_index*10000+replica*1000+probe*10+draw",
        "draws": list(DRAWS), "preflight": "both_roots_all5regimes_replica1_probe9_draw0_then1910_remaining",
        "retained": coverage.noise.contract()["retained"], "controls": coverage.noise.contract()["control"],
        "scopes": list(learning.SCOPES), "variants": list(learning.VARIANTS),
        "metrics": "paired_cost_wait_delta_mean_label_wait_harm_and_draw_wait_harm_rate_gate_regret_fixed_action_switch",
        "aggregation": "equal20_blocks_then_equal_cells_with_source_admitted_only_and_prefix_concentration_breakdowns",
        "budget": {"seal_tasks": 1, "collection_tasks": 1920, "report_tasks": 1, "new_native_episodes": 15360,
            "new_native_clock_ticks": 15360*61380, "new_prefix_scenes": 0, "new_raw_reference_fits": 0,
            "new_residual_fits": 0, "training_updates": 0},
        "storage": "sealed_actions_draw_labels_rollouts_and_models_server_only_small_receipts_and_aggregate_sync",
        "boundary": "conditional_future_replication_on_reused_development_prefixes_not_fresh_seed_CI_or_end_to_end_HRL",
        "confirmation": "Stage169_reserved_third_fourth_cohorts_remain_closed_default_unchanged"}


def load_seal(run_name):
    sealed = json.loads(sealed_path(run_name).read_text())
    if (sealed["protocol"] != EXPERIMENT_PROTOCOL or sealed["contract"] != contract()
            or not sealed["software_qualified"] or [c["key"] for c in sealed["cells"]] != [list(k) for k in KEYS]):
        raise ValueError("Complete the unchanged Stage241 action seal before collecting futures")
    return sealed


def seal(run_name):
    training, previous = prior.training, prior.previous
    bank, _, registered, _ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"], registered)
    panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
    results = []
    for key in KEYS:
        source = json.loads((raw_directory(previous.result_path(SOURCE_RUN, *key))/"result.json").read_text())
        previous.qualify(source, key, views)
        refit = json.loads((raw_directory(previous.result_path(REFIT_RUN, *key))/"result.json").read_text())
        previous.previous.previous.qualify(refit, key, views)
        refs = json.loads((raw_directory(previous.result_path(REFIT_RUN, *key))/"models.json").read_text())["references"]
        models = json.loads((raw_directory(previous.result_path(SOURCE_RUN, *key))/"models.json").read_text())
        train, _ = training.cohort.split_rows(views[key[0]], key[1])
        results.append({"key": list(key), **learning.seal_cell(train, panels, refit["cell"], refs, source["cell"], models)})
        print(f"FROZEN_FUTURE_ACTIONS_SEALED {key}", flush=True)
    states = {learning.state_key(r) for c in results for b in c["blocks"] for r in b["states"]}
    expected = {(r, coverage.noise.source.scene_seed(r, s, i), p) for r, s, i, p in coverage.states()}
    if states != expected or sum(c["reproduction"]["records"] for c in results) != 12800:
        raise ValueError("Sealed OOF actions lost existing states or added a new cohort")
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True, "cells": results}
    write_json(sealed_path(run_name), result)
    cached = [{"blocks": [{"held_group": b["held_group"], "scopes": {s: b[s] for s in learning.SCOPES[:2]}}
        for b in c["blocks"]]} for c in results if c["key"][0].startswith("mixed")]
    write_json(ROOT/"results"/run_name/"seal/result.json", {k: v for k, v in result.items() if k != "cells"} | {
        "sealed_cells": len(results), "states": len(states), "prefixes": len({k[:2] for k in states}),
        "selected_oof_records": 12800, "max_abs_difference": max(c["reproduction"]["max_abs_difference"] for c in results),
        "mixed_cached": learning.aggregate(cached, scopes=learning.SCOPES[:2]),
        "training_updates": 0})


def qualify_draw(cell):
    key = cell["root"], cell["scenario"], cell["replica"], cell["probe_index"], cell["draw"]
    if not (key in cells() and cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract()
            and cell["software_qualified"] and cell["scene_seed"] == coverage.noise.source.scene_seed(*key[:3])
            and cell["continuation_seed"] == continuation_seed(*key)
            and cell["original_source_reproductions"] == cell["exact_resampled_identities"] == 1
            and cell["prefix_state_rng_matches"] == 7 and cell["frozen_weight_qualified_episodes"] == 8
            and cell["window_ledger_qualified_episodes"] == 8 and cell["continuation_qualified"]
            and cell["native_clock_ticks"] == 8*61380 and cell["training_updates"] == 0):
        raise ValueError("Changed independent future, prefix pairing, weights or native budget")
    for labels in (cell["labels"], cell["existing_labels"]):
        if [l["candidate"] for l in labels] != list(range(6)):
            raise ValueError("Missing paired candidate labels")
        for label in labels:
            if not (np.isfinite([label[f] for f in coverage.expected.LABEL_FIELDS]).all()
                    and label["future_fleet_cost"] >= 0
                    and np.isclose(label["actual_cost_delta"], label["fleet_cost_delta"]+label["nonfleet_cost_delta"], atol=1e-12, rtol=0)
                    and np.isclose(label["future_fleet_cost"]-labels[cell["original_choice"]]["future_fleet_cost"], label["fleet_cost_delta"], atol=1e-12, rtol=0)
                    and np.isclose(label["physical_return_delta"], -100*label["actual_cost_delta"], atol=1e-12, rtol=0)):
                raise ValueError("Changed physical target units")
        if any(labels[cell["original_choice"]][f] != 0 for f in coverage.expected.LABEL_FIELDS if f != "future_fleet_cost"):
            raise ValueError("Same-tape original action is not exactly its baseline")


def collect(run_name, key):
    load_seal(run_name)
    root, scenario, replica, probe, draw = key
    bank = json.loads(coverage.source.bank_path(coverage.SOURCE_RUN).read_text())
    scene = coverage.noise.source.scene_seed(root, scenario, replica)
    row = next(r for r in bank["training_data"] if learning.state_key(r) == (root, scene, probe))
    models = json.loads((ROOT/"results"/coverage.noise.COLLECTION_RUN/"inputs/models.json").read_text())
    output = draw_path(run_name, *key)
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "replica": replica, "draw": draw,
        **coverage.noise.replay_draw(root, scenario, replica, probe, continuation_seed(*key), row, models[str(root)], output)}
    qualify_draw(result)
    write_json(raw_directory(output)/"result.json", result)
    write_json(output, {k: v for k, v in result.items() if k not in ("labels", "existing_labels")})
    print(f"FROZEN_FUTURE_DRAW_COMPLETE {key}", flush=True)


def report(run_name):
    sealed, panels = load_seal(run_name), {}
    for root, scenario, replica, probe in coverage.states():
        draws = []
        for draw in DRAWS:
            cell = json.loads((raw_directory(draw_path(run_name, root, scenario, replica, probe, draw))/"result.json").read_text())
            qualify_draw(cell)
            draws.append([[l[f] for f in ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")] for l in cell["labels"]])
        panels[root, coverage.noise.source.scene_seed(root, scenario, replica), probe] = np.asarray(draws)
    evaluated = [{"key": c["key"], **learning.evaluate_cell(c, panels)} for c in sealed["cells"]]
    mixed = [c for c in evaluated if c["key"][0].startswith("mixed")]
    admitted = {}
    for anchor in learning.joint.ANCHORS:
        for family in learning.previous.FAMILIES:
            for mode in learning.previous.FILTERS:
                xs = [c for c in mixed if c["selection"][anchor][family][mode]["candidate"] is not None]
                name = f"{family}_{anchor}_{mode}"
                admitted[name] = {"cells": len(xs), "scopes": {s: {"all": p["all"]["variants"][name],
                    "by_root": {r: v["variants"][name] for r, v in p["by_root"].items()}}
                    for s, p in learning.aggregate(xs).items()} if xs else {}}
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "views": {v: learning.aggregate([c for c in evaluated if c["key"][0] == v]) for v in VIEWS},
        "mixed_pooled": learning.aggregate(mixed), "mixed_admitted": admitted,
        "prefix_by_cell": [{"key": c["key"], "selection": c["selection"], "effects": learning.prefix_effects(c)} for c in evaluated],
        "regression_case": learning.aggregate([c for c in evaluated if tuple(c["key"]) == prior.REGRESSION_KEY]),
        "selected_oof_records": sum(c["reproduction"]["records"] for c in sealed["cells"]),
        "oof_max_abs_difference": max(c["reproduction"]["max_abs_difference"] for c in sealed["cells"]),
        **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}
    write_json(ROOT/"results"/run_name/"report/result.json", result)
    print("FROZEN_FUTURE_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("seal", "collect", "report"), required=True)
    p.add_argument("--seed", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--replica", type=int, choices=coverage.REPLICAS)
    p.add_argument("--probe", type=int, choices=coverage.PROBES)
    p.add_argument("--draw", type=int, choices=DRAWS)
    args = p.parse_args()
    if args.operation == "seal": seal(args.run_name)
    elif args.operation == "report": report(args.run_name)
    else: collect(args.run_name, (args.seed, args.scenario, args.replica, args.probe, args.draw))


if __name__ == "__main__": main()
