#!/usr/bin/env python3
"""Localize Stage274 alignment gaps without fitting, inference or simulation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_risk_alignment_stage274 as source
from freq_hrl.domains.transit import native_alignment_localization as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_alignment_localization_stage275_v1"
SOURCE_RUN = "native_transit_risk_alignment_stage274_20261016_r1"
ARM, PENALTY = "raw_bounded_clip", .01
KEYS, VIEWS, cache = source.KEYS, source.VIEWS, source.cache


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arm": ARM, "penalty": PENALTY,
        "attribution": "same_Stage274_natural_and_common_action_paired_weights; per_model_probe_component_sums; each_bin_contribution_uses_its_population_global_denominator_not_average_of_bin_means; reproduce_all_Stage274_primary_components",
        "bins": {k: list(v) for k, v in learning.DIMENSIONS.items()},
        "stability": "compare_same_and_opposite16_for_the_same_frozen_head_and_probe; positive_both_negative_both_sign_flip_zero_either; repeated_contexts_not_independent_seeds",
        "concentration": "rank_prefix_alignment_contributions_on_one_future_bank; inspect_top1_3_5_on_other_bank_both_directions; report_signed_net_share_absolute_mass_share_and_renormalized_remaining_evaluation_gap; diagnostic_ranking_not_policy_selection_or_training_data_removal",
        "frozen": "all_predictions_priors_weights_actions_and_selectors_cached; no_new_fit_solve_inference_simulation_gate_or_default_change",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_prefix_contexts": 640, "model_probe_rows": 3840,
            "new_fits": 0, "new_model_inferences": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "per_model_probe_ledger_and_full_bank_breakdowns_server_only; pull_compact_scalar_report_with_pooled_breakdowns",
        "boundary": "development_cohort_localization_not_causal_explanation_seed_CI_or_policy_gain; common_action_coverage_and_historical_raw_reference_conditioning_unchanged; Stage253_248_169_closed"}


def analyze_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    load = lambda run, name: json.loads((here(run)/(name+".json")).read_text())
    models, queries = load(source.PREDICTION_RUN, "models"), load(source.PREDICTION_RUN, "queries")
    previous = {(r["index"], r["future"]): r for r in load(SOURCE_RUN, "errors")}
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    held_queries = load(source.SOURCE_RUN, "queries")
    if held_queries["held_states"] != [list(cache.sampling.old_future.learning.state_key(r)) for r in held]: raise ValueError("Changed held query order")
    qs = {(q["index"], q["scope"]): q for q in queries}
    chosen = [i for i, m in enumerate(models) if m["removed"] is not None and m["penalty"] == PENALTY]
    if len(chosen) != 20: raise ValueError("Incomplete primary contexts")
    with np.load(here(source.PREDICTION_RUN)/"predictions.npz") as p:
        excluded = {(i, a): p[f"{i}_excluded_{a}"].copy() for i in chosen for a in (ARM, "raw_prior")}
    with np.load(here(source.SOURCE_RUN)/"held_predictions.npz") as p: predictions = {a: p[a].copy() for a in (ARM, "raw_prior")}
    targets = {}
    for bank in (0, 1):
        ds, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
        for label, rows in (("train", train), ("held", held)):
            values = cache.learning.base.training_draws(rows, panels, ds)[..., 1]
            originals = np.asarray([r["original_choice"] for r in rows])
            targets[label, bank] = (values-values[np.arange(len(rows)), :, originals][:, :, None]).mean(axis=1)
    rows, identity_error, replay_error = [], 0., 0.
    for i in chosen:
        m = models[i];bank = m["bank"];q = qs[i, "excluded"];ex = np.asarray(q["ids"])
        ex_rows = [train[j] for j in ex];condition = ex_rows[0]["root"], ex_rows[0]["scenario"]
        if q["label"] != "train" or any((r["root"], r["scene_seed"]) != tuple(m["removed"]) for r in ex_rows): raise ValueError("Changed removed-prefix identity")
        full_i = next(j for j, v in enumerate(models) if v["removed"] is None and v["bank"] == bank and v["penalty"] == PENALTY)
        he = np.asarray([j for j, r in enumerate(held) if (r["root"], r["scenario"]) == condition])
        he_rows = [held[j] for j in he]
        eq, hq = np.asarray(q["eligible"], bool), np.asarray(qs[full_i, "held"]["eligible"], bool)[he]
        eys = {f: targets["train", b][ex] for f, b in (("same", bank), ("opposite", 1-bank))}
        hys = {f: targets["held", b][he] for f, b in (("same", bank), ("opposite", 1-bank))}
        probes, error = learning.probe_records(ex_rows, he_rows, excluded[i, ARM], predictions[ARM][i, he],
            excluded[i, "raw_prior"], predictions["raw_prior"][i, he], eq, hq, eys, hys)
        identity_error = max(identity_error, error)
        for f in learning.FUTURES:
            for pop in source.POPULATIONS:
                old = previous[i, f]["metrics"][ARM][pop]
                for term in ("units", *source.learning.FIELDS): replay_error = max(replay_error, abs(sum(r["metrics"][f][pop][term] for r in probes)-old[term]))
        rows.extend({"index": i, "bank": bank, "root": condition[0], "scene_seed": m["removed"][1], "scenario": condition[1], **r} for r in probes)
    if len(rows) != 120 or replay_error > 1e-10: raise ValueError("Stage274 per-context component replay failed")
    directory.mkdir(parents=True, exist_ok=True);write_json(directory/"probes.json", rows)
    return {"key": list(key), "rows": rows, "identity_max_abs_error": identity_error, "metric_max_abs_error": replay_error}


def report(cells, before):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, replay_error, partition_error = {}, 0., 0.
    for group, predicate in groups.items():
        rows = [r for c in cells if predicate(c["key"]) for r in c["rows"]];output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            selected = [r for r in rows if bank is None or r["bank"] == bank]
            s = learning.summarize(selected);output[group][label] = s;partition_error = max(partition_error, s["partition_max_abs_error"])
            for f in learning.FUTURES:
                old = before["groups"][group]["banks"][label][str(PENALTY)][f][ARM]
                for pop in source.POPULATIONS:
                    for field in ("units", *source.learning.FIELDS): replay_error = max(replay_error, abs(s["totals"][f][pop][field]-old[pop][field]))
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(len(c["rows"]) for c in cells) != 3840 or replay_error > 1e-10: raise ValueError("Incomplete Stage275 primary replay")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"cell_metric_max_abs_error": max(c["metric_max_abs_error"] for c in cells), "group_metric_max_abs_error": replay_error,
            "pointwise_identity_max_abs_error": max(c["identity_max_abs_error"] for c in cells), "partition_max_abs_error": partition_error},
        "default_changed": False, "native_deployment_eligible": False}


def compact_report(payload):
    return {**payload, "groups": {g: {b: {k: v for k, v in s.items() if b == "pooled" or k != "breakdown"}
        for b, s in banks.items()} for g, banks in payload["groups"].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(analyze_cell(directory/("_".join(map(str, key))), key, views, panels));print(f"ALIGNMENT_LOCALIZATION_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text());payload = report(cells, before)
    write_json(directory/"aggregate.json", payload);write_json(path, compact_report(payload));print("ALIGNMENT_LOCALIZATION_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
