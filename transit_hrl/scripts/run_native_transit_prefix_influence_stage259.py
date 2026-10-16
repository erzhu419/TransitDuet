#!/usr/bin/env python3
"""Apply saved inner16 models beyond the prefix that each fit excluded."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_training_count_stage258 as source
from freq_hrl.domains.transit import native_prefix_influence as diagnostic
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_prefix_influence_stage259_v1"
SOURCE_RUN = "native_transit_training_count_stage258_20261016_r1"
KEYS, VIEWS = source.KEYS, source.VIEWS
PRIMARY = ("free_old_expanded8_pooled_safe", "shrunk_expanded8_pooled_safe")
CATEGORIES = ("all", "removed", "removed_point_harm", "removed_no_strict_gain")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "contrast": "saved_prefix_excluded_inner16_minus_frozen_full16_under_same_old_keys_reference_basis_prior_scale_cap_and_opposite16_futures",
        "regions": list(diagnostic.REGIONS), "keys": "Stage252_old_keys_and_Stage255_admission_categories_frozen; no_selection",
        "diagnostics": "action_disagreement_and_feasible_cost_service_risk_prediction_RMS; signed_and_absolute_state_prefix_effects; first8_last8; largest_removed_prefix_influence",
        "controls": "all8_arms_excluded_actions_and_inner16_full_outcomes_reproduce_Stage258; full_train_and_held_actions_and_outcomes_reproduce_Stage255",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_inner16_fits": 640,
            "new_critic_fits": 0, "new_raw_reference_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "all32_cells_new_actions_written_server_only_before_any_evaluation_panel_read; pull_scalar_report_only",
        "boundary": "development_influence_diagnostic_not_seed_CI_or_performance_selection; retained_states_are_training_exposed; repeated_held_states_across_removed_prefixes_not_independent; removal_changes54_vs60_fit_and_calibration_population_together; function_change_not_prediction_accuracy_or_nonlinearity_proof; Stage253_248_169_closed; no_default_change"}


def read_cell(run, key, file):
    return json.loads((raw_directory(source.result_path(run, *key))/file).read_text())


def freeze_cell(run, key, views):
    before = read_cell(SOURCE_RUN, key, "aggregate.json")
    if before["protocol"] != source.EXPERIMENT_PROTOCOL or before["contract"] != source.contract() or not before["coverage_qualified"]:
        raise ValueError("Require completed Stage258 cell")
    models = read_cell(SOURCE_RUN, key, "models.json")
    excluded = read_cell(SOURCE_RUN, key, "actions.json")
    full_actions = read_cell(source.CACHE_RUN, key, "actions.json")
    reference = read_cell(source.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    full_models = read_cell(source.cached.SOURCE_RUN, key, "models.json")
    train, held = source.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train60": train, "held60": held}
    data = {s: source.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    for scope, rs in rows.items():
        states = [list(source.sampling.old_future.learning.state_key(r)) for r in rs]
        for a in (a for a in full_actions if a["scope"] == scope):
            if a["states"] != states or a["choices"]["raw"] != data[scope]["raw_choices"].tolist():
                raise ValueError("Changed frozen full state order or raw actions")
    frozen = []
    for bank in (0, 1):
        draws, _ = source.sampling.learning.draw_assignment(bank, "expanded")
        pieces = [p for p in before["pieces"] if p["bank"] == bank]
        full_paths = {scope: {m: source.learning.base.candidates(d, full_models[str(bank)][m]) for m in source.learning.MODELS}
            for scope, d in data.items()}
        for fitted in (m for m in models if m["bank"] == bank):
            removed = fitted["held_group"]
            cached = next(c for c in excluded if c["bank"] == bank and c["held_group"] == removed)
            if fitted["training_draws"] != draws or cached["training_draws"] != draws:
                raise ValueError("Changed inner16 training assignment")
            for scope, d in data.items():
                full = next(a for a in full_actions if a["bank"] == bank and a["scope"] == scope)
                masks = diagnostic.regions(full["states"], scope, removed)
                paths = {m: source.learning.base.candidates(d, fitted["models"][m]) for m in source.learning.MODELS}
                for p in pieces:
                    model = p["arm"].rsplit("_", 1)[0]
                    variant = f"{p['arm']}_{p['filter']}"
                    old = p["old_key"] or "zero"
                    inner, fp = paths[model][old], full_paths[scope][model][old]
                    fc = full["choices"][f"prefix_refit_{variant}"]
                    if fp["chosen"].tolist() != fc: raise ValueError("Frozen full model actions changed")
                    if scope == "train60":
                        ix = np.flatnonzero(masks["excluded"])
                        if ([full["states"][i] for i in ix] != cached["states"] or
                                inner["chosen"][ix].tolist() != cached["choices"][model][old]):
                            raise ValueError("Stage258 excluded actions changed")
                    frozen.append({"cell": list(key), "bank": bank, "removed": removed, "variant": variant,
                        "category": p["category"], "old_key": p["old_key"], "old_key_new_gate": p["old_key_new_gate"],
                        "scope": scope, "states": full["states"], "raw": full["choices"]["raw"],
                        "inner": inner["chosen"].tolist(), "full": fc,
                        "function_gaps": {r: diagnostic.function_gap(d, inner, fp, mask) for r, mask in masks.items() if mask.any()}})
    if len(models) != 20: raise ValueError("Incomplete saved inner16 prefix models")
    write_json(raw_directory(source.result_path(run, *key))/"actions.json", frozen)
    print(f"PREFIX_INFLUENCE_ACTIONS_FROZEN {key}", flush=True)


def evaluate_cell(run, key, panels):
    frozen = read_cell(run, key, "actions.json")
    before = read_cell(SOURCE_RUN, key, "aggregate.json")
    anchor = read_cell(source.CACHE_RUN, key, "aggregate.json")
    records, maximum = {}, 0.
    for c in frozen:
        _, ev = source.sampling.learning.draw_assignment(c["bank"], "expanded")
        values = np.stack([panels[tuple(st)]["values"][ev] for st in c["states"]])
        ident = (c["bank"], tuple(c["removed"]), c["variant"])
        record = records.setdefault(ident, {f: c[f] for f in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate")})
        record.setdefault("regions", {})
        for region, mask in diagnostic.regions(c["states"], c["scope"], c["removed"]).items():
            if not mask.any(): continue
            a, b, raw = (np.asarray(c[f])[mask] for f in ("inner", "full", "raw"))
            v = values[mask]
            record["regions"][region] = {"whole": diagnostic.outcomes(v, a, b, raw),
                "first8": diagnostic.outcomes(v[:, :8], a, b, raw), "last8": diagnostic.outcomes(v[:, 8:], a, b, raw),
                "function_gap": c["function_gaps"][region]}
        full = diagnostic.outcomes(values, c["inner"], c["full"], c["raw"])["full"]
        bank = next(b for b in anchor["banks"] if b["bank"] == c["bank"])
        expected = bank["matrix"][c["scope"]]["all"]["variants"][f"prefix_refit_{c['variant']}"]
        maximum = max(maximum, *(abs(full[f]-expected[f]) for f in diagnostic.FIELDS))
    pieces = list(records.values())
    for old in before["pieces"]:
        variant = f"{old['arm']}_{old['filter']}"
        xs = [p for p in pieces if p["bank"] == old["bank"] and p["variant"] == variant]
        if len(xs) != 10 or sum(x["regions"]["excluded"]["whole"]["cases"] for x in xs) != 60:
            raise ValueError("Incomplete excluded prefix coverage")
        summary = diagnostic.summarize(xs)["regions"]["excluded"]
        for label, prior in (("inner", "inner16"), ("full", "full_seen")):
            for f in diagnostic.FIELDS[:4]: maximum = max(maximum, abs(summary[label][f]-old["paths"][prior]["all"][f]))
        maximum = max(maximum, abs(summary["action_disagreement_rate"]-old["action_disagreement"]["inner16_full"]))
        for h in (0, 1):
            for label, prior in (("inner", "inner16"), ("full", "full_seen")):
                for f in diagnostic.FIELDS[:4]:
                    maximum = max(maximum, abs(summary["opposite_halves"][h][label][f]-old["evaluation_halves"][h][prior]["all"][f]))
    if maximum > 1e-12: raise ValueError("Prefix influence controls failed Stage258/255 reproduction")
    return {"key": list(key), "pieces": pieces, "reproduction_max_abs_difference": maximum}


def in_category(p, category):
    if category == "all" or p["category"] == category: return True
    return p["category"] == "removed" and category == ("removed_no_strict_gain" if p["old_key_new_gate"]["strict_gain_only"] else "removed_point_harm")


def report(records):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output = {}
    for group, predicate in groups.items():
        ps = [p for r in records if predicate(r["key"]) for p in r["pieces"]]
        output[group] = {label: {variant: {cat: diagnostic.summarize([p for p in ps
            if p["variant"] == variant and (bank is None or p["bank"] == bank) and in_category(p, cat)])
            for cat in CATEGORIES} for variant in PRIMARY} for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1))}
    concentration = {}
    for variant in PRIMARY:
        ps = [p for r in records if r["key"][0].startswith("mixed") for p in r["pieces"]
            if p["variant"] == variant and p["category"] == "removed"]
        totals = {}
        for p in ps:
            ident = tuple(p["removed"])
            row = totals.setdefault(ident, {"removed": list(ident), "model_exposures": 0,
                **{r: {"cases": 0, "disagreements": 0, "absolute_wait_sum": 0.} for r in ("excluded", "retained", "held")}})
            row["model_exposures"] += 1
            for region in ("excluded", "retained", "held"):
                s, target = p["regions"][region]["whole"], row[region]
                target["cases"] += s["cases"]; target["disagreements"] += s["disagreements"]
                target["absolute_wait_sum"] += s["cases"]*s["absolute_state_effect"]["delta_wait_min"]
        concentration[variant] = sorted(totals.values(), key=lambda x: -sum(x[r]["absolute_wait_sum"] for r in ("excluded", "retained", "held")))
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "groups": output, "mixed_removed_prefix_influence": concentration,
        "reproduction": {"cached_cells": len(records), "cached_inner16_fits": 640,
            "max_abs_difference": max(r["reproduction_max_abs_difference"] for r in records)},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    data, _, registered, _ = source.cached.training.cohort.source.source.load_data()
    views = source.cached.training.cohort.source.matched_views(data["training_data"], registered)
    for key in KEYS: freeze_cell(args.run_name, key, views)
    panels = source.sampling.read_panels(source.cached.previous.DATA_RUN)
    records = []
    for key in KEYS:
        records.append(evaluate_cell(args.run_name, key, panels))
        print(f"PREFIX_INFLUENCE_EVALUATED {key}", flush=True)
    path = ROOT/"results"/args.run_name/"report/result.json"
    write_json(raw_directory(path)/"cells.json", records)
    write_json(path, report(records))
    print("PREFIX_INFLUENCE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
