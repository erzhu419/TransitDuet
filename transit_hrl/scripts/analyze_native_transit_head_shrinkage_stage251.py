#!/usr/bin/env python3
"""Separate model and selected-key changes without fitting or new simulation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_head_shrinkage_stage251 as spec
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json


def outcome(data, draws, prediction):
    mean, ix = draws.mean(axis=1), np.arange(len(draws))
    delta = mean[ix, prediction["chosen"]]-mean[ix, data["raw_choices"]]
    return {**{f: float(delta[:, c].mean()) for f, c in
        (("delta_cost", 0), ("delta_risk", 1), ("delta_service", 2), ("delta_wait_min", 3))},
        "changed_rate": float((prediction["chosen"] != data["raw_choices"]).mean())}


def selection_swap(paths, selection):
    return {f"{model}_on_{selector}": paths[model][selection[selector]["pooled_safe"] or "zero"]
        for model in ("free_old", "shrunk") for selector in ("free_old", "shrunk")}


def decompose(swaps):
    fields = ("delta_cost", "delta_wait_min", "delta_risk", "delta_service")
    model = {f: swaps["shrunk_on_free_old"][f]-swaps["free_old_on_free_old"][f] for f in fields}
    selector = {f: swaps["shrunk_on_shrunk"][f]-swaps["shrunk_on_free_old"][f] for f in fields}
    total = {f: swaps["shrunk_on_shrunk"][f]-swaps["free_old_on_free_old"][f] for f in fields}
    return {"model_at_free_selected_key": model, "key_change_on_shrunk_model": selector, "total": total}


def analyze(run):
    report = json.loads((ROOT/"results"/run/"report/result.json").read_text())
    if report["contract"] != spec.contract() or not report["coverage_qualified"]: raise ValueError("Require qualified Stage251 report")
    panels = spec.information.read_panels(spec.DATA_RUN)
    bank, _, registered, _ = spec.training.cohort.source.source.load_data()
    views = spec.training.cohort.source.matched_views(bank["training_data"], registered)
    records = []
    for key in spec.KEYS:
        path = raw_directory(spec.result_path(run, *key))
        r, models = json.loads((path/"aggregate.json").read_text()), json.loads((path/"models.json").read_text())
        ref = raw_directory(spec.result_path(spec.fixed.contract()["reference_run"], *key))/"models.json"
        reference = json.loads(ref.read_text())["reference"]
        train, held = spec.training.cohort.split_rows(views[key[0]], key[1])
        data = {s: spec.learning.base.causal_data(rows, reference) for s, rows in (("train60", train), ("held60", held))}
        for b in r["banks"]:
            _, evaluation = spec.information.learning.draw_assignment(b["bank"], "expanded")
            for scope, rows in (("train60", train), ("held60", held)):
                d, draws = data[scope], spec.learning.base.training_draws(rows, panels, evaluation)
                paths = {a: spec.learning.base.candidates(d, models[str(b["bank"])][a]) for a in spec.learning.ARMS}
                records.append({"key": key, "bank": b["bank"], "scope": scope,
                    "swaps": {name: outcome(d, draws, p) for name, p in selection_swap(paths, b["selection"]).items()},
                    "fixed_candidates": {a: {c: outcome(d, draws, p) for c, p in ps.items()} for a, ps in paths.items()}})
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in spec.VIEWS}}
    output, maximum = {}, 0.
    for name, predicate in groups.items():
        output[name] = {}
        for scope in ("train60", "held60"):
            output[name][scope] = {}
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                rs = [r for r in records if predicate(r["key"]) and r["scope"] == scope and (bank is None or r["bank"] == bank)]
                swaps = spec.average([r["swaps"] for r in rs])
                for arm in ("free_old", "shrunk"):
                    expected = report["groups"][name][scope][label]["outcomes"][f"{arm}_pooled_safe"]
                    maximum = max(maximum, *(abs(value-expected[f]) for f, value in swaps[f"{arm}_on_{arm}"].items()))
                output[name][scope][label] = {"swaps": swaps, "decomposition": decompose(swaps),
                    "fixed_candidates": spec.average([r["fixed_candidates"] for r in rs])}
    if maximum > 1e-12: raise ValueError("Cached selection-swap baseline differs from report")
    write_json(ROOT/"results"/run/"analysis/result.json", {"protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "groups": output, "baseline_max_abs_difference": maximum,
        "interpretation": "post_hoc diagnostic all17 registered candidates; swap keys only, never select a winner or change any model; not CI",
        "new_native_episodes": 0, "new_raw_reference_fits": 0, "critic_fits": 0})
    print("HEAD_SHRINKAGE_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    analyze(p.parse_args().run_name)
