#!/usr/bin/env python3
"""Reduce server-side Stage249 fits to matched error and outcome summaries."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_training_information_stage249 as spec
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json


def average(rows):
    if isinstance(rows[0], dict):
        return {k: average([r[k] for r in rows]) for k in rows[0]}
    return float(np.mean(rows))


def summarize_diagnostics(records, scope, bank=None):
    blocks = [b for r in records for b in r["banks"] if bank is None or b["bank"] == bank]
    diagnostics = [b["diagnostics"][scope] for b in blocks]
    result = {"model_bank_exposures": len(blocks),
        "evaluation_mc_noise": average([d["matched"]["evaluation_mc_noise"] for d in diagnostics]),
        "zero_prediction_error": average([d["matched"]["zero_prediction_error"] for d in diagnostics]),
        "target_mean_error": {arm: average([d[arm]["target_mean_error"] for d in diagnostics])
            for arm in ("matched", "expanded")},
        "models": {name: {"evaluation": average([d["matched"]["predictor_error"][name] for d in diagnostics])}
            for name in ("raw", "old_pooled_safe", "matched_pooled_safe", "expanded_pooled_safe")},
        "ranks": {arm: average([d[arm]["unselected_rank_error"] for d in diagnostics]) for arm in ("matched", "expanded")},
        "admitted": {arm: sum(b["selection"][arm]["pooled_safe"] is not None for b in blocks) for arm in spec.learning.ARMS}}
    for arm in ("matched", "expanded"):
        result["models"][f"{arm}_pooled_safe"]["training_bank"] = average(
            [d[arm]["predictor_error_on_training_bank"][f"{arm}_pooled_safe"] for d in diagnostics])
    return result


def verify_frozen_actions(run):
    compared = 0
    aliases = {"raw": "raw", "old_pooled_safe": "state_gls_pooled_safe", "old_half_safe": "state_gls_half_safe"}
    for key in spec.KEYS:
        def load(source):
            return json.loads((raw_directory(spec.result_path(source, *key))/"actions.json").read_text())
        previous = {b["scope"]: b for b in load(spec.SOURCE_RUN) if b["scope"] in ("train60", "held60")}
        current = load(run)
        if {(b["bank"], b["scope"]) for b in current} != {(i, s) for i in (0, 1) for s in previous}:
            raise ValueError("Dropped frozen bank or prefix scope")
        for b in current:
            old = previous[b["scope"]]
            if old["states"] != b["states"]: raise ValueError("Changed frozen full-model state order")
            for name, source_name in aliases.items():
                if old["choices"][source_name] != b["choices"][name]: raise ValueError("Frozen reference actions changed")
                compared += len(b["states"])
    return {"action_exposures_compared": compared, "mismatches": 0}


def analyze(run):
    report = json.loads((spec.ROOT/"results"/run/"report/result.json").read_text())
    if report["protocol"] != spec.EXPERIMENT_PROTOCOL or report["contract"] != spec.contract() or not report["coverage_qualified"]:
        raise ValueError("Require qualified Stage249 report")
    fits = [json.loads((raw_directory(spec.result_path(run, *k))/"aggregate.json").read_text()) for k in spec.KEYS]
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in spec.VIEWS}}
    output = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
        "frozen_reproduction": verify_frozen_actions(run), "groups": {},
        "interpretation": {"train60": "seen_prefixes_evaluated_on_opposite16_futures",
            "held60": "held_prefixes_evaluated_on_opposite16_futures",
            "training_bank": "fit_exposed_only_for_own_arm_on_train60_not_for_held60",
            "pooled_banks": "descriptive_crossfit_mean_not_independent_seed_CI",
            "rank_curves": "all_registered_ranks_alpha1_diagnostic_only_no_reselection",
            "sign_error": "zero_is_a_class_including_zero_mean_fleet_ties"}}
    fields = ("delta_cost", "delta_wait_min", "delta_risk", "delta_service", "changed_rate")
    variants = ("raw", "old_pooled_safe", "matched_pooled_safe", "expanded_pooled_safe")
    for group, predicate in groups.items():
        records = [r for r in report["information_error"] if predicate(r["key"])]
        selected_fits = [r for r in fits if predicate(r["key"])]
        output["groups"][group] = {}
        for scope in ("train60", "held60"):
            output["groups"][group][scope] = {}
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
                summary = summarize_diagnostics(records, scope, bank)
                matrix = spec.learning.aggregate([b["matrix"][scope] for r in selected_fits for b in r["banks"]
                    if bank is None or b["bank"] == bank])
                summary["outcomes"] = {v: {f: matrix["all"]["variants"][v][f] for f in fields} for v in variants}
                output["groups"][group][scope][label] = summary
    write_json(spec.ROOT/"results"/run/"analysis/result.json", output)
    print("TRAINING_INFORMATION_ANALYSIS_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    analyze(parser.parse_args().run_name)


if __name__ == "__main__": main()
