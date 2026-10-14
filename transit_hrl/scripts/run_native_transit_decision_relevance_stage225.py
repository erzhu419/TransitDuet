#!/usr/bin/env python3
"""Diagnose wait-loss relevance with frozen Stage224 controllers."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_refit_binding_stage224 as previous
from scripts import run_native_transit_best_action_crossfit_stage220 as cohort
from scripts import run_native_transit_precision_stage223 as precision_spec
from freq_hrl.domains.transit import native_decision_relevance as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_decision_relevance_stage225_v1"
SOURCE_RUN = "native_transit_refit_binding_stage224_development_20261014_r1"
VIEWS, PARTITIONS, KEYS = previous.VIEWS, previous.PARTITIONS, previous.KEYS
SCOPES = (*previous.SCOPES, "oof_same", "oof_other")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": previous.contract()["cohort"], "outer": previous.contract()["outer"],
        "partitions": previous.contract()["partitions"], "models": "exact_final4_and_c2v4_OOF_predictions_calibration_fixed_no_fitting",
        "strata": {"all": "five_nonidentity_candidates", "contender": "nonidentity_predicted_cost_lt_zero",
            "front": "identity_first_unfiltered_predicted_cost_winner_if_negative", "excluded": "nonidentity_predicted_cost_ge_zero"},
        "masks": "causal_cost_only_shared_by_all_wait_heads_and_future_labels",
        "gradient": "uniform_mass_loss_residual_alpha_derivative_at0_at1_and_frozen_uniform_risk_alpha",
        "future_noise": "same_vs_other_four_draw_mean_wait_sign_and_changed_choice_wait_harm",
        "oracle": "diagnostic_true_wait_menu_fixed_predicted_cost_ranking_never_inference",
        "aggregation": "sum_candidate_and_state_statistics_before_dividing_not_mean_of_subset_means",
        "scopes": list(SCOPES), "gates": list(learning.binding.GATES),
        "budget": {"analysis_tasks": 32, "report_tasks": 1, "reused_final4_cells": 32, "reused_oof_cells": 32,
            "wait_residual_updates": 0, "calibration_scalar_fits": 0, "new_native_episodes": 0, "rl_actor_updates": 0},
        "boundary": "development_diagnostics_reused240_states8_draws_no_independent_CI_no_default_replacement",
        "storage": "remote_compact_cell_totals_only_pull_aggregate_report_no_panels_models"}


def result_path(run_name, *key):
    return previous.result_path(run_name, *key)


def analyze_cell(train, held, panels, bundle, oof, calibration, partition, direction):
    a, b = PARTITIONS[partition][direction], PARTITIONS[partition][1-direction]
    data = {}
    for name, rows in (("train", train), ("held", held)):
        first = learning.binding.final_data(rows, panels, a, bundle)
        second = learning.truth(rows, panels, b, first)
        data[f"{name}_same"], data[f"{name}_other"] = first, second
    data["oof_same"] = oof
    data["oof_other"] = learning.binding.precision.common_validation(train, panels, b, oof)
    alternate = lambda s: s.removesuffix("same")+"other" if s.endswith("same") else s.removesuffix("other")+"same"
    subset = lambda d, root: {k: v[d["roots"] == root] for k, v in d.items()}
    return {"state_keys": {name: [list(learning.binding.learned.state_key(r)) for r in rows]
            for name, rows in (("train", train), ("held", held), ("oof", learning.binding.precision.validation_rows(train)))},
        "scopes": {s: learning.evaluate(data[s], data[alternate(s)], calibration) for s in SCOPES},
        "held_other_by_root": {str(root): learning.evaluate(subset(data["held_other"], root), subset(data["held_same"], root), calibration)
            for root in sorted(set(data["held_other"]["roots"]))}}


def qualify(result, key, views):
    train, held = cohort.split_rows(views[key[0]], key[1])
    cell = result["cell"]
    if result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract() or result["key"] != list(key):
        raise ValueError("Changed decision relevance protocol")
    for name, rows, repeat in (("train", train, 1), ("held", held, 1), ("oof", train, 2)):
        expected = Counter({learning.binding.learned.state_key(r): repeat for r in rows})
        if Counter(map(tuple, cell["state_keys"][name])) != expected: raise ValueError("Changed state/future coverage")
    if set(cell["scopes"]) != set(SCOPES): raise ValueError("Missing diagnostic scope")
    for scope, data in cell["scopes"].items():
        n = 120 if scope.startswith("oof") else 60
        strata = data["strata"]
        if (data["state_cases"] != n or set(strata) != set(learning.STRATA)
                or strata["all"]["candidate_cases"] != 5*n
                or strata["contender"]["candidate_cases"]+strata["excluded"]["candidate_cases"] != 5*n
                or strata["front"]["candidate_cases"] > n
                or set(data["decisions"]) != set(learning.binding.GATES)):
            raise ValueError("Changed causal candidate partition")
    expected_roots = {str(r["root"]): sum(x["root"] == r["root"] for x in held) for r in held}
    if {k: v["state_cases"] for k, v in cell["held_other_by_root"].items()} != expected_roots:
        raise ValueError("Missing held source root")


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 diagnostic cells")
    for key, result in results.items(): qualify(result, key, views)
    output = {}
    for view in VIEWS:
        cells = [r["cell"] for k, r in results.items() if k[0] == view]
        output[view] = {"scopes": {s: learning.summarized(learning.combine([c["scopes"][s] for c in cells])) for s in SCOPES},
            "held_other_by_root": {str(root): learning.summarized(learning.combine([c["held_other_by_root"][str(root)] for c in cells]))
                for root in sorted({r["root"] for r in views[view]})},
            "by_future_direction": {f"{p}/{d}": {k: v for k, v in learning.summarized(learning.combine(
                [r["cell"]["scopes"]["held_other"] for key, r in results.items() if key[0] == view and key[2:] == (p, d)])).items()
                if k in ("state_cases", "decisions", "switches")} for p in PARTITIONS for d in (0, 1)}}
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "views": output, **contract()["budget"], "native_deployment_eligible": False}


def reproduce_controls(result, prior):
    differences = []
    for view in VIEWS:
        for scope in previous.SCOPES:
            a, b = result["views"][view]["scopes"][scope], prior["views"][view]["scopes"][scope]
            for g in learning.binding.GATES:
                for m in ("mean_cost", "mean_wait_min", "changed_fraction", "unsafe_fraction"):
                    differences.append(abs(a["decisions"][g][m]-b["variants"][g][m]))
                differences.append(abs(a["strata"]["all"]["heads"][g]["physical_mass_loss_min"]-b["wait_heads"][g]["physical_mass_loss_min"]))
    if max(differences) > 1e-12: raise ValueError("Stage224 controls failed reproduction")
    return {"comparisons": len(differences), "max_abs_difference": max(differences)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("analyze", "report"), required=True)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--partition", choices=PARTITIONS)
    parser.add_argument("--direction", type=int, choices=(0, 1))
    args = parser.parse_args()
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if prior["protocol"] != previous.EXPERIMENT_PROTOCOL or prior["contract"] != previous.contract() or not prior["coverage_qualified"]:
        raise ValueError("Complete Stage224 before relevance diagnostics")
    bank, _, registered, _ = cohort.source.source.load_data()
    views = cohort.source.matched_views(bank["training_data"], registered)
    if args.operation == "analyze":
        key = args.view, args.fold, args.partition, args.direction
        source = raw_directory(previous.result_path(SOURCE_RUN, *key))
        previous.qualify(json.loads((source/"result.json").read_text()), key, views)
        models = json.loads((source/"models.json").read_text())
        oof_path = raw_directory(precision_spec.result_path(previous.SOURCE_RUN, *key))/"inner_oof.json"
        oof = {k: np.asarray(v) for k, v in json.loads(oof_path.read_text())["c2v4"].items()}
        train, held = cohort.split_rows(views[args.view], args.fold)
        panels = cohort.source.source.wait_source.source.diagnostic.load_panels()
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key),
            "cell": analyze_cell(train, held, panels, models["bundle"], oof, models["calibration"], args.partition, args.direction)}
        qualify(result, key, views)
        output = result_path(args.run_name, *key)
        write_json(raw_directory(output)/"result.json", result)
        write_json(output, {k: v for k, v in result.items() if k != "cell"})
        print("NATIVE_DECISION_RELEVANCE_ANALYSIS_COMPLETE", flush=True)
    else:
        result = summarize({k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS}, views)
        result["control_reproduction"] = reproduce_controls(result, prior)
        write_json(ROOT/"results"/args.run_name/"report/result.json", result)
        print("NATIVE_DECISION_RELEVANCE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
