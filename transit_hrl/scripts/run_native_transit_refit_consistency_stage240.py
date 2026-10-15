#!/usr/bin/env python3
"""Separate fixed-candidate OOF instability from residual/reference refits."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_objective_oof_stage239 as previous
from freq_hrl.domains.transit import native_refit_consistency as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_refit_consistency_stage240_v1"
SOURCE_RUN = "native_transit_objective_oof_stage239_development_20261015_r1"
REFIT_RUN = previous.REFIT_RUN
VIEWS, PARTITIONS, KEYS = previous.VIEWS, previous.PARTITIONS, previous.KEYS
training, joint = previous.training, learning.joint
result_path = previous.result_path
REGRESSION_KEY = ("mixed_b", 1, "even_odd", 1)


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL, "refit_run": REFIT_RUN,
        "reference_run": parent["reference_run"], "cohort": parent["cohort"], "outer": parent["outer"], "partitions": parent["partitions"],
        "endpoints": list(learning.ENDPOINTS), "variants": list(learning.VARIANTS), "scopes": list(learning.SCOPES),
        "selection": "all20_Stage239_objective_anchor_filter_candidates_frozen_no_reselection_or_admission",
        "paired_design": "20_prefix_future_blocks_per_cell_same_probe_and_common_full_raw_baseline_at_all5_endpoints",
        "state_expansion": "54/2_cached_to60/2_residual_refit_under_same54/2_reference_then60/2_cached_native_reference",
        "draw_expansion": "60/2_native_to60/4_residual_refit_under_same60/2_reference_then60/4_cached_native_reference",
        "residual": "same_four_objectives_rank_strength_keys_198features_TSVD_rank_RMS_rule_each_fit_uses_own_training_scales_basis",
        "reference": "all_raw_references_from_Stage237_and_Stage226_zero_new_raw_training",
        "probe_label_exposure": {"54_2_cached": "prefix_unseen_future_unseen", "60_2_fixed54ref": "residual_prefix_seen_future_unseen",
            "60_2_native": "prefix_seen_future_unseen", "60_4_fixed60ref": "residual_prefix_and_future_seen", "60_4_native": "prefix_and_future_seen"},
        "prefix_diagnostic": "delete_one_validation_prefix_from_existing_curves_frozen_candidate_fixed_scales_no_refit_or_new_filter",
        "metrics": "physical_cost_wait_risk_vs_own_and_common_raw_gate_regret_action_switch_vs54/2_and_preceding_endpoint",
        "averaging": "equal20_matched_blocks_then_equal_cells_reused60/4_endpoint_not20_independent_replicates",
        "controls": "selected54/2_probe_outcomes_counts_and60/4_full_metrics_reproduce_Stage239",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "matched_blocks": 640, "new_raw_reference_fits": 0,
            "raw_wait_updates": 0, "new_all_MSE_paths": 1408, "residual_SVDs": 4480, "objective_solves": 18432,
            "new_native_episodes": 0, "rl_actor_updates": 0},
        "boundary": parent["boundary"], "storage": "remote_model_and_block_caches_pull_aggregate_only",
        "limitation": "paired_refit_diagnosis_includes_basis_scale_RMS_changes_reused_development_data_not_CI_or_clean_probe_performance_comparison"}


def qualify(result, key, views):
    if result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract() or result["key"] != list(key):
        raise ValueError("Changed refit consistency protocol")
    c = result["cell"]
    train, _ = training.cohort.split_rows(views[key[0]], key[1])
    groups = {(r["root"], r["scene_seed"]) for r in train}
    roots = {str(r["root"]) for r in train}
    a = PARTITIONS[key[2]][key[3]]
    if (len(c["blocks"]) != 20 or {(tuple(b["held_group"]), b["half"]) for b in c["blocks"]} != {(g, h) for g in groups for h in (0, 1)}
            or c["training_draws"] != list(a) or c["evaluation_draws"] != list(PARTITIONS[key[2]][1-key[3]])):
        raise ValueError("Dropped consistency prefix/future block")
    fits = []
    for block in c["blocks"]:
        fits.append(block["expanded_fits"])
        if set(block["endpoints"]) != set(learning.ENDPOINTS): raise ValueError("Dropped refit endpoint")
        for scopes in block["endpoints"].values():
            if set(scopes) != set(learning.SCOPES): raise ValueError("Dropped refit scope")
            for scope, stats in scopes.items():
                n = 6 if scope == "prefix_probe" else 60
                expected_roots = {str(block["held_group"][0])} if n == 6 else roots
                if (stats["all"]["cases"] != n or set(stats["by_root"]) != expected_roots
                        or sum(p["cases"] for p in stats["by_root"].values()) != n): raise ValueError("Dropped refit case/source")
                if any(set(p["variants"]) != set(learning.VARIANTS) for p in (stats["all"], *stats["by_root"].values())):
                    raise ValueError("Dropped frozen refit candidate")
    if set(c["half_fits"]) != {"0", "1"}: raise ValueError("Dropped native half refit")
    for record in c["half_fits"].values():
        if set(record) != set(learning.ENDPOINTS[2:4]): raise ValueError("Dropped half refit intervention")
        fits.extend(record.values())
    expected = {(o, l) for o in learning.previous.OBJECTIVES[1:] for l in learning.strength.lowrank.LEVELS[1:]}
    for record in fits:
        if set(record) != set(joint.ANCHORS): raise ValueError("Dropped refit anchor")
        for fs in record.values():
            if len(fs) != 12 or {(f["objective"], f["level"]) for f in fs} != expected: raise ValueError("Dropped objective refit")
            for f in fs:
                if (not f["success"] or f["parameter_count"] != f["retained_rank"] or f["optimality"] > 1e-8
                        or f["norm"] > f["radius"]+1e-12 or f["loss_after"] > f["loss_before"]+1e-12*max(1., f["loss_before"])):
                    raise ValueError("Refit capacity or optimizer changed")
    if set(c["prefix_diagnostics"]) != set(learning.VARIANTS[1:]): raise ValueError("Dropped frozen prefix diagnosis")
    for anchor in joint.ANCHORS:
        for family in learning.previous.FAMILIES:
            for mode in learning.previous.FILTERS:
                d = c["prefix_diagnostics"][f"{family}_{anchor}_{mode}"]
                admitted = c["selection"][anchor][family][mode]["candidate"] is not None
                if d["admitted"] != admitted or (admitted and {tuple(x["removed_group"]) for x in d["deletions"]} != groups):
                    raise ValueError("Changed frozen prefix deletion checks")
    r = c["reproduction"]
    if (r["probe_records"] != 400 or r["full_metric_records"] != 84*(len(roots)+1)
            or max(r[f] for f in ("probe_max_abs_difference", "full_max_abs_difference", "subspace_max_abs_difference")) > 1e-12):
        raise ValueError("Cached refit endpoints changed")


def aggregate(cells):
    blocks = [b for c in cells for b in c["blocks"]]
    return {e: {s: {"all": learning.averaged([b["endpoints"][e][s]["all"] for b in blocks]),
        "by_root": {r: learning.averaged([b["endpoints"][e][s]["by_root"][r] for b in blocks if r in b["endpoints"][e][s]["by_root"]])
            for r in sorted({r for b in blocks for r in b["endpoints"][e][s]["by_root"]})}} for s in learning.SCOPES} for e in learning.ENDPOINTS}


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 consistency cells")
    for k, r in results.items(): qualify(r, k, views)
    cells = [results[k]["cell"] for k in KEYS]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "views": {v: aggregate([results[k]["cell"] for k in KEYS if k[0] == v]) for v in VIEWS},
        "mixed_pooled": aggregate([results[k]["cell"] for k in KEYS if k[0].startswith("mixed")]),
        "regression_case": {"key": list(REGRESSION_KEY), "matrix": aggregate([results[REGRESSION_KEY]["cell"]])},
        "prefix_by_cell": [{"key": list(k), "selection": results[k]["cell"]["selection"], "diagnostics": {
            name: {"admitted": d["admitted"], "deletion_checks": len(d["deletions"]), "eligibility_losses": sum(not x["eligible"] for x in d["deletions"]),
                "concentration": d["concentration"]} for name, d in results[k]["cell"]["prefix_diagnostics"].items()}} for k in KEYS],
        "reproduction": {**{f: sum(c["reproduction"][f] for c in cells) for f in ("probe_records", "full_metric_records")},
            **{f: max(c["reproduction"][f] for c in cells) for f in ("probe_max_abs_difference", "full_max_abs_difference", "subspace_max_abs_difference")}},
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
    bank, _, registered, _ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"], registered)
    if args.operation == "fit":
        key = args.view, args.fold, args.partition, args.direction
        source = json.loads((raw_directory(result_path(SOURCE_RUN, *key))/"result.json").read_text())
        previous.qualify(source, key, views)
        refit = json.loads((raw_directory(result_path(REFIT_RUN, *key))/"result.json").read_text())
        previous.previous.previous.qualify(refit, key, views)
        refits = json.loads((raw_directory(result_path(REFIT_RUN, *key))/"models.json").read_text())
        cached = json.loads((raw_directory(result_path(SOURCE_RUN, *key))/"models.json").read_text())
        reference = json.loads((raw_directory(result_path(contract()["reference_run"], *key))/"models.json").read_text())["reference"]
        train, held = training.cohort.split_rows(views[args.view], args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell, models = learning.fit_cell(train, held, panels, reference, refits, refit["cell"], source["cell"], cached)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "cell": cell}
        qualify(result, key, views)
        path = result_path(args.run_name, *key)
        write_json(raw_directory(path)/"models.json", models)
        write_json(raw_directory(path)/"result.json", result)
        write_json(path, {k: v for k, v in result.items() if k != "cell"})
        print("NATIVE_REFIT_CONSISTENCY_FIT_COMPLETE", flush=True)
    else:
        report = summarize({k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS}, views)
        write_json(ROOT/"results"/args.run_name/"report/result.json", report)
        print("NATIVE_REFIT_CONSISTENCY_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
