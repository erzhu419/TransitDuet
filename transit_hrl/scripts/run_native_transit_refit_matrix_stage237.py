#!/usr/bin/env python3
"""Separate sample/draw refits, raw-reference changes and residual coordinates."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_stability_admission_stage236 as previous
from freq_hrl.domains.transit import native_refit_matrix as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

joint, training = learning.joint, previous.training
strength_spec = previous.previous.previous
stability_spec = previous.previous
EXPERIMENT_PROTOCOL = "native_transit_refit_matrix_stage237_v1"
SOURCE_RUN = "native_transit_stability_admission_stage236_development_20261015_r1"
OUTER_RUN, STABILITY_RUN = previous.OUTER_RUN, previous.SOURCE_RUN
MODEL_RUN, REFERENCE_RUN = strength_spec.MODEL_RUN, strength_spec.REFERENCE_RUN
VIEWS, PARTITIONS, KEYS = previous.VIEWS, previous.PARTITIONS, previous.KEYS
CONFIGS = ("54/2", "54/4", "60/2", "60/4")
REGRESSION_KEY = ("mixed_b", 1, "even_odd", 1)


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "outer_run": OUTER_RUN, "stability_run": STABILITY_RUN, "model_run": MODEL_RUN, "reference_run": REFERENCE_RUN,
        "cohort": parent["cohort"], "outer": parent["outer"], "partitions": parent["partitions"],
        "candidates": parent["candidates"], "selection": "all6_Stage234_candidates_frozen_no_new_selection_or_admission",
        "configs": list(CONFIGS), "arms": list(learning.ARMS), "variants": list(learning.VARIANTS), "scopes": list(learning.SCOPES),
        "fixed": "cached60_state4_draw_raw_reference_and_residual_coordinates",
        "changing_fixed_x": "refit_raw_reference_on_same_residual_training_subset_keep_fixed_residual_coordinates",
        "changing_native_x": "same_refitted_raw_reference_with_its_own_residual_coordinates",
        "design": "33_unique_fits_one60/4_two60/2_ten54/4_twenty54/2_mapped_to20_prefix_half_matched_blocks",
        "metrics": "physical_cost_MSE_rebased_to_original_action_own_gate_cost_regret_common_and_own_raw_cost_wait_risk_deltas",
        "averaging": "equal_matched_blocks_within_cell_then_equal_cells_each_root_separately_reused_full_endpoint_not_new_samples",
        "controls": "cached60/4_actions_scores_clean54/2_raw_OOF_and_all_selected_prefix_contributions_reproduce",
        "regression_key": list(REGRESSION_KEY),
        "budget": {"fit_tasks": 32, "report_tasks": 1, "unique_designs": 1056, "matched_blocks": 640,
            "new_raw_reference_fits": 1024, "raw_wait_updates": 614400, "residual_SVDs": 6144,
            "positive_rank_solutions": 24576, "residual_optimizer_updates": 0, "new_native_episodes": 0, "rl_actor_updates": 0},
        "boundary": parent["boundary"], "storage": "cache_all_new_references_and_paths_remote_pull_summary_only",
        "limitation": "fixed_outer_reference_saw_all_outer_training_labels_diagnostic_not_clean_inner_policy_selection"}


def result_path(run_name, *key): return strength_spec.result_path(run_name, *key)


def qualify(result, key, views):
    train, _ = training.cohort.split_rows(views[key[0]], key[1])
    groups = {(r["root"], r["scene_seed"]) for r in train}
    a = PARTITIONS[key[2]][key[3]]
    c = result["cell"]
    if result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract() or result["key"] != list(key):
        raise ValueError("Changed refit matrix protocol")
    expected = {(g, h) for g in (None, *groups) for h in (None, 0, 1)}
    actual = {(tuple(t["removed_group"]) if t["removed_group"] else None, t["half"]) for t in c["trials"]}
    if len(c["trials"]) != 33 or actual != expected: raise ValueError("Dropped canonical refit design")
    for t in c["trials"]:
        group = tuple(t["removed_group"]) if t["removed_group"] else None
        gs, n = groups-{group}, 54 if group else 60
        draws = list(a if t["half"] is None else a[2*t["half"]:2*t["half"]+2])
        full = group is None and t["half"] is None
        if t["training_states"] != n or t["training_draws"] != draws or set(t["arms"]) != set(learning.ARMS):
            raise ValueError("Changed refit state/draw/arm")
        if not full:
            f = t["reference_fit"]
            if (f["draws"] != draws or f["training_states"] != n or f["raw_steps"] != 600 or f["raw_parameters"] != 3393
                    or set(map(tuple, f["training_groups"])) != gs or set(map(tuple, f["wait_training_groups"])) != gs):
                raise ValueError("Reference refit inputs changed")
        for arm, record in t["arms"].items():
            if not full:
                for kind in joint.ANCHORS:
                    f = record["fits"][kind]
                    if (f["training_states"] != n or f["anchor"] != kind or f["steps"] != 0 or f["solver"] != "training_pair_TSVD"
                            or set(map(tuple, f["training_groups"])) != gs): raise ValueError("Residual refit inputs changed")
            if set(record["scopes"]) != set(learning.SCOPES): raise ValueError("Dropped refit scope")
            for scope, stats in record["scopes"].items():
                cases = n if scope == "fit_training" else 60
                if stats["all"]["cases"] != cases or sum(v["cases"] for v in stats["by_root"].values()) != cases:
                    raise ValueError("Dropped refit state")
                if set(stats["all"]["variants"]) != set(learning.VARIANTS): raise ValueError("Dropped refit variant")
    if len(c["blocks"]) != 20 or {(tuple(b["removed_group"]), b["half"]) for b in c["blocks"]} != {(g, h) for g in groups for h in (0, 1)}:
        raise ValueError("Changed matched refit blocks")
    for b in c["blocks"]:
        if set(b["trials"]) != set(CONFIGS): raise ValueError("Dropped matched refit factor")
        for name, n, k in (("54/2", 54, 2), ("54/4", 54, 4), ("60/2", 60, 2), ("60/4", 60, 4)):
            t = c["trials"][b["trials"][name]]
            if (t["training_states"] != n or len(t["training_draws"]) != k
                    or t["removed_group"] != (b["removed_group"] if n == 54 else None)
                    or t["half"] != (b["half"] if k == 2 else None)):
                raise ValueError("Matched refit comparison changed")
    r = c["reproduction"]
    if (r["choice_comparisons"] != 1680 or r["score_comparisons"] != 3360 or r["changed_choices"] or r["oof_changed_choices"]
            or r["prefix_count_differences"] or any(r[k] > 1e-12 for k in ("max_abs_difference", "oof_max_abs_difference", "prefix_max_abs_difference"))):
        raise ValueError("Refit endpoint controls failed")


def aggregate(cells):
    output = {}
    for config in CONFIGS:
        trials = [c["trials"][b["trials"][config]] for c in cells for b in c["blocks"]]
        output[config] = {arm: {scope: {"all": learning.averaged([t["arms"][arm]["scopes"][scope]["all"] for t in trials]),
            "by_root": {r: learning.averaged([t["arms"][arm]["scopes"][scope]["by_root"][r] for t in trials])
                for r in sorted(trials[0]["arms"][arm]["scopes"][scope]["by_root"])}} for scope in learning.SCOPES} for arm in learning.ARMS}
    return output


def summarize(results, views):
    if set(results) != set(KEYS): raise ValueError("Complete all32 refit cells")
    for key, r in results.items(): qualify(r, key, views)
    summaries = {v: aggregate([results[k]["cell"] for k in KEYS if k[0] == v]) for v in VIEWS}
    mixed = aggregate([results[k]["cell"] for k in KEYS if k[0] in ("mixed_a", "mixed_b")])
    checks = [r["cell"]["reproduction"] for r in results.values()]
    repro = {k: sum(r[k] for r in checks) for k in ("choice_comparisons", "score_comparisons", "changed_choices", "oof_changed_choices", "prefix_count_differences")}
    repro.update({k: max(r[k] for r in checks) for k in ("max_abs_difference", "oof_max_abs_difference", "prefix_max_abs_difference")})
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "views": summaries, "mixed_pooled": mixed, "regression_case": {"key": list(REGRESSION_KEY), "matrix": aggregate([results[REGRESSION_KEY]["cell"]])},
        "reproduction": repro, **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}


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
        raise ValueError("Complete Stage236 before matched refits")
    bank, _, registered, _ = training.cohort.source.source.load_data()
    views = training.cohort.source.matched_views(bank["training_data"], registered)
    if args.operation == "fit":
        key = args.view, args.fold, args.partition, args.direction
        old = json.loads((raw_directory(strength_spec.result_path(OUTER_RUN, *key))/"result.json").read_text())
        stable = json.loads((raw_directory(stability_spec.result_path(STABILITY_RUN, *key))/"result.json").read_text())
        strength_spec.qualify(old, key, views)
        stability_spec.qualify(stable, key, views)
        reference = json.loads((raw_directory(strength_spec.result_path(REFERENCE_RUN, *key))/"models.json").read_text())["reference"]
        paths = json.loads((raw_directory(strength_spec.result_path(MODEL_RUN, *key))/"models.json").read_text())["paths"]
        oof_path = raw_directory(training.precision_spec.result_path(training.binding_spec.SOURCE_RUN, *key))/"inner_oof.json"
        oof = {k: np.asarray(v) for k, v in json.loads(oof_path.read_text())["c2v4"].items()}
        train, held = training.cohort.split_rows(views[args.view], args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell, cache = learning.fit_matrix(train, held, panels, reference, paths, old["cell"], stable["cell"], oof)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "cell": cell}
        qualify(result, key, views)
        path = result_path(args.run_name, *key)
        cache.update(fixed_reference_run=REFERENCE_RUN, fixed_model_run=MODEL_RUN, key=list(key))
        write_json(raw_directory(path)/"models.json", cache)
        write_json(raw_directory(path)/"result.json", result)
        write_json(path, {k: v for k, v in result.items() if k != "cell"})
        print("NATIVE_REFIT_MATRIX_FIT_COMPLETE", flush=True)
    else:
        report = summarize({k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS}, views)
        write_json(ROOT/"results"/args.run_name/"report/result.json", report)
        print("NATIVE_REFIT_MATRIX_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
