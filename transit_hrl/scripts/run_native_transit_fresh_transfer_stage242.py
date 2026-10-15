#!/usr/bin/env python3
"""Reuse independent future panels to separate prefix and refit transfer."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_frozen_future_stage241 as source
from freq_hrl.domains.transit import native_fresh_transfer as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_fresh_transfer_stage242_v1"
SOURCE_RUN = "native_transit_frozen_future_stage241_development_20261015_r1"
CONSISTENCY_RUN = source.PREVIOUS_RUN
KEYS, VIEWS, training, result_path = source.KEYS, source.VIEWS, source.prior.training, source.prior.result_path
FOCAL_VARIANTS = ("raw", "joint_raw_half_safe", "all_mse_original_pooled_safe")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL, "consistency_run": CONSISTENCY_RUN,
        "selection_run": source.SOURCE_RUN, "refit_run": source.REFIT_RUN,
        "endpoints": list(learning.ENDPOINTS), "scopes": list(learning.SCOPES), "label_sets": list(learning.LABELS),
        "variants": list(learning.VARIANTS), "cohort": source.contract()["cohort"],
        "focal_variants": list(FOCAL_VARIANTS), "secondary": "all_MSE_original_pooled_named_after_Stage241_inspection_not_a_new_confirmatory_primary",
        "prediction": "all_Stage239_selections_and_Stage237_239_240_models_fixed_actions_written_before_reading_fresh_panels",
        "scopes_detail": "54_seen_prefix_states_6_leave_prefix_out_states_60_full_train_states_60_outer_held_prefix_states",
        "exposure": "54/2_seen54_old_inner2_is_training_exposed_full60/4_train60_old_outer4_is_training_exposed_all_fresh_labels_unseen_by_all_models",
        "old_matched": "probe6_uses_old_OOF_opposite2_other_scopes_use_old_outer_unused4_exact_Stage240_controls",
        "controls": "all5_old_matched_Stage240_probe_train_held_metrics_and_80640_Stage241_sealed_probe_action_records_reproduce",
        "metrics": "own_common_raw_cost_wait_risk_regret_score_MSE_action_switch_plus_fleet_nonfleet_cost_and_draw_wait_harm_components",
        "averaging": "equal20_prefix_future_blocks_then_equal_cells_no_independent_seed_CI",
        "budget": {"prepare_tasks": 1, "evaluation_tasks": 32, "report_tasks": 1,
            "new_raw_reference_fits": 0, "new_residual_fits": 0, "new_native_episodes": 0, "training_updates": 0},
        "storage": "compressed_panels_actions_full_block_and_aggregate_metrics_remote_only_pull_focal_matrix_plus_all21_fresh8_endpoint_summary",
        "boundary": "reused_development_data_fresh241_already_inspected_no_new_confirmation_or_candidate_reselection_defaults_unchanged"}


def panel_path(run_name):
    return ROOT/"results"/run_name/"prepare_raw/panels.npz"


def prepare(run_name):
    qualified = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    if not (qualified["protocol"] == source.EXPERIMENT_PROTOCOL and qualified["contract"] == source.contract() and qualified["coverage_qualified"]):
        raise ValueError("Complete qualified Stage241 before reusing its panels")
    old = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
    fresh = {}
    fields = ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")
    for root, scenario, replica, probe in source.coverage.states():
        values = []
        for draw in source.DRAWS:
            c = json.loads((raw_directory(source.draw_path(SOURCE_RUN, root, scenario, replica, probe, draw))/"result.json").read_text())
            values.append([[l[f] for f in fields] for l in c["labels"]])
        fresh[root, source.coverage.noise.source.scene_seed(root, scenario, replica), probe] = {"values": np.asarray(values)}
    path = panel_path(run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    learning.write_panel_cache(path, old, fresh)
    write_json(ROOT/"results"/run_name/"prepare/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "states": len(fresh), "prefixes": len({k[:2] for k in fresh}),
        "panel_shape": [len(fresh), 8, 6, 4], "fields": list(fields), "cache_bytes": path.stat().st_size,
        "new_native_episodes": 0, "training_updates": 0})
    print("FRESH_TRANSFER_PANEL_CACHE_COMPLETE", flush=True)


def evaluate(run_name, key, views):
    previous = source.prior.previous
    load = lambda run, name: json.loads((raw_directory(result_path(run, *key))/name).read_text())
    selected = load(source.SOURCE_RUN, "result.json")
    previous.qualify(selected, key, views)
    source240 = load(CONSISTENCY_RUN, "result.json")
    source.prior.qualify(source240, key, views)
    refit = load(source.REFIT_RUN, "result.json")
    reference = load(source.prior.contract()["reference_run"], "models.json")["reference"]
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    frozen = learning.freeze(train, held, reference, load(source.REFIT_RUN, "models.json"), refit["cell"], selected["cell"],
        load(source.SOURCE_RUN, "models.json"), load(CONSISTENCY_RUN, "models.json"))
    sealed = next(c for c in source.load_seal(SOURCE_RUN)["cells"] if c["key"] == list(key))
    comparisons = learning.match_sealed_actions(frozen, sealed)
    path = result_path(run_name, *key)
    write_json(raw_directory(path)/"actions.json", learning.action_cache(frozen))
    old, fresh = learning.read_panel_cache(panel_path(run_name))
    cell = learning.evaluate(frozen, old, fresh, selected["cell"], source240["cell"])
    cell["reproduction"]["sealed_action_records"] = comparisons
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "cell": cell}
    qualify(result)
    write_json(raw_directory(path)/"result.json", result)
    write_json(raw_directory(path)/"aggregate.json", {k: v for k, v in result.items() if k != "cell"} | {
        "coverage_qualified": True, "matrix": learning.aggregate([cell]), "reproduction": cell["reproduction"]})
    write_json(path, {k: v for k, v in result.items() if k != "cell"})
    print(f"FRESH_TRANSFER_EVALUATION_COMPLETE {key}", flush=True)


def qualify(result):
    if result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract() or tuple(result["key"]) not in KEYS:
        raise ValueError("Changed fresh transfer protocol")
    cell = result["cell"]
    if len(cell["blocks"]) != 20: raise ValueError("Dropped prefix/future block")
    groups = {(tuple(b["held_group"]), b["half"]) for b in cell["blocks"]}
    if len(groups) != 20 or {h for _, h in groups} != {0, 1}: raise ValueError("Dropped matched prefix/future direction")
    for b in cell["blocks"]:
        if set(b["endpoints"]) != set(learning.ENDPOINTS): raise ValueError("Dropped cached endpoint")
        for scopes in b["endpoints"].values():
            if set(scopes) != set(learning.SCOPES): raise ValueError("Dropped transfer scope")
            for scope, labels in scopes.items():
                if set(labels) != set(learning.LABELS): raise ValueError("Dropped old/fresh future comparison")
                for stats in labels.values():
                    n = {"probe6": 6, "seen54": 54, "train60": 60, "held60": 60}[scope]
                    if stats["all"]["cases"] != n or sum(p["cases"] for p in stats["by_root"].values()) != n:
                        raise ValueError("Dropped source-prefix state")
                    if any(set(p["variants"]) != set(learning.VARIANTS) for p in (stats["all"], *stats["by_root"].values())):
                        raise ValueError("Dropped frozen candidate")
    if cell["reproduction"]["max_abs_difference"] > 1e-12 or cell["reproduction"]["sealed_action_records"] != 2520:
        raise ValueError("Old endpoint or sealed action changed")


def summarize(results):
    if set(results) != set(KEYS): raise ValueError("Complete all32 cached transfer cells")
    for key, r in results.items():
        if (r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or r["key"] != list(key)
                or not r["coverage_qualified"] or r["reproduction"]["sealed_action_records"] != 2520
                or r["reproduction"]["max_abs_difference"] > 1e-12): raise ValueError("Changed qualified cell aggregate")
    cells = [results[k] for k in KEYS]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "views": {v: learning.merge_matrices([results[k]["matrix"] for k in KEYS if k[0] == v]) for v in VIEWS},
        "mixed_pooled": learning.merge_matrices([results[k]["matrix"] for k in KEYS if k[0].startswith("mixed")]),
        "regression_case": results[source.prior.REGRESSION_KEY]["matrix"],
        "reproduction": {"sealed_action_records": sum(c["reproduction"]["sealed_action_records"] for c in cells),
            "metric_records": sum(c["reproduction"]["records"] for c in cells),
            "max_abs_difference": max(c["reproduction"]["max_abs_difference"] for c in cells)},
        **contract()["budget"], "default_changed": False, "native_deployment_eligible": False}


def compact_report(report):
    def matrix(m):
        return {e: {s: {label: {"all": {**p["all"], "variants": {v: p["all"]["variants"][v] for v in FOCAL_VARIANTS}},
            "by_root": {r: {**x, "variants": {v: x["variants"][v] for v in FOCAL_VARIANTS}} for r, x in p["by_root"].items()}}
            for label, p in labels.items()} for s, labels in scopes.items()} for e, scopes in m.items()}
    return {**{k: v for k, v in report.items() if k not in ("views", "mixed_pooled", "regression_case")},
        "views": {v: matrix(m) for v, m in report["views"].items()},
        "mixed_pooled": matrix(report["mixed_pooled"]), "regression_case": matrix(report["regression_case"]),
        "all_registered_fresh8": {e: {s: report["mixed_pooled"][e][s]["fresh8"] for s in learning.SCOPES}
            for e in (learning.ENDPOINTS[0], learning.ENDPOINTS[-1])},
        "full_aggregate_storage": "report_raw/result.json_server_only"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("prepare", "evaluate", "report"), required=True)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=source.prior.PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.operation == "prepare": prepare(args.run_name)
    elif args.operation == "evaluate":
        bank, _, registered, _ = training.cohort.source.source.load_data()
        views = training.cohort.source.matched_views(bank["training_data"], registered)
        evaluate(args.run_name, (args.view, args.fold, args.partition, args.direction), views)
    else:
        report = summarize({k: json.loads((raw_directory(result_path(args.run_name, *k))/"aggregate.json").read_text()) for k in KEYS})
        path = ROOT/"results"/args.run_name/"report/result.json"
        write_json(raw_directory(path)/"result.json", report)
        write_json(path, compact_report(report))
        print("FRESH_TRANSFER_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
