#!/usr/bin/env python3
"""Compare cost losses without retraining any cached raw reference."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_refit_matrix_stage237 as previous
from freq_hrl.domains.transit import native_cost_objectives as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_cost_objectives_stage238_v2"
SOURCE_RUN = "native_transit_refit_matrix_stage237_development_20261015_r1"
VIEWS, PARTITIONS, KEYS, CONFIGS = previous.VIEWS, previous.PARTITIONS, previous.KEYS, previous.CONFIGS
training, joint = previous.training, learning.joint
result_path = previous.result_path


def contract():
    parent = previous.contract()
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "reference_run": previous.REFERENCE_RUN, "model_run": previous.MODEL_RUN, "outer_run": previous.OUTER_RUN,
        "cohort": parent["cohort"], "outer": parent["outer"], "partitions": parent["partitions"],
        "configs": list(CONFIGS), "objectives": list(learning.OBJECTIVES), "variants": list(learning.VARIANTS), "scopes": list(learning.previous.SCOPES),
        "representation": "cached_fixed_outer_reference_198_causal_features_same_unweighted_TSVD_basis_for_each_design_anchor_rank",
        "capacity": "all6_Stage234_rank_alpha_keys_frozen_no_reselection_equal_retained_rank_and_fit_pair_residual_RMS_ball",
        "all_mse": "reuse_cached_unweighted_nonanchor_pair_cost_MSE_solution_and_verify_reconstructed_subspace",
        "feasible_mse": "equal_state_MSE_over_nonanchor_actions_with_cached_raw_logit_le0_empty_states_zero_weight",
        "rank_regret": "equal_state_mean_feasible_pair_logistic_loss_weighted_by_absolute_training_cost_gap_divided_by_cached_cost_scale",
        "constraint": "unshrunk_fit_pair_residual_RMS_no_larger_than_cached_all_MSE_same_ball_for_all_objectives_no_tuned_penalty",
        "optimizer": {"solver": "SciPy_trust_constr_analytic_Hessian", "max_iterations": learning.MAX_ITERATIONS, "gtol": learning.GTOL,
            "xtol": 1e-12, "barrier_tol": 1e-10, "optimality_and_constraint_limit": 1e-8,
            "initial_barrier_parameter": 1e-10, "initial_barrier_tolerance": learning.GTOL,
            "initial": "cached_all_MSE_coefficients", "alpha": "same_frozen_postfit_alpha_no_new_strength_selection"},
        "training_labels": "same54_or60_states_same2_or4_training_draws_no_held_prefix_or_other_future_labels",
        "gate": "cached_predicted_raw_gate_unchanged_true_wait_only_for_evaluation",
        "averaging": parent["averaging"], "controls": "all33_cached_fixed_arm_all_MSE_scope_metrics_and_all_choices_reproduce",
        "budget": {"fit_tasks": 32, "report_tasks": 1, "unique_designs": 1056, "matched_blocks": 640,
            "residual_SVDs": 2112, "max_objective_solves": 12672, "new_raw_reference_fits": 0,
            "raw_wait_updates": 0, "new_native_episodes": 0, "rl_actor_updates": 0},
        "boundary": parent["boundary"], "limitation": parent["limitation"],
        "storage": "all_models_remote_summary_only_pull_no_raw_reference_cache_download"}


def qualify(result, key):
    if result["protocol"] != EXPERIMENT_PROTOCOL or result["contract"] != contract() or result["key"] != list(key):
        raise ValueError("Changed cost-objective protocol")
    cell = result["cell"]
    if len(cell["trials"]) != 33 or len(cell["blocks"]) != 20: raise ValueError("Dropped objective design")
    groups = {tuple(t["removed_group"]) for t in cell["trials"] if t["removed_group"]}
    designs = {(tuple(t["removed_group"]) if t["removed_group"] else None, t["half"]) for t in cell["trials"]}
    if len(groups) != 10 or designs != {(g, h) for g in (None, *groups) for h in (None, 0, 1)}:
        raise ValueError("Changed canonical objective designs")
    if {(tuple(b["removed_group"]), b["half"]) for b in cell["blocks"]} != {(g, h) for g in groups for h in (0, 1)}:
        raise ValueError("Changed matched objective blocks")
    for block in cell["blocks"]:
        if set(block["trials"]) != set(CONFIGS): raise ValueError("Dropped matched objective config")
        for cfg, i in block["trials"].items():
            t = cell["trials"][i]
            if (t["training_states"] != int(cfg[:2]) or len(t["training_draws"]) != int(cfg[-1])
                    or t["removed_group"] != (block["removed_group"] if cfg.startswith("54") else None)
                    or t["half"] != (block["half"] if cfg.endswith("2") else None)):
                raise ValueError("Changed matched objective alias")
    a = PARTITIONS[key[2]][key[3]]
    for t in cell["trials"]:
        expected = list(a if t["half"] is None else a[t["half"]*2:t["half"]*2+2])
        if t["training_states"] != (54 if t["removed_group"] else 60) or t["training_draws"] != expected:
            raise ValueError("Changed objective training inputs")
        if set(t["fits"]) != set(joint.ANCHORS) or set(t["scopes"]) != set(learning.previous.SCOPES):
            raise ValueError("Dropped objective anchor or scope")
        for kind, fits in t["fits"].items():
            keys = [cell["selection"][kind][f] for f in learning.previous.diagnostic.FIELDS.values()]
            levels = {learning.strength.CANDIDATES[k]["level"] for k in keys if k not in (None, "zero")}
            if len(fits) != 2*len(levels) or {(f["objective"], f["level"]) for f in fits} != {(o, l) for o in learning.OBJECTIVES[1:] for l in levels}:
                raise ValueError("Dropped selected-capacity objective solve")
            for f in fits:
                if (not f["success"] or f["objective"] not in learning.OBJECTIVES[1:] or f["parameter_count"] != f["retained_rank"]
                        or f["residual_rms_physical"] > f["residual_rms_limit_physical"]+1e-12 or f["loss_after"] > f["loss_before"]+1e-8*max(1., f["loss_before"])):
                    raise ValueError("Objective solver or matched capacity failed")
        for scope, stats in t["scopes"].items():
            n = t["training_states"] if scope == "fit_training" else 60
            if stats["all"]["cases"] != n or sum(x["cases"] for x in stats["by_root"].values()) != n:
                raise ValueError("Dropped objective cases")
            if any(set(p["variants"]) != set(learning.VARIANTS) for p in (stats["all"], *stats["by_root"].values())):
                raise ValueError("Dropped objective variants")
    repro = cell["reproduction"]
    if repro["choice_comparisons"] != 68040 or repro["changed_choices"] or max(repro["metric_max_abs_difference"], repro["subspace_prediction_max_abs_difference"]) > 1e-12:
        raise ValueError("Cached MSE controls failed")


def aggregate(cells):
    return {cfg: {scope: {"all": learning.averaged([c["trials"][b["trials"][cfg]]["scopes"][scope]["all"] for c in cells for b in c["blocks"]]),
        "by_root": {r: learning.averaged([c["trials"][b["trials"][cfg]]["scopes"][scope]["by_root"][r] for c in cells for b in c["blocks"]])
            for r in cells[0]["trials"][0]["scopes"][scope]["by_root"]}} for scope in learning.previous.SCOPES} for cfg in CONFIGS}


def summarize(results):
    if set(results) != set(KEYS): raise ValueError("Complete all32 objective cells")
    for key, result in results.items(): qualify(result, key)
    checks = [r["cell"]["reproduction"] for r in results.values()]
    fits = [f for r in results.values() for t in r["cell"]["trials"] for fs in t["fits"].values() for f in fs]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "views": {v: aggregate([results[k]["cell"] for k in KEYS if k[0] == v]) for v in VIEWS},
        "mixed_pooled": aggregate([results[k]["cell"] for k in KEYS if k[0].startswith("mixed")]),
        "regression_case": {"key": list(previous.REGRESSION_KEY), "matrix": aggregate([results[previous.REGRESSION_KEY]["cell"]])},
        "reproduction": {"choice_comparisons": sum(r["choice_comparisons"] for r in checks), "changed_choices": sum(r["changed_choices"] for r in checks),
            **{f: max(r[f] for r in checks) for f in ("metric_max_abs_difference", "subspace_prediction_max_abs_difference")}},
        "objective_solves": len(fits), "optimizer_iterations": sum(f["iterations"] for f in fits),
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
    if args.operation == "fit":
        key = args.view, args.fold, args.partition, args.direction
        bank, _, registered, _ = training.cohort.source.source.load_data()
        views = training.cohort.source.matched_views(bank["training_data"], registered)
        source = json.loads((raw_directory(result_path(SOURCE_RUN, *key))/"result.json").read_text())
        previous.qualify(source, key, views)
        selection = json.loads((raw_directory(result_path(previous.OUTER_RUN, *key))/"result.json").read_text())["cell"]["selection"]
        reference = json.loads((raw_directory(result_path(previous.REFERENCE_RUN, *key))/"models.json").read_text())["reference"]
        paths = json.loads((raw_directory(result_path(previous.MODEL_RUN, *key))/"models.json").read_text())["paths"]
        cached = json.loads((raw_directory(result_path(SOURCE_RUN, *key))/"models.json").read_text())
        train, held = training.cohort.split_rows(views[args.view], args.fold)
        panels = training.cohort.source.source.wait_source.source.diagnostic.load_panels()
        cell, models = learning.compare_cell(train, held, panels, reference, paths, source["cell"]|{"selection": selection}, cached)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "cell": cell}
        qualify(result, key)
        path = result_path(args.run_name, *key)
        write_json(raw_directory(path)/"models.json", models)
        write_json(raw_directory(path)/"result.json", result)
        write_json(path, {k: v for k, v in result.items() if k != "cell"})
        print("NATIVE_COST_OBJECTIVES_FIT_COMPLETE", flush=True)
    else:
        report = summarize({k: json.loads((raw_directory(result_path(args.run_name, *k))/"result.json").read_text()) for k in KEYS})
        write_json(ROOT/"results"/args.run_name/"report/result.json", report)
        print("NATIVE_COST_OBJECTIVES_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
