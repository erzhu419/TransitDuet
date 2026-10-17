#!/usr/bin/env python3
"""Fit matched state-level service heads with every action-contrast head frozen."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_centering_stage284 as source
from freq_hrl.domains.transit import native_service_level as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_service_level_stage285_v1"
SOURCE_RUN = "native_transit_service_centering_stage284_20261017_r1"
cache, KEYS, VIEWS, SCOPES, policy = source.cache, source.KEYS, source.VIEWS, source.SCOPES, source.policy
PRIMARY = source.PRIMARY
transfer = source.source.source


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arms": list(learning.ARMS), "report_services": list(learning.REPORT_SERVICES), "report_arms": list(learning.REPORT_ARMS),
        "fit": "assigned16_training_only_state_mean_eligible_nonoriginal_service_residual; same_full60_prefix54_contexts; no_held_or_opposite_labels_in_fit; skip_empty_eligible_states; state_balanced_ridge",
        "representation": "two9_parameter_level_heads_action_mean_of_cached_Linear9_coordinates_vs8_observed_context_tanh_coordinates_using_cached_context_stats; fit_only_level_normalizers_on_fit_states; no_basis_or_raw_risk_contrast_refit",
        "complexity": "common_target_df=min_of_both_feature_only_df_at_lambda0.01_per_fit; increase_each_penalty_to_match_common_target_without_labels; uniform_training_mean_reproduces_cached_intercept",
        "controllers": "new_level_only_and_new_level_plus_each_of3_frozen_uncut_centered_action_contrasts; unchanged_Stage282_training_radius_after_addition_original_anchor0; same3_risks_raw_wait_gate_candidates_ties; all_choices_frozen_before_evaluation",
        "evaluation": "same_train_retained_excluded_full60_held_prefix54_held_all_views_banks; assigned_opposite16_fixed8+8; primary_mixed_excluded_opposite_held_opposite; no_selected_winner; retain_uniform_derived_level_original_full_and_MC_controls",
        "diagnostics": "exact_state_level_plus_contrast_MSE_pair_inversion_and_switch_partition_as_Stage284; costs_relative_raw_switches_relative_own_risk; replay_all69_old_choices_cell_outcomes_metrics_and_all_compact_group_values_before_report_compaction",
        "budget": {"fit_tasks": 1, "cached_cells": 32, "cached_contexts": 704, "cached_queries": 2048,
            "model_state_exposures": 84480, "new_level_head_fits": 1408,
            "new_raw_risk_basis_or_contrast_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "models_predictions_and_full_query_outcomes_server_only_compact_scalar_report_pull",
        "boundary": "development_supervised_calibration_not_end_to_end_HRL_new_prefix_confirmation_seed_CI_or_deployment; MC_not_deployable; Stage253_248_169_closed; defaults_unchanged"}


def fit_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    old_models = json.loads((here(source.source.SOURCE_RUN)/"models.json").read_text())
    queries = json.loads((here(SOURCE_RUN)/"predictions.json").read_text())
    with np.load(here(transfer.geometry.source.BASIS_RUN)/"predictions.npz") as archive:
        basis = archive["basis"].copy()
    reference = transfer.geometry.source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    observed = {s: learning.service.state.observed_context(rs) for s, rs in rows.items()}
    z, eligible = {}, {}
    for label, d in data.items():
        x = policy.joint.relative(d["x"][..., :198], d["originals"])@basis
        z[label] = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=-1)
        eligible[label] = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
    folds = {tuple(g): np.asarray(ix) for g, ix, _ in cache.learning.joint.binding.policy.prefix_folds(train)}
    models, predictions = [], {}
    with np.load(here(transfer.RISK_RUN)/"predictions.npz") as archive:
        for old in old_models:
            ix = np.asarray(old["fit_indices"])
            if not np.array_equal(ix, np.arange(60) if old["removed"] is None else folds[tuple(old["removed"])]):
                raise ValueError("Changed level training population")
            draws, _ = cache.sampling.learning.draw_assignment(old["bank"], "expanded")
            if draws != old["training_draws"]:
                raise ValueError("Changed assigned training futures")
            coordinates = {s: learning.features(old, z[s], observed[s], eligible[s]) for s in rows}
            d = {k: v[ix] for k, v in data["train"].items()}
            prior = policy.joint.relative(d["heads"], d["originals"])[..., 0]-policy.joint.relative(d["components"], d["originals"])[..., 1]
            values = cache.learning.base.training_draws([train[j] for j in ix], panels, draws)[..., 2]
            target = (values-values[np.arange(len(ix)), :, d["originals"]][:, :, None]).mean(axis=1)
            model = learning.fit(old, {k: v[ix] for k, v in coordinates["train"].items()}, target, prior, eligible["train"][ix])
            models.append({"index": old["index"], "bank": old["bank"], "removed": old["removed"], "fit_indices": ix, "training_draws": draws, **model})
            for qi, q in enumerate(queries):
                if q["index"] != old["index"]:
                    continue
                label, ids = q["label"], np.asarray(q["ids"])
                if (q["bank"], q["removed"], q["radius"]) != (old["bank"], old["removed"], model["radius"]) or not np.array_equal(eligible[label][ids], q["eligible"]):
                    raise ValueError("Changed frozen query context")
                d = {k: v[ids] for k, v in data[label].items()}
                risks = {a: archive[f'{q["index"]}_{q["scope"]}_{a}'].copy() for a in learning.service.CONTROLS}
                services = {a: np.asarray(v) for a, v in q["services"].items()}
                services, clips = learning.predict(model, {k: v[ids] for k, v in coordinates[label].items()}, services, eligible[label][ids], d["originals"])
                choices = learning.service.decisions(d, risks, services)
                if any(not np.array_equal(choices[a], q["choices"][a]) for a in learning.previous.ARMS):
                    raise ValueError("Stage284 frozen choice replay failed")
                predictions[qi] = {**q, "services": services, "choices": choices,
                    "clips": {**{a: np.asarray(v) for a, v in q["clips"].items()}, **clips}}
    if len(models) != 22 or len(predictions) != 64:
        raise ValueError("Incomplete Stage285 fit/query coverage")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"models.json", models)
    write_json(directory/"predictions.json", [predictions[i] for i in range(64)])
    print(f"SERVICE_LEVEL_PREDICTIONS_FROZEN {key}", flush=True)
    before = json.loads((here(SOURCE_RUN)/"outcomes.json").read_text())
    records, error, identity, switch_error = [], 0., 0., 0.
    for qi in range(64):
        q = predictions[qi]
        label, ids = q["label"], np.asarray(q["ids"])
        originals, mask = data[label]["originals"][ids], eligible[label][ids]
        for fi, (future, bank) in enumerate((("same", q["bank"]), ("opposite", 1-q["bank"]))):
            draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws([rows[label][i] for i in ids], panels, draws)
            outcomes = policy.outcomes(values, q["choices"])
            metrics = learning.service.metrics(q["services"], q["clips"], q["services"]["raw"], values, originals, mask, q["radius"])
            geometry, residual = learning.previous.error_geometry(values, q["services"], originals, mask)
            identity = max(identity, residual)
            scope = q["scope"]+"_"+future
            switches = learning.previous.previous.switch_stats(values, {a: q["choices"][a] for a in learning.REPORT_ARMS}, originals, learning.ARM_RISK) if scope in PRIMARY else {}
            old = before[2*qi+fi]
            if (old["bank"], old["scope"]) != (q["bank"], scope):
                raise ValueError("Changed cached outcome order")
            for h, arms in outcomes.items():
                for a in learning.previous.ARMS:
                    for f in ("cases", "changed_count", "linear9_disagreement_count"):
                        error = max(error, abs(arms[a][f]-old["outcomes"][h][a][f]))
                    for f in policy.FIELDS:
                        error = max(error, abs(arms[a]["sums"][f]-old["outcomes"][h][a]["sums"][f]))
                for a, entry in switches.get(h, {}).items():
                    control = arms[learning.ARM_RISK[a]]
                    switch_error = max(switch_error, abs(sum(t["cases"] for t in entry["types"].values())-entry["cases"]))
                    for f in policy.FIELDS:
                        switch_error = max(switch_error, abs(sum(t["sums"][f] for t in entry["types"].values())-(arms[a]["sums"][f]-control["sums"][f])))
            for a in learning.previous.SERVICES:
                for f, v in metrics[a].items():
                    if v is not None:
                        error = max(error, abs(v-old["metrics"][a][f]))
                for h, entries in geometry.items():
                    for f, v in entries[a]["sums"].items():
                        error = max(error, abs(v-old["geometry"][h][a]["sums"][f]))
                    for f in ("cases", "rank_cases", "rank_error_sum"):
                        error = max(error, abs(entries[a][f]-old["geometry"][h][a][f]))
            records.append({"bank": q["bank"], "scope": scope, "metrics": metrics,
                "outcomes": outcomes, "geometry": geometry, "switches": switches})
    if max(error, switch_error) > 1e-10:
        raise ValueError("Stage284 cell replay or switch partition failed")
    write_json(directory/"outcomes.json", records)
    return {"key": list(key), "records": records, "model_state_exposures": sum(len(q["ids"]) for q in predictions.values()),
        "head_fits": sum(m["head_fits"] for m in models), "normal_error": max(m["normal_error"] for m in models),
        "df_error": max(m["df_error"] for m in models), "uniform_error": max(m["uniform_error"] for m in models),
        "replay_error": error, "identity_error": identity, "switch_error": switch_error}


def report(cells, before):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["model_state_exposures"] for c in cells) != 84480 or sum(c["head_fits"] for c in cells) != 1408:
        raise ValueError("Incomplete Stage285 coverage")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, error = {}, 0.
    report_arms = (*learning.previous.REPORT_ARMS, *(a for a in learning.REPORT_ARMS if a not in learning.previous.REPORT_ARMS))
    for g, predicate in groups.items():
        rows = [r for c in cells if predicate(c["key"]) for r in c["records"]]
        output[g] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[g][label] = {}
            for scope in SCOPES:
                rs = [r for r in rows if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                old = before["groups"][g][label][scope]
                halves = tuple(old["outcomes"])
                metrics = {a: learning.base.summarize([r["metrics"][a] for r in rs], True) for a in learning.SERVICES}
                outcomes = {h: {a: policy.summarize([r["outcomes"][h][a] for r in rs]) for a in report_arms} for h in halves}
                geometry = {h: {a: learning.previous.summarize([r["geometry"][h][a] for r in rs]) for a in learning.SERVICES} for h in halves}
                for a, m in old["service_metrics"].items():
                    error = max(error, *(abs(metrics[a][f]-v) for f, v in m.items() if v is not None))
                for h, arms in old["outcomes"].items():
                    for a, m in arms.items():
                        error = max(error, *(abs(outcomes[h][a][f]-v) for f, v in m.items()))
                    for a, m in old["error_geometry"][h].items():
                        error = max(error, *(abs(geometry[h][a][f]-v) for f, v in m.items() if v is not None))
                result = {"service_metrics": {a: {f: metrics[a][f] for f in ("MSE", "bias", "clipped_action_fraction")} for a in learning.REPORT_SERVICES},
                    "outcomes": {h: {a: outcomes[h][a] for a in learning.REPORT_ARMS} for h in halves},
                    "error_geometry": {h: {a: geometry[h][a] for a in learning.REPORT_SERVICES} for h in halves}}
                if g == "mixed_pooled" and scope in PRIMARY:
                    result["switches"] = {a: learning.previous.previous.summarize_switches([r["switches"]["all16"][a] for r in rs]) for a in learning.REPORT_ARMS}
                output[g][label][scope] = result
    if error > 1e-10:
        raise ValueError("Stage284 aggregate replay failed")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {"level_head_fits": 1408, "parameters_per_head": 9, "normal_error": max(c["normal_error"] for c in cells),
            "df_error": max(c["df_error"] for c in cells), "uniform_error": max(c["uniform_error"] for c in cells)},
        "reproduction": {"cell_replay_error": max(c["replay_error"] for c in cells), "group_replay_error": error,
            "MSE_identity_error": max(c["identity_error"] for c in cells), "switch_partition_error": max(c["switch_error"] for c in cells)},
        "default_changed": False, "native_deployment_eligible": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    path = ROOT/"results"/a.run_name/"report/result.json"
    directory = raw_directory(path)
    cells = []
    for key in KEYS:
        cells.append(fit_cell(directory/("_".join(map(str, key))), key, views, panels))
        print(f"SERVICE_LEVEL_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", [{k: v for k, v in c.items() if k != "records"} for c in cells])
    write_json(path, report(cells, before))
    print("SERVICE_LEVEL_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
