#!/usr/bin/env python3
"""Learn causal service residuals while keeping every fleet-risk head frozen."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_component_probe_stage281 as source
from scripts import run_native_transit_effective_risk_stage278 as geometry
from freq_hrl.domains.transit import native_service_transfer as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_service_transfer_stage282_v1"
SOURCE_RUN = "native_transit_component_probe_stage281_20261017_r1"
RISK_RUN = "native_transit_effective_risk_stage278_20261017_r1"
cache, KEYS, VIEWS, SCOPES = source.cache, source.KEYS, source.VIEWS, source.SCOPES


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "risk_run": RISK_RUN, "cohort": source.contract()["cohort"], "arms": list(learning.ARMS),
        "fit": "service_only_raw_prior_residual_physical_MSE_assigned16_state_balanced_eligible_actions; full60_or_prefix54_only; no_held_or_opposite_labels_in_fit_or_clip_radius",
        "geometry": "reuse_Stage278_cached_basis_normalizers_context_stats_and_feature_only_df_matched_penalties; Linear9_lambda0.01_and_both73_maps; no_geometry_or_penalty_selection",
        "clip": "common_service_radius_per_fit=sqrt_state_balanced_training_mean_service_residual_squared; original_correction0; pointwise_plus_minus_radius_and_uncut_controls; no_risk_radius_reuse_or_query_rescaling",
        "controllers": "cross_each_service_head_with_frozen_raw_Linear9_observed_df_risk; raw_wait_gate_candidates_and_original_first_ties_unchanged; zero_service_update_reproduces_all3_controls",
        "benchmarks": "reuse_Stage281_assigned16_MC_service_and_saved_choices_with_all3_risks; benchmark_query_labels_never_enter_learned_predictions_or_fit; not_deployable",
        "evaluation": "same64_queries_per_cell_train_retained_excluded_full_held_prefix_head_held; assigned_opposite16_and_fixed8+8_evaluation_halves; primary_mixed_excluded_opposite_and_held_opposite; primary_learned_comparisons_clipped_each_vs_own_frozen_risk_control_no_winner_selection",
        "controls": "frozen_controls_and_MC_service_choices_replay_exactly; per_cell_and_all_group_bank_scope_half_outcomes_and_raw_MC_service_MSE_reproduce_Stage281; all_predictions_actions_saved_before_evaluation",
        "budget": {"fit_tasks": 1, "cached_cells": 32, "cached_contexts": 704, "cached_queries": 2048,
            "model_state_exposures": 84480, "new_service_head_fits": 2112,
            "new_raw_risk_basis_or_prior_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "models_features_service_predictions_and_query_outcomes_server_only; compact_scalar_report_pull",
        "boundary": "development_supervised_single_intervention_service_transfer_not_end_to_end_HRL_new_prefix_confirmation_seed_CI_or_deployment; MC_benchmark_is_information_probe; Stage253_248_169_closed; defaults_unchanged"}


def fit_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    old = json.loads((here(RISK_RUN)/"models.json").read_text())
    linear_models = json.loads((here(geometry.source.SOURCE_RUN)/"models.json").read_text())
    queries = json.loads((here(source.source.SOURCE_RUN)/"actions.json").read_text())
    with np.load(here(geometry.source.BASIS_RUN)/"predictions.npz") as saved:
        basis = saved["basis"].copy()
    reference = geometry.source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    if {(r["root"], r["scene_seed"]) for r in train}&{(r["root"], r["scene_seed"]) for r in held}:
        raise ValueError("Outer prefix overlap")
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    observed = {s: learning.state.observed_context(rs) for s, rs in rows.items()}
    z, eligible = {}, {}
    for label, d in data.items():
        x = learning.probe.policy.joint.relative(d["x"][..., :198], d["originals"])@basis
        z[label] = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=-1)
        eligible[label] = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
    folds = {tuple(g): np.asarray(ix) for g, ix, _ in cache.learning.joint.binding.policy.prefix_folds(train)}
    models, predictions = [], {}
    raw_error = 0.
    with np.load(here(RISK_RUN)/"predictions.npz") as saved:
        for m in old:
            i, ix = m["index"], np.asarray(m["fit_indices"])
            linear = linear_models[i]
            if any(m[f] != linear[f] for f in ("bank", "removed", "fit_indices", "training_draws")) or linear["penalty"] != .01:
                raise ValueError("Changed cached Linear9 context")
            draws, _ = cache.sampling.learning.draw_assignment(m["bank"], "expanded")
            if draws != m["training_draws"] or not np.array_equal(ix, np.arange(60) if m["removed"] is None else folds[tuple(m["removed"])]):
                raise ValueError("Changed fit population/futures")
            d = {k: v[ix] for k, v in data["train"].items()}
            prior = learning.probe.policy.joint.relative(d["heads"], d["originals"])[..., 0]-learning.probe.policy.joint.relative(d["components"], d["originals"])[..., 1]
            values = cache.learning.base.training_draws([train[j] for j in ix], panels, draws)[..., 2]
            target = values-values[np.arange(len(ix)), :, d["originals"]][:, :, None]
            model = learning.fit(m, linear["maps"]["linear"], z["train"][ix], observed["train"][ix], target.mean(axis=1), prior, eligible["train"][ix])
            models.append({"index": i, "bank": m["bank"], "removed": m["removed"], "fit_indices": ix, "training_draws": draws, **model})
            for qi, q in enumerate(queries):
                if q["index"] != i:
                    continue
                label, ids = q["label"], np.asarray(q["ids"])
                d = {k: v[ids] for k, v in data[label].items()}
                if q["bank"] != m["bank"] or q["removed"] != m["removed"] or not np.array_equal(q["eligible"], eligible[label][ids]):
                    raise ValueError("Changed query context")
                risks = {a: saved[f'{i}_{q["scope"]}_{a}'].copy() for a in learning.probe.policy.ARMS}
                replay, error = learning.probe.policy.decisions(d, risks)
                raw_error = max(raw_error, error)
                if any(not np.array_equal(replay[a], q["choices"][a]) for a in learning.probe.policy.ARMS):
                    raise ValueError("Changed frozen risk choices")
                prior = learning.probe.policy.joint.relative(d["heads"], d["originals"])[..., 0]-risks["raw_prior"]
                services, clips = learning.predict(model, z[label][ids], observed[label][ids], prior, eligible[label][ids], d["originals"])
                benchmark = json.loads((here(SOURCE_RUN)/f"proposal_{qi}.json").read_text())
                if any(benchmark[f] != q[f] for f in ("index", "bank", "scope", "label", "ids")):
                    raise ValueError("Changed MC benchmark query")
                services["mc"] = np.asarray(benchmark["components"]["service_mc_all16"])
                choices = learning.decisions(d, risks, services)
                if any(not np.array_equal(choices[a], replay[a]) for a in learning.CONTROLS) or any(
                        not np.array_equal(choices[a], benchmark["choices"][b]) for a, b in learning.BENCHMARKS.items()):
                    raise ValueError("Frozen control/MC benchmark replay failed")
                predictions[qi] = {**q, "choices": choices, "services": services, "clips": clips, "radius": model["radius"]}
    if len(old) != 22 or len(predictions) != 64:
        raise ValueError("Incomplete service fit/query coverage")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"models.json", models)
    write_json(directory/"predictions.json", [predictions[i] for i in range(64)])
    print(f"SERVICE_TRANSFER_PREDICTIONS_FROZEN {key}", flush=True)
    previous = json.loads((here(SOURCE_RUN)/"outcomes.json").read_text())
    records, error = [], 0.
    for qi in range(64):
        q = predictions[qi]
        ids, label = np.asarray(q["ids"]), q["label"]
        for fi, (future, bank) in enumerate((("same", q["bank"]), ("opposite", 1-q["bank"]))):
            draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws([rows[label][j] for j in ids], panels, draws)
            outcomes = learning.probe.policy.outcomes(values, q["choices"])
            old_record = previous[2*qi+fi]
            if (old_record["bank"], old_record["scope"]) != (q["bank"], q["scope"]+"_"+future):
                raise ValueError("Changed outcome record order")
            for h, arms in outcomes.items():
                for a, b in {**{a: a for a in learning.CONTROLS}, **learning.BENCHMARKS}.items():
                    before = old_record["outcomes"][h][b]
                    for f in ("cases", "changed_count", "linear9_disagreement_count"):
                        error = max(error, abs(arms[a][f]-before[f]))
                    for f in learning.probe.policy.FIELDS:
                        error = max(error, abs(arms[a]["sums"][f]-before["sums"][f]))
            metrics = learning.metrics(q["services"], q["clips"], q["services"]["raw"], values,
                data[label]["originals"][ids], eligible[label][ids], q["radius"])
            for a, b in (("raw", "service_raw"), ("mc", "service_mc_all16")):
                error = max(error, abs(metrics[a]["squared_error_sum"]-old_record["component_errors"]["sums"][b]))
            records.append({"bank": q["bank"], "scope": q["scope"]+"_"+future, "metrics": metrics, "outcomes": outcomes})
    if error > 1e-10:
        raise ValueError("Stage281 cell outcome/MSE replay failed")
    write_json(directory/"outcomes.json", records)
    return {"key": list(key), "records": records, "head_fits": sum(m["head_fits"] for m in models),
        "model_state_exposures": sum(len(q["ids"]) for q in predictions.values()),
        "normal_error": max(m["normal_error"] for m in models), "raw_error": raw_error, "replay_error": error}


def report(cells, before):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["head_fits"] for c in cells) != 2112 or sum(c["model_state_exposures"] for c in cells) != 84480:
        raise ValueError("Incomplete Stage282 coverage")
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
    output, error = {}, 0.
    for g, predicate in groups.items():
        rows = [r for c in cells if predicate(c["key"]) for r in c["records"]]
        output[g] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[g][label] = {}
            for scope in SCOPES:
                rs = [r for r in rows if r["scope"] == scope and (bank is None or r["bank"] == bank)]
                old = before["groups"][g][label][scope]
                metrics = {a: learning.base.summarize([r["metrics"][a] for r in rs], True) for a in learning.SERVICES}
                outcomes = {h: {a: learning.probe.policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.ARMS}
                    for h in learning.probe.policy.DRAW_GROUPS}
                for h, arms in outcomes.items():
                    if h not in old["outcomes"]:
                        continue
                    for a, b in {**{a: a for a in learning.CONTROLS}, **learning.BENCHMARKS}.items():
                        error = max(error, *(abs(v-old["outcomes"][h][b][f]) for f, v in arms[a].items()))
                for a, b in (("raw", "service_raw"), ("mc", "service_mc_all16")):
                    error = max(error, abs(metrics[a]["MSE"]-old["component_errors"]["MSE"][b]))
                if g != "mixed_pooled" or scope not in ("excluded_opposite", "held_opposite", "matched_held_opposite"):
                    outcomes = {"all16": outcomes["all16"]}
                output[g][label][scope] = {"service_metrics": metrics, "outcomes": outcomes}
    if error > 1e-10:
        raise ValueError("Stage281 group replay failed")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "fits": {"service_head_fits": 2112, "max_normal_error": max(c["normal_error"] for c in cells)},
        "reproduction": {"raw_risk_error": max(c["raw_error"] for c in cells),
            "cell_replay_error": max(c["replay_error"] for c in cells), "group_replay_error": error},
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
        print(f"SERVICE_TRANSFER_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", cells)
    write_json(path, report(cells, before))
    print("SERVICE_TRANSFER_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
