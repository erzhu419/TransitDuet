#!/usr/bin/env python3
"""Separate frozen service offset calibration from feature-dependent slopes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_service_transfer_stage282 as source
from freq_hrl.domains.transit import native_service_offset as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_service_offset_stage283_v1"
SOURCE_RUN = "native_transit_service_transfer_stage282_20261017_r1"
cache, KEYS, VIEWS, SCOPES = source.cache, source.KEYS, source.VIEWS, source.SCOPES
policy = learning.previous.probe.policy
PRIMARY = ("excluded_opposite", "held_opposite")


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "arms": list(learning.ARMS), "report_arms": list(learning.REPORT_ARMS),
        "decomposition": "cached_uncut_service_delta=beta_intercept_on_nonoriginal_actions+remaining_feature_slopes; both_original_corrections0; raw_MSE+offset_energy+slopes_energy+2offset_slopes+2raw_error_offset+2raw_error_slopes=full_MSE; keep_signed_bias_parts",
        "ablation": "offset_only_and_slopes_only_each_clipped_to_same_Stage282_training_service_radius; no_refit_recenter_rescale_penalty_or_threshold_change; old_uncut_and_clipped_predictions_and_all24_choices_reproduced",
        "selection": "same_raw_Linear9_observed_df_risk_wait_gate_candidates_and_ties; freeze_new_choices_before_evaluation; no_MC_query_label_in_new_offset_or_slopes_choices",
        "switches": "compare_each_service_choice_to_own_frozen_risk_controller; partition_unchanged_original_to_other_other_to_original_other_to_other; outcome_contributions_use_all_query_states_as_denominator_not_conditional_counts",
        "evaluation": "same_train_retained_excluded_full60_held_and_prefix54_held; assigned_opposite16_and_fixed8+8; primary_mixed_excluded_opposite_and_held_opposite; all_views_banks_reported; no_winner_selection",
        "controls": "all_old_cell_and_group_service_metrics_and_policy_outcomes_replay_Stage282; exact_uncut_MSE_and_switch_partition_identities; clipped_ablations_not_assumed_additive",
        "budget": {"analysis_tasks": 1, "cached_cells": 32, "cached_queries": 2048, "model_state_exposures": 84480,
            "new_ablation_decisions": 1520640, "new_fits": 0, "new_native_episodes": 0, "actor_updates": 0},
        "storage": "full_component_predictions_choices_and_query_ledgers_remote_only; compact_scalar_report_pull",
        "boundary": "frozen_parameter_ablation_not_unique_causal_feature_attribution_or_new_training; reused_development_single_intervention_not_seed_CI_or_end_to_end_HRL; MC_benchmark_not_deployable; Stage253_248_169_closed; defaults_unchanged"}


def analyze_cell(directory, key, views, panels):
    here = lambda run: ROOT/"results"/run/"report_raw"/("_".join(map(str, key)))
    models = {m["index"]: m for m in json.loads((here(SOURCE_RUN)/"models.json").read_text())}
    queries = json.loads((here(SOURCE_RUN)/"predictions.json").read_text())
    reference = source.geometry.source.read_cell(cache.cached.fixed.contract()["reference_run"], key, "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    rows = {"train": train, "held": held}
    data = {s: cache.learning.base.causal_data(rs, reference) for s, rs in rows.items()}
    predictions = []
    with np.load(here(source.RISK_RUN)/"predictions.npz") as saved:
        for q in queries:
            m = models[q["index"]]
            if (q["bank"], q["removed"]) != (m["bank"], m["removed"]) or q["radius"] != m["radius"]:
                raise ValueError("Changed frozen service context")
            ids, label = np.asarray(q["ids"]), q["label"]
            d = {k: v[ids] for k, v in data[label].items()}
            eligible = (d["raw_logits"] <= 0)&(np.arange(6)[None] != d["originals"][:, None])
            if not np.array_equal(q["eligible"], eligible):
                raise ValueError("Changed feasibility")
            risks = {a: saved[f'{q["index"]}_{q["scope"]}_{a}'].copy() for a in learning.CONTROLS}
            services = {a: np.asarray(v) for a, v in q["services"].items()}
            services, clips, parts = learning.ablate(m, services, d["originals"])
            choices = learning.previous.decisions(d, risks, services)
            if any(not np.array_equal(choices[a], q["choices"][a]) for a in learning.previous.ARMS):
                raise ValueError("Stage282 frozen choice replay failed")
            predictions.append({**q, "services": services, "choices": choices, "parts": parts,
                "clips": {**{a: np.asarray(v) for a, v in q["clips"].items()}, **clips}})
    if len(models) != 22 or len(predictions) != 64:
        raise ValueError("Incomplete Stage283 model/query coverage")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"predictions.json", predictions)
    print(f"SERVICE_OFFSET_CHOICES_FROZEN {key}", flush=True)
    previous = json.loads((here(SOURCE_RUN)/"outcomes.json").read_text())
    records, error, part_error, switch_error = [], 0., 0., 0.
    for qi, q in enumerate(predictions):
        label, ids = q["label"], np.asarray(q["ids"])
        originals = data[label]["originals"][ids]
        eligible = np.asarray(q["eligible"])
        for fi, (future, bank) in enumerate((("same", q["bank"]), ("opposite", 1-q["bank"]))):
            draws, _ = cache.sampling.learning.draw_assignment(bank, "expanded")
            values = cache.learning.base.training_draws([rows[label][i] for i in ids], panels, draws)
            outcomes = policy.outcomes(values, q["choices"])
            metrics = learning.previous.metrics(q["services"], q["clips"], q["services"]["raw"], values, originals, eligible, q["radius"])
            parts, residual = learning.error_parts(values, originals, q["services"]["raw"], eligible, q["parts"])
            part_error = max(part_error, residual)
            switches = learning.switch_stats(values, {a: q["choices"][a] for a in learning.REPORT_ARMS}, originals) if q["scope"]+"_"+future in PRIMARY else {}
            old = previous[2*qi+fi]
            if (old["bank"], old["scope"]) != (q["bank"], q["scope"]+"_"+future):
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
            for k in learning.MAPS:
                error = max(error, abs(parts["all16"][k]["sums"]["full_MSE"]-old["metrics"][k]["squared_error_sum"]))
            records.append({"bank": q["bank"], "scope": q["scope"]+"_"+future,
                "metrics": metrics, "outcomes": outcomes, "parts": parts, "switches": switches})
    if error > 1e-10 or switch_error > 1e-10:
        raise ValueError("Stage282 replay or switch partition failed")
    write_json(directory/"outcomes.json", records)
    return {"key": list(key), "records": records, "model_state_exposures": sum(len(q["ids"]) for q in queries),
        "replay_error": error, "part_error": part_error, "switch_error": switch_error}


def report(cells, before):
    if len(cells) != 32 or {tuple(c["key"]) for c in cells} != set(KEYS) or sum(c["model_state_exposures"] for c in cells) != 84480:
        raise ValueError("Incomplete Stage283 coverage")
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
                metrics = {a: learning.previous.base.summarize([r["metrics"][a] for r in rs], True) for a in learning.SERVICES}
                outcomes = {h: {a: policy.summarize([r["outcomes"][h][a] for r in rs]) for a in learning.ARMS} for h in policy.DRAW_GROUPS}
                parts = {h: {k: learning.summarize_parts([r["parts"][h][k] for r in rs]) for k in learning.MAPS} for h in policy.DRAW_GROUPS}
                for a in learning.previous.SERVICES:
                    for f, value in metrics[a].items():
                        if value is not None:
                            error = max(error, abs(value-old["service_metrics"][a][f]))
                for h, arms in outcomes.items():
                    if h in old["outcomes"]:
                        for a in learning.previous.ARMS:
                            error = max(error, *(abs(v-old["outcomes"][h][a][f]) for f, v in arms[a].items()))
                for k in learning.MAPS:
                    error = max(error, abs(parts["all16"][k]["full_MSE"]-metrics[k]["MSE"]))
                halves = policy.DRAW_GROUPS if g == "mixed_pooled" and scope in (*PRIMARY, "matched_held_opposite") else ("all16",)
                result = {"service_metrics": {a: {f: m[f] for f in ("MSE", "bias", "clipped_action_fraction")} for a, m in metrics.items()},
                    "outcomes": {h: {a: outcomes[h][a] for a in learning.REPORT_ARMS} for h in halves},
                    "error_parts": {h: parts[h] for h in halves}}
                if g == "mixed_pooled" and scope in PRIMARY:
                    result["switches"] = {a: learning.summarize_switches([r["switches"]["all16"][a] for r in rs]) for a in learning.REPORT_ARMS}
                output[g][label][scope] = result
    if error > 1e-10:
        raise ValueError("Stage282 aggregate replay failed")
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "reproduction": {"cell_replay_error": max(c["replay_error"] for c in cells), "group_replay_error": error,
            "component_identity_error": max(c["part_error"] for c in cells), "switch_partition_error": max(c["switch_error"] for c in cells)},
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
        cells.append(analyze_cell(directory/("_".join(map(str, key))), key, views, panels))
        print(f"SERVICE_OFFSET_CELL_COMPLETE {key}", flush=True)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    write_json(directory/"cells.json", [{k: v for k, v in c.items() if k != "records"} for c in cells])
    write_json(path, report(cells, before))
    print("SERVICE_OFFSET_REPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
