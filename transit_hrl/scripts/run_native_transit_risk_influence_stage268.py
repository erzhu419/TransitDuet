#!/usr/bin/env python3
"""Cached fleet-event ANOVA and unique future influence, no fitting or simulation."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_coupled_noise_stage267 as source
from freq_hrl.domains.transit import native_risk_influence as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_risk_influence_stage268_v1"
SOURCE_RUN = "native_transit_coupled_noise_stage267_20261016_r2"
cache, KEYS, VIEWS, PRIMARY = source.cache, source.KEYS, source.VIEWS, source.PRIMARY
CATEGORIES, REPORT_REGIONS, FOCUS_PREFIX = source.CATEGORIES, source.REPORT_REGIONS, source.FOCUS_PREFIX


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": source.EXPERIMENT_PROTOCOL,
        "cohort": source.contract()["cohort"], "primary": list(PRIMARY), "categories": list(CATEGORIES),
        "frozen": source.contract()["frozen"], "noise": source.contract()["noise"], "independence": source.contract()["independence"],
        "event": "any_eligible_action_pair_has_physical_fleet_margin_above_1e-12_existing_label_zero_tolerance; not_common_fleet_cost_or_counting_action_pairs_as_independent_events",
        "scatter": "state_and_bank_centered_coupled_risk_response; exact_sample_ANOVA_within_event+within_none+between_event_none_means; same16_mean_difference_normalization; preserve_endpoint_and_operator_change_signed_cross_terms_in_each_part",
        "influence": "delete_one_cached_state_future_label_from_its_bank_mean_and_unbiased_mean_variance_at_fixed_operators; exact_algebra_not_policy_refit; aggregate_energy_by_unique_state_key_and_actual_draw_id_across_reused_model_exposures; not_variance_of_average_models",
        "controls": "reproduce_Stage267_full_query_risk_variance_and_observed_squared_response_for_all5_terms_all3_label_populations_all3_query_regions_and_all_group_bank_categories; matched_exposures; ANOVA_and_both_cross_identities",
        "focus_prefix": list(FOCUS_PREFIX), "focus_basis": source.contract()["focus_basis"],
        "budget": source.contract()["budget"], "storage": "label_and_model_caches_full_ledgers_per_future_influence_NPZ_and_full_aggregate_server_only; pull_compact_scalar_report",
        "influence_array_axes": ["cached_piece_order", list(REPORT_REGIONS), list(learning.POPULATIONS), "train_state_order", "draw_0_to_31"],
        "boundary": "conditional_plugin_scatter_and_frozen_mean_influence_not_unseen_tail_probability_refit_accuracy_performance_seed_CI_or_fitting_safety; reused_futures_are_deduplicated_not_new_replications; no_selection_or_default_change; Stage253_248_169_closed"}


def analyze_cell(key, views, panels, before):
    read = lambda run, f: source.source.source.read_cell(run, key, f)
    models, calibrations, actions = (read(source.source.source.FIT_RUN, f) for f in ("models.json", "calibrations.json", "actions.json"))
    cs = {c["bank"]: c for c in calibrations}
    full_models = read(cache.cached.SOURCE_RUN, "models.json")
    prior = read(cache.cached.information.SOURCE_RUN, "models.json")["full"]["all_mse"]
    reference = read(cache.cached.fixed.contract()["reference_run"], "models.json")["reference"]
    train, held = cache.cached.training.cohort.split_rows(views[key[0]], key[1])
    data = {s: cache.learning.base.causal_data(rows, reference) for s, rows in (("train60", train), ("held60", held))}
    geometry = cache.learning.deployment_geometry(data["train60"])
    x = {s: cache.learning.joint.relative(d["x"][..., :198], d["originals"]) for s, d in data.items()}
    paired, draws_all, label_error = [], [], 0.
    for bank in (0, 1):
        draws, other = cache.sampling.learning.draw_assignment(bank, "expanded")
        if cs[bank]["training_draws"]!=draws or len(draws)!=16 or set(draws)&set(other): raise ValueError("Changed future banks")
        values = cache.learning.base.training_draws(train, panels, draws)
        residual = source.source.source.learning.previous.previous.residual_target(values, data["train60"], prior["scale"])
        label_error = max(label_error, float(np.max(np.abs(residual*np.asarray(cs[bank]["reliability"])-np.asarray(cs[bank]["target"])))))
        v = values[..., (2, 1)];paired.append(v-v[np.arange(len(train)), :, data["train60"]["originals"]][:, :, None]);draws_all.extend(draws)
    if label_error>1e-12 or draws_all!=list(range(32)): raise ValueError("Changed cached labels or draw order")
    paired = np.concatenate(paired, axis=1)
    events = learning.fleet_events(paired, data["train60"]["raw_logits"]<=0)
    states = [list(cache.sampling.old_future.learning.state_key(r)) for r in train]
    fits = {(m["bank"], tuple(m["held_group"])): m for m in models}
    lookup = {(a["bank"], tuple(a["removed"]), a["variant"], a["scope"]): a for a in actions if a["variant"] in PRIMARY}
    records, maximum = [], 0.
    for old_record in before["pieces"]:
        bank, group, variant = old_record["bank"], old_record["removed"], old_record["variant"]
        fitted = fits[bank, tuple(group)];a = lookup[bank, tuple(group), variant, "train60"]
        meta = {k: a[k] for k in ("cell", "bank", "removed", "variant", "category", "old_key", "old_key_new_gate")}
        if any(meta[k]!=old_record[k] for k in meta) or fitted["training_draws"]!=cs[bank]["training_draws"]: raise ValueError("Changed cached record")
        masks = source.source.source.source.diagnostic.regions(a["states"], "train60", group)
        retained = masks["retained"];indices = np.flatnonzero(retained)
        if (len(indices), int(masks["excluded"].sum()))!=(54, 6): raise ValueError("Changed fit population")
        active = a["old_key"] not in (None, "zero")
        record = {**meta, "active": active, "events": learning.event_counts(events, retained), "regions": {},
            "_states": states, "_events": events, "_retained": retained, "_influence": {}}
        if active:
            model = variant.removesuffix("_expanded8_pooled_safe");option = cache.learning.strength.CANDIDATES[a["old_key"]]
            level, alpha = option["level"], option["alpha"];basis, coordinates = geometry[level]
            radius = next(f["old_radius"] for f in fitted["fit"]["fits"] if f["level"]==level)
            scores = learning.risk_scores(x["train60"]@basis, cs[bank], paired, data["train60"]["raw_logits"]<=0,
                np.asarray(prior["weights"][level])@coordinates.T,
                np.asarray(full_models[str(bank)][model]["weights"][level])@coordinates.T,
                np.asarray(fitted["models"][model]["weights"][level])@coordinates.T, indices, model, radius, level, prior["scale"])
        else:
            scores = {t: np.zeros((60, 32, 1)) for t in learning.TERMS}
        parts = {t: learning.scatter_parts(s, events) for t, s in scores.items()}
        for scope, d in data.items():
            old = lookup[bank, tuple(group), variant, scope]
            state_order = [list(cache.sampling.old_future.learning.state_key(r)) for r in (train if scope=="train60" else held)]
            if old["states"]!=state_order or old["raw"]!=d["raw_choices"].tolist(): raise ValueError("Changed states/reference")
            feasible = (d["raw_logits"]<=0)&(np.arange(6)[None, :]!=d["originals"][:, None])
            for region, mask in source.source.source.source.diagnostic.regions(state_order, scope, group).items():
                if region not in REPORT_REGIONS or not mask.any(): continue
                n = int(feasible[mask].sum())
                design = (x[scope]@basis)[mask[:, None]&feasible]*(alpha*prior["scale"]) if active else np.zeros((n, 1))
                s, influence = learning.query_stats(design, scores, parts, events, retained, bank)
                previous = old_record["regions"][region]["terms"]
                for t in learning.TERMS:
                    if n!=previous[t]["feasible_action_cases"]: raise ValueError("Changed query denominator")
                    for pop, prefix in (("all", ""), ("excluded", "excluded_"), ("retained", "retained_")):
                        expected = previous[t]["bands"]["full"]
                        maximum = max(maximum, source.scalar_difference(s["populations"][pop]["variance"][t]["total"], expected["noise_squared_sum"][prefix+"risk"]),
                            source.scalar_difference(s["populations"][pop]["observed_squared_sum"][t], expected["observed_squared_sum"][prefix+"risk"]))
                record["regions"][region] = s;record["_influence"][region] = influence
        records.append(record)
    if len(models)!=20 or len(records)!=40 or maximum>1e-9: raise ValueError("Stage268 risk replay failed")
    print(f"RISK_INFLUENCE_CELL_COMPLETE {key}", flush=True)
    return {"key": list(key), "pieces": records, "target_max_abs_difference": label_error, "baseline_max_relative_difference": maximum}


def future_summary(pieces, region, population, variance, observed):
    keys = sorted({tuple(s) for p in pieces for s in p["_states"]});index = {s: i for i, s in enumerate(keys)}
    arrays = {k: np.zeros((len(keys), 32)) for k in ("energy", "variance_delta", "observed_squared_delta", "exposures", "event_exposures")}
    for p in pieces:
        if region not in p["_influence"]: continue
        ids = np.asarray([index[tuple(s)] for s in p["_states"]]);mask = np.ones(len(ids), bool) if population=="all" else p["_retained"] if population=="retained" else ~p["_retained"]
        for f in ("energy", "variance_delta", "observed_squared_delta"):
            np.add.at(arrays[f], ids, p["_influence"][region][population][f])
        np.add.at(arrays["exposures"], ids, np.broadcast_to(mask[:, None], (len(ids), 32)))
        np.add.at(arrays["event_exposures"], ids, p["_events"]*mask[:, None])
    energy = arrays["energy"].ravel();order = np.argsort(-energy);vd = arrays["variance_delta"].ravel();od = arrays["observed_squared_delta"].ravel()
    if abs(float(energy.sum())-variance)>1e-9*max(1., variance): raise ValueError("Unique future energy failed replay")
    rows = []
    for j in order[:4]:
        if energy[j]<=0: continue
        i, draw = divmod(int(j), 32)
        rows.append({"state": list(keys[i]), "draw": draw, "energy": float(energy[j]), "variance_delta": float(vd[j]),
            "observed_squared_delta": float(od[j]), "exposures": int(arrays["exposures"].ravel()[j]), "event_exposures": int(arrays["event_exposures"].ravel()[j])})
    return {"unique_state_future_keys": int(np.count_nonzero(arrays["exposures"])),
        "unique_keys_with_eligible_event": int(np.count_nonzero(arrays["event_exposures"])),
        "top_energy_fraction": {str(k): float(energy[order[:k]].sum()/variance) if variance else None for k in (1, 4, 16)},
        "max_single_future_variance_reduction_fraction": max(0., -float(vd.min()))/variance if variance else None,
        "max_single_future_observed_squared_reduction_fraction": max(0., -float(od.min()))/observed if observed else None,
        "top_futures": rows}


def summarize(pieces):
    out = {"prefix_model_exposures": len(pieces), "active_prefix_models": sum(p["active"] for p in pieces), "regions": {}}
    out["events"] = {pop: {f: sum(p["events"][pop][f] for p in pieces) for f in ("state_future_exposures", "event_exposures", "state_bank_exposures", "zero_event_state_banks", "one_event_state_banks", "all_event_state_banks")} for pop in learning.POPULATIONS}
    out["events_active_models"] = {pop: {f: sum(p["events"][pop][f] for p in pieces if p["active"]) for f in out["events"][pop]} for pop in learning.POPULATIONS}
    for region in REPORT_REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        n = sum(x["feasible_action_cases"] for x in xs);populations = {}
        for pop in learning.POPULATIONS:
            variance = {t: {part: sum(x["populations"][pop]["variance"][t][part] for x in xs) for part in ("total", *learning.PARTS)} for t in learning.TERMS}
            observed = {t: sum(x["populations"][pop]["observed_squared_sum"][t] for x in xs) for t in learning.TERMS}
            v, o = variance["coupled"]["total"], observed["coupled"]
            populations[pop] = {"variance": variance, "observed_squared_sum": observed,
                "coupled_noise_RMS": float(np.sqrt(v/n)) if n else None,
                "coupled_observed_over_noise_RMS": float(np.sqrt(o/v)) if v else None,
                "coupled_scatter_fractions": {part: variance["coupled"][part]/v if v else None for part in learning.PARTS},
                "cross": {t: {part: sum(x["populations"][pop]["cross"][t][part] for x in xs) for part in ("total", *learning.PARTS)} for t in learning.coupled.PAIRS},
                "centered_energy_on_event_draws": sum(x["populations"][pop]["centered_energy_on_event_draws"] for x in xs),
                "centered_energy_on_none_draws": sum(x["populations"][pop]["centered_energy_on_none_draws"] for x in xs),
                "future_influence": future_summary(pieces, region, pop, v, o)}
        out["regions"][region] = {"feasible_action_exposures": n, "populations": populations}
    return out


def report(records, expected):
    groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0]==v for v in VIEWS}}
    output, maximum = {}, 0.
    for group, predicate in groups.items():
        ps = [p for r in records if predicate(r["key"]) for p in r["pieces"]];output[group] = {}
        for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1)):
            output[group][label] = {}
            for variant in PRIMARY:
                output[group][label][variant] = {}
                for cat in CATEGORIES:
                    now = summarize([p for p in ps if p["variant"]==variant and (bank is None or p["bank"]==bank) and source.source.source.in_category(p, cat)])
                    before = expected["groups"][group][label][variant][cat]
                    if now["prefix_model_exposures"]!=before["prefix_model_exposures"]: raise ValueError("Changed aggregate exposure")
                    for reg, old in before["regions"].items():
                        actual = now["regions"][reg]
                        if actual["feasible_action_exposures"]!=old["feasible_action_exposures"]: raise ValueError("Changed query exposure")
                        for pop, prefix in (("all", ""), ("excluded", "excluded_"), ("retained", "retained_")):
                            for t in learning.TERMS:
                                maximum = max(maximum, source.scalar_difference(actual["populations"][pop]["variance"][t]["total"], old["terms"][t]["bands"]["full"]["noise_squared_sum"][prefix+"risk"]))
                                if pop=="all":
                                    maximum = max(maximum, source.scalar_difference(actual["populations"][pop]["observed_squared_sum"][t], old["terms"][t]["bands"]["full"]["observed_squared_sum"]["risk"]))
                    output[group][label][variant][cat] = now
    ps = [p for r in records for p in r["pieces"]]
    if len(records)!=32 or len(ps)!=1280 or maximum>1e-9: raise ValueError("Incomplete Stage268 replay")
    focus = [p for p in ps if p["cell"][0].startswith("mixed") and tuple(p["removed"])==FOCUS_PREFIX and source.source.source.in_category(p, "removed")]
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "groups": output,
        "focus_mixed_removed": {v: summarize([p for p in focus if p["variant"]==v]) for v in PRIMARY},
        "reproduction": {"cached_cells": len(records), "prefix_model_records": len(ps), "baseline_cell_max_relative_difference": max(r["baseline_max_relative_difference"] for r in records),
            "baseline_group_max_relative_difference": maximum, "target_max_abs_difference": max(r["target_max_abs_difference"] for r in records),
            "query_decomposition_residual": max(s["decomposition_residual"] for p in ps for s in p["regions"].values())},
        "default_changed": False, "native_deployment_eligible": False}


def compact_report(payload):
    def summary(s):
        return {**s, "regions": {r: {**a, "populations": {p: {**v, "future_influence": {k: x for k, x in v["future_influence"].items() if k!="top_futures"}} for p, v in a["populations"].items()}} for r, a in s["regions"].items()}}
    return {**payload, "groups": {g: {b: {v: {c: summary(s) for c, s in cats.items()} for v, cats in vs.items()} for b, vs in banks.items()} for g, banks in payload["groups"].items()},
        "focus_mixed_removed": payload["focus_mixed_removed"],
        "mixed_removed_future_leaders": {v: payload["groups"]["mixed_pooled"]["pooled"][v]["removed"]["regions"]["held"]["populations"]["all"]["future_influence"]["top_futures"] for v in PRIMARY}}


def save_cell_arrays(directory, cell):
    arrays = {name: np.asarray([[[p["_influence"][r][pop][name] for pop in learning.POPULATIONS] for r in REPORT_REGIONS] for p in cell["pieces"]])
        for name in ("energy", "variance_delta", "observed_squared_delta")}
    arrays.update(events=cell["pieces"][0]["_events"], state_keys=np.asarray(cell["pieces"][0]["_states"]), retained=np.asarray([p["_retained"] for p in cell["pieces"]]))
    np.savez_compressed(directory/("influence_"+"_".join(map(str, cell["key"]))+".npz"), **arrays)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument("--run-name", required=True);a = p.parse_args()
    data, _, registered, _ = cache.cached.training.cohort.source.source.load_data()
    views = cache.cached.training.cohort.source.matched_views(data["training_data"], registered)
    panels = cache.sampling.read_panels(cache.cached.previous.DATA_RUN)
    before = json.loads((ROOT/"results"/SOURCE_RUN/"report_raw/cells.json").read_text());lookup = {tuple(r["key"]): r for r in before}
    if len(before)!=32 or set(lookup)!=set(KEYS): raise ValueError("Incomplete Stage267 ledger")
    path = ROOT/"results"/a.run_name/"report/result.json";directory = raw_directory(path)
    records = []
    for key in KEYS:
        cell = analyze_cell(key, views, panels, lookup[key]);save_cell_arrays(directory, cell);records.append(cell)
    write_json(directory/"cells.json", [{**r, "pieces": [{k: v for k, v in p.items() if k!="_influence"} for p in r["pieces"]]} for r in records])
    expected = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text());payload = report(records, expected)
    write_json(directory/"aggregate.json", payload);write_json(path, compact_report(payload))
    print("RISK_INFLUENCE_REPORT_COMPLETE", flush=True)


if __name__ == "__main__": main()
