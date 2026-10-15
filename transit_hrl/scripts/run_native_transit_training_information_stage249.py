#!/usr/bin/env python3
"""Collect independent training futures before a fixed-structure information test."""

import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_balanced_critic_stage247 as previous
from scripts import run_native_transit_frozen_future_stage241 as old_future
from freq_hrl.domains.transit import native_training_information as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_training_information_stage249_v1"
SOURCE_RUN = "native_transit_balanced_critic_stage247_development_20261015_r1"
coverage, source = old_future.coverage, old_future.coverage.noise.source
ROOTS, SCENARIOS, KEYS, VIEWS = old_future.ROOTS, old_future.SCENARIOS, previous.KEYS, previous.VIEWS
DRAWS, CHUNKS = tuple(range(32)), tuple(range(8))
training, result_path = previous.training, previous.result_path
VARIANTS = learning.VARIANTS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "cohort": "same240_states40_prefixes32_cells4_views_Stage169_reserved_cohorts_closed",
        "arms": list(learning.ARMS), "variants": list(VARIANTS),
        "primary_contrast": "expanded_pooled_safe_minus_matched_pooled_safe_no_arm_selection",
        "futures": "32_new_independent_futures_per_state_all6_actions_CRN_two16_future_banks_crossfit",
        "future_seed": "1490000000+Stage195_state_index*32+draw_disjoint_from_all_previous_training_and_confirmation",
        "training_counts": {"matched": {"inner": 2, "full": 4}, "expanded": {"inner": 8, "full": 16}},
        "nesting": "each_bank_two8_blocks_matched_first2_each_inner_full_union4_expanded8_each_inner_full16",
        "evaluation": "opposite16_bank_only_independent_per_fit_no_new_future_labels_in_OOF_selection",
        "selector": "Stage247_pooled_and_half_safe_old_A_validation2_curves_same17_candidates_per_arm",
        "frozen": "raw_references_198features_TSVD_ranks_A_scale_A_RMS_radius_cached_head_difference_model_capacity_and_loss",
        "fit": "same_state_balanced_GLS_noise_shrinkage_precision_reestimated_only_from_assigned_training_futures",
        "controls": "old_Stage247_state_GLS_models_and_selections_read_verbatim_no_raw_reference_refitting",
        "diagnostics": "train60_vs_held60_cost_wait_service_fleet_MSE_feasible_pair_sign_error_target_mean_error_and_fleet_events",
        "label_boundary": "never_read_Stage248_labels_or_reports_for_training_selection_or_evaluation_no_scenario_gate",
        "next": "freeze_all_arms_before_collecting_genuinely_new_prefix_control_outcomes",
        "budget": {"states": 240, "draws": 7680, "native_episodes": 61440, "native_clock_ticks": 61440*61380,
            "preflight_tasks": 10, "remaining_chunk_tasks": 1920, "draws_per_chunk": 4,
            "fit_tasks": 32, "new_raw_reference_fits": 0, "rl_actor_updates": 0, "new_prefix_scenes": 0},
        "storage": "all_native_rollouts_labels_panels_models_server_only_small_receipts_and_aggregate_pull",
        "boundary": "training_information_diagnostic_conditional_future_crossfit_not_new_prefix_confirmation_or_complete_HRL"}


def continuation_seed(root, scenario, replica, probe, draw):
    return 1490000000+coverage.states().index((root, scenario, replica, probe))*32+draw


def preflight_keys():
    return [(r, s, 1, 9, 0) for r in ROOTS for s in SCENARIOS]


def chunk_keys():
    return [(*st, c) for st in coverage.states() for c in CHUNKS]


def chunk_draws(key):
    *state, chunk = key
    return [(*state, d) for d in range(4*chunk, 4*(chunk+1)) if (*state, d) not in preflight_keys()]


def draw_path(run, *key):
    return coverage.draw_path(run, *key)


def chunk_path(run, key):
    r, s, i, p, c = key
    return ROOT/"results"/run/"chunks"/s/f"root_{r}"/f"replica_{i}"/f"probe_{p}"/f"chunk_{c}/receipt.json"


def panel_path(run):
    return ROOT/"results"/run/"panels_raw/panels.npz"


def qualify_draw(r):
    key = r["root"], r["scenario"], r["replica"], r["probe_index"], r["draw"]
    if key[:4] not in coverage.states() or key[-1] not in DRAWS or r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract():
        raise ValueError("Changed information-collection cohort")
    old_future.qualify_physical_draw(r)
    if not r["software_qualified"] or r["scene_seed"] != source.scene_seed(*key[:3]):
        raise ValueError("Changed original prefix scene")
    if r["continuation_seed"] != continuation_seed(*key): raise ValueError("Changed independent training seed")


def collect(run, key, row, model):
    started = time.monotonic()
    r, s, i, p, d = key
    output = draw_path(run, *key)
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "replica": i, "draw": d,
        **coverage.noise.replay_draw(r, s, i, p, continuation_seed(*key), row, model, output)}
    result.update(wall_seconds=time.monotonic()-started,
        peak_process_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
    qualify_draw(result)
    write_json(raw_directory(output)/"result.json", result)
    write_json(output, {k: v for k, v in result.items() if k not in ("labels", "existing_labels")})
    print(f"TRAINING_INFORMATION_DRAW_COMPLETE {key}", flush=True)


def collect_task(run, key, *, preflight=False):
    bank = json.loads(coverage.source.bank_path(coverage.SOURCE_RUN).read_text())
    root, scenario, replica, probe, _ = key
    scene = source.scene_seed(root, scenario, replica)
    row = next(r for r in bank["training_data"] if old_future.learning.state_key(r) == (root, scene, probe))
    model = json.loads((ROOT/"results"/coverage.noise.COLLECTION_RUN/"inputs/models.json").read_text())[str(root)]
    keys = [key] if preflight else chunk_draws(key)
    for k in keys: collect(run, k, row, model)
    if not preflight:
        write_json(chunk_path(run, key), {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
            "key": list(key), "draws": [k[-1] for k in keys], "software_qualified": True,
            "native_episodes": 8*len(keys), "training_updates": 0})


def prepare(run, *, preflight=False):
    keys = preflight_keys() if preflight else [(*st, d) for st in coverage.states() for d in DRAWS]
    rss, seconds, panels = [], [], {}
    for key in keys:
        r = json.loads((raw_directory(draw_path(run, *key))/"result.json").read_text())
        qualify_draw(r)
        st = old_future.learning.state_key(r)
        panels.setdefault(st, []).append([[l[f] for f in ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")] for l in r["labels"]])
        rss.append(r["peak_process_rss_mb"])
        seconds.append(r["wall_seconds"])
    if not preflight:
        path = panel_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        states = sorted(panels)
        values = np.asarray([panels[k] for k in states])
        if values.shape != (240, 32, 6, 4): raise ValueError("Dropped independent training futures")
        np.savez_compressed(path, keys=np.asarray(states), values=values)
    write_json(ROOT/"results"/run/("preflight_report" if preflight else "prepare")/"result.json",
        {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
            "draws": len(keys), "native_episodes": 8*len(keys), "native_clock_ticks": 8*len(keys)*61380,
            "wall_seconds_max": max(seconds), "peak_process_rss_mb_max": max(rss), "training_updates": 0})
    print("TRAINING_INFORMATION_PANELS_QUALIFIED", flush=True)


def read_panels(run):
    with np.load(panel_path(run)) as p:
        return {tuple(k): {"values": v} for k, v in zip(p["keys"].tolist(), p["values"])}


def fit_cell(run, key, views):
    load = lambda source_run, name: json.loads((raw_directory(result_path(source_run, *key))/name).read_text())
    old = load(SOURCE_RUN, "aggregate.json")
    previous.qualify(old)
    cached = load(SOURCE_RUN, "models.json")
    ledger = load(previous.SOURCE_RUN, "fits.json")
    old_blocks = {(tuple(b["held_group"]), b["half"]): b for b in ledger["blocks"]}
    reference = load(previous.contract()["reference_run"], "models.json")["reference"]
    refs = load(previous.contract()["refit_run"], "models.json")["references"]
    trials = load(previous.contract()["refit_run"], "result.json")["cell"]["trials"]
    lookup = {(tuple(t["removed_group"]), t["half"]): str(i) for i, t in enumerate(trials) if t["removed_group"] and t["half"] is not None}
    train, held = training.cohort.split_rows(views[key[0]], key[1])
    past, _ = previous.panel_cache()
    panels = read_panels(run)
    a, b = previous.PARTITIONS[key[2]][key[3]], previous.PARTITIONS[key[2]][1-key[3]]
    output, models_cache, actions = [], {}, []
    for bank in (0, 1):
        blocks = []
        for group, indices, validation in learning.previous.joint.binding.policy.prefix_folds(train):
            rows, val_rows = [train[i] for i in indices], [train[i] for i in validation]
            for half in (0, 1):
                index = lookup[tuple(group), half]
                data = learning.previous.base.causal_data(rows, refs[index])
                val = learning.previous.base.causal_data(val_rows, refs[index])
                paths = {"old": learning.previous.base.candidates(val, cached[index]["state_gls"])}
                for arm in ("matched", "expanded"):
                    ds, _ = learning.draw_assignment(bank, arm, half)
                    models, fit = learning.previous.fit_paths(rows, data,
                        learning.previous.base.training_draws(rows, panels, ds), cached[index], old_blocks[tuple(group), half]["fits"])
                    previous.qualify_fit(fit)
                    paths[arm] = learning.previous.base.candidates(val, models["state_gls"])
                _, _, validation_draws = previous.fixed.draw_assignment(a, b, half)
                truth = learning.previous.base.training_draws(val_rows, past, validation_draws)
                blocks.append({"held_group": list(group), "half": half, "scales": old_blocks[tuple(group), half]["scales"],
                    "curves": {arm: {c: learning.previous.base.curve(val, truth, p) for c, p in ps.items()} for arm, ps in paths.items()}})
        selected, summaries = learning.select(blocks)
        if selected["old"] != old["selection"]["state_gls"]: raise ValueError("Frozen Stage247 OOF selection changed")
        data = learning.previous.base.causal_data(train, reference)
        models = {"old": copy.deepcopy(cached["full"]["state_gls"])}
        full_fits = {}
        for arm in ("matched", "expanded"):
            ds, _ = learning.draw_assignment(bank, arm)
            fitted, fit = learning.previous.fit_paths(train, data,
                learning.previous.base.training_draws(train, panels, ds), cached["full"], ledger["full"])
            previous.qualify_fit(fit)
            models[arm], full_fits[arm] = fitted["state_gls"], fit
        models_cache[str(bank)] = models
        # Freeze all selected actions before reading this bank's evaluation outcomes.
        frozen = []
        for scope, rows in (("train60", train), ("held60", held)):
            d = data if scope == "train60" else learning.previous.base.causal_data(rows, reference)
            paths = {arm: learning.previous.base.candidates(d, m) for arm, m in models.items()}
            pred = learning.predict(paths, selected)
            actions.append({"bank": bank, "scope": scope, "states": [list(old_future.learning.state_key(r)) for r in rows],
                "choices": {v: p["chosen"].tolist() for v, p in pred.items()}})
            frozen.append((scope, rows, d, pred, paths))
        write_json(raw_directory(result_path(run, *key))/"actions.json", actions)
        matrix, diagnostic = {}, {}
        for scope, rows, d, pred, paths in frozen:
            _, evaluation = learning.draw_assignment(bank, "expanded")
            values = learning.previous.base.training_draws(rows, panels, evaluation)
            matrix[scope] = learning.previous.metrics(d, values, pred)
            diagnostic[scope] = {}
            for arm in ("matched", "expanded"):
                ds, _ = learning.draw_assignment(bank, arm)
                diagnostic[scope][arm] = learning.information_error(
                    learning.previous.base.training_draws(rows, panels, ds), values, d, pred)
                rank_error = learning.information_error(learning.previous.base.training_draws(rows, panels, ds),
                    values, d, learning.rank_predictions(paths[arm]))
                diagnostic[scope][arm]["unselected_rank_error"] = {k: rank_error[k] for k in
                    ("predictor_error", "predictor_error_on_training_bank")}
        output.append({"bank": bank, "selection": selected, "summaries": summaries, "matrix": matrix,
            "information_error": diagnostic, "full_fits": full_fits})
        print(f"TRAINING_INFORMATION_FIT_BANK_COMPLETE {key}/{bank}", flush=True)
    path = result_path(run, *key)
    write_json(raw_directory(path)/"models.json", models_cache)
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key), "coverage_qualified": True,
        "banks": output, "raw_reference_fits": 0, "native_episodes": 0}
    write_json(raw_directory(path)/"aggregate.json", result)
    write_json(path, {k: v for k, v in result.items() if k != "banks"})


def report(run):
    results = [json.loads((raw_directory(result_path(run, *k))/"aggregate.json").read_text()) for k in KEYS]
    for key, r in zip(KEYS, results):
        if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or tuple(r["key"]) != key or not r["coverage_qualified"]:
            raise ValueError("Incomplete information-fit matrix")
    def merge(xs):
        return {scope: learning.aggregate([bank["matrix"][scope] for r in xs for bank in r["banks"]])
            for scope in ("train60", "held60")}
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "mixed_pooled": merge([r for r in results if r["key"][0].startswith("mixed")]),
        "views": {v: merge([r for r in results if r["key"][0] == v]) for v in VIEWS},
        "information_error": [{"key": r["key"], "banks": [{"bank": b["bank"], "selection": b["selection"],
            "diagnostics": b["information_error"]} for b in r["banks"]]} for r in results],
        "raw_reference_fits": 0, "default_changed": False, "native_deployment_eligible": False,
        "next_step": "freeze_all_arms_then_new_prefix_transfer_test_before_performance_claim"}
    write_json(ROOT/"results"/run/"report/result.json", payload)
    print("TRAINING_INFORMATION_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("preflight", "collect", "preflight_report", "prepare", "fit", "report"), required=True)
    p.add_argument("--seed", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--replica", type=int, choices=coverage.REPLICAS)
    p.add_argument("--probe", type=int, choices=coverage.PROBES)
    p.add_argument("--chunk", type=int, choices=CHUNKS)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--fold", type=int, choices=(0, 1))
    p.add_argument("--partition", choices=previous.PARTITIONS)
    p.add_argument("--direction", type=int, choices=(0, 1))
    args = p.parse_args()
    if args.operation in ("preflight", "collect"):
        collect_task(args.run_name, (args.seed, args.scenario, args.replica, args.probe,
            0 if args.operation == "preflight" else args.chunk), preflight=args.operation == "preflight")
    elif args.operation in ("preflight_report", "prepare"): prepare(args.run_name, preflight=args.operation == "preflight_report")
    elif args.operation == "fit":
        bank, _, registered, _ = training.cohort.source.source.load_data()
        views = training.cohort.source.matched_views(bank["training_data"], registered)
        fit_cell(args.run_name, (args.view, args.fold, args.partition, args.direction), views)
    else: report(args.run_name)


if __name__ == "__main__": main()
