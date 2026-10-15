#!/usr/bin/env python3
"""Confirm frozen Stage247 policies on 64 independent conditional futures."""

import argparse
from functools import partial
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_balanced_critic_stage247 as previous
from scripts import run_native_transit_frozen_future_stage241 as old_future
from freq_hrl.domains.transit import native_frozen_confirmation as learning
from freq_hrl.domains.transit.native_continuation_value import NativeContinuationValueAudit
from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_frozen_confirmation_stage248_v1"
SOURCE_RUN = "native_transit_balanced_critic_stage247_development_20261015_r1"
coverage, source = old_future.coverage, old_future.coverage.noise.source
ROOTS, SCENARIOS, KEYS = old_future.ROOTS, old_future.SCENARIOS, previous.KEYS
DRAWS = tuple(range(64))
VARIANTS, PRIMARY = learning.VARIANTS, learning.PRIMARY


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "variants": list(VARIANTS), "primary": PRIMARY, "scope": "full_model_held60_actions_only",
        "cohort": "same240_development_states40_prefixes32_cells_all4_views_reserved_Stage169_cohorts_closed",
        "frozen": "Stage247_actions_and_selectors_read_verbatim_no_inference_refit_or_reselection",
        "draws": list(DRAWS), "halves": [list(range(32)), list(range(32, 64))],
        "future_seed": "1480000000+index_in_Stage195_states*64+draw_unique_disjoint_from_Stage191_195_212_241",
        "replay": "union_of_actions_on_any_frozen_policy_divergence_plus_original_identity_CRN_original_greedy_suffix",
        "zero_states": "same_action_contrasts_exact_zero_included_in_full_cohort_denominators_no_absolute_outcome_imputation",
        "controls": "one_exact_original_source_one_resampled_observer_and_original_candidate_identity_per_draw",
        "qualification": "unchanged_prefix_physics_RNG_causal_context_frozen_weights_window_ledger_and61380_tick_episodes",
        "primary_test": "mixed_pooled_vs_raw_cost_and_wait_normal_MC_upper95_below_zero_and_both32_halves_negative",
        "secondary": "paired_vs_all_MSE_and_pair_GLS_all4_views_root_and_scenario_diagnostics_not_new_selection",
        "uncertainty": "fixed_prefix_conditional_MC_normal_intervals_shared_state_draw_counted_once_not_seed_CI_or_tail_guarantee",
        "budget": {"draws_per_active_state": 64, "episodes_per_draw": "2+sealed_candidate_union_including_original",
            "new_prefix_scenes": 0, "new_raw_reference_fits": 0, "new_residual_fits": 0, "training_updates": 0},
        "storage": "server_only_seal_rollouts_and_sparse_labels_sync_scalar_seal_preflight_and_final_reports",
        "boundary": "independent_future_replication_on_development_prefixes_one_forced_action_not_full_end_to_end_HRL"}


def continuation_seed(root, scenario, replica, probe, draw):
    return 1480000000+coverage.states().index((root, scenario, replica, probe))*64+draw


def seal_path(run):
    return ROOT/"results"/run/"seal_raw/choices.json"


def draw_path(run, *key):
    return coverage.draw_path(run, *key)


def seal(run):
    bank = json.loads(coverage.source.bank_path(coverage.SOURCE_RUN).read_text())
    rows = {learning.state_key(r): r for r in bank["training_data"]}
    cells = []
    for key in KEYS:
        raw = raw_directory(previous.result_path(SOURCE_RUN, *key))
        previous.qualify(json.loads((raw/"aggregate.json").read_text()))
        blocks = [b for b in json.loads((raw/"actions.json").read_text()) if b["scope"] == "held60"]
        if len(blocks) != 1 or len(blocks[0]["states"]) != 60: raise ValueError("Changed frozen held-out scope")
        b = blocks[0]
        cells.append({"key": list(key), "blocks": [{"states": [{**dict(zip(("root", "scene_seed", "probe_index"), st)),
            "scenario": rows[tuple(st)]["scenario"]} for st in b["states"]],
            "choices": {v: b["choices"][v] for v in VARIANTS}}]})
    lookup = {(r, source.scene_seed(r, s, i), p): (r, s, i, p) for r, s, i, p in coverage.states()}
    keys = {learning.state_key(r) for c in cells for b in c["blocks"] for r in b["states"]}
    if keys != lookup.keys(): raise ValueError("Frozen choices changed the existing 240-state cohort")
    active = [{"state": list(st), "key": list(lookup[st]),
        "candidates": sorted(set(cs)|{rows[st]["original_choice"]})} for st, cs in learning.action_union(cells).items()]
    # The old eight outcomes only check that the sealed actions are the published ones.
    _, old = previous.panel_cache()
    maximum = 0.
    prior = json.loads((ROOT/"results"/SOURCE_RUN/"report/result.json").read_text())
    for view in previous.VIEWS:
        actual = learning.summarize([c for c in cells if c["key"][0] == view], old)
        expected = prior["views"][view]["held60"]["fresh8"]["all"]["variants"]
        for v in VARIANTS:
            for f, name in (("cost", "delta_cost"), ("wait_min", "delta_wait_min")):
                maximum = max(maximum, abs(actual["variants"][v]["fields"][f]["mean"]-expected[v][name]))
    if maximum > 1e-12: raise ValueError("Sealed actions changed Stage247 published outcomes")
    for root in ROOTS:
        _, checkpoint = source.native.load_source(root)
        if not checkpoint.is_dir(): raise ValueError("Missing native source checkpoint")
    for item in active:
        if not coverage.noise.source_path(ROOT/"results"/coverage.noise.COLLECTION_RUN, *item["key"]).is_file():
            raise ValueError("Missing original native source reproduction")
    frozen = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "cells": cells, "active": active}
    write_json(seal_path(run), frozen)
    write_json(ROOT/"results"/run/"seal/result.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "coverage_qualified": True, "active": active, "fixed_states": len(keys), "frozen_cells": len(cells),
        "active_states": len(active), "reproduction_max_abs_difference": maximum,
        "collection_tasks": len(active)*64, "native_episodes": sum(2+len(a["candidates"]) for a in active)*64,
        "exhaustive_native_episodes_avoided": 240*64*8-sum(2+len(a["candidates"]) for a in active)*64,
        "training_updates": 0})
    print("FROZEN_CONFIRMATION_ACTIONS_SEALED", flush=True)


def load_seal(run):
    frozen = json.loads(seal_path(run).read_text())
    if frozen["protocol"] != EXPERIMENT_PROTOCOL or frozen["contract"] != contract(): raise ValueError("Changed sealed future protocol")
    return frozen


def cells(sealed, *, preflight=False):
    active = [tuple(a["key"]) for a in sealed["active"]]
    if preflight:
        active = [next(k for k in active if k[:2] == (r, s)) for r in ROOTS for s in SCENARIOS
            if any(k[:2] == (r, s) for k in active)]
    return [(*key, d) for key in active for d in ((0,) if preflight else DRAWS)]


def labels_from_rollouts(baseline, branches, historical_peak):
    base_cost = baseline["service_cost_restricted"]
    base_fleet = max(baseline["peak_fleet"]-12, 0)**2/12
    floor = max(historical_peak-12, 0)**2/12
    result = []
    for b in branches:
        cost, fleet = b["service_cost_restricted"]-base_cost, max(b["peak_fleet"]-12, 0)**2/12
        result.append({"candidate": b["probe"]["selected"], "actual_cost_delta": cost,
            "fleet_cost_delta": fleet-base_fleet, "nonfleet_cost_delta": cost-(fleet-base_fleet),
            "wait_delta_min": b["restricted_wait_horizon_min"]-baseline["restricted_wait_horizon_min"],
            "physical_return_delta": -100*cost, "episode_reward_delta": b["ep_reward"]-baseline["ep_reward"],
            "future_fleet_cost": fleet-floor})
    return result


def replay_draw(key, item, seed, output):
    """Replay a sealed action union with the native source and identity controls."""
    started = time.monotonic()
    root, scenario, replica, probe, draw = key
    bank = json.loads(coverage.source.bank_path(coverage.SOURCE_RUN).read_text())
    registered = next(r for r in bank["training_data"] if learning.state_key(r) == tuple(item["state"]))
    model = json.loads((ROOT/"results"/coverage.noise.COLLECTION_RUN/"inputs/models.json").read_text())[str(root)]
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(source.native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = source.native.NativeRoutingTracker
    torch.set_num_threads(1)
    source.shared.source.incumbent.source.distributional.validate_model(model)
    _, checkpoint = source.native.load_source(root)
    scene, raw = source.scene_seed(root, scenario, replica), raw_directory(output)
    original, original_plan = source.shared.episode(root, scenario, scene, raw/"original", checkpoint, model,
        probe, plan_class=NativeTimetableContextAudit)
    prior = json.loads(coverage.noise.source_path(ROOT/"results"/coverage.noise.COLLECTION_RUN, *key[:4]).read_text())
    source.qualify_cell(prior)
    if ({k: original[k] for k in source.OUTCOME_KEYS} != prior["observer"]
            or source.shared.policy_trace(original["supply"]) != prior["policy_trace"]):
        raise ValueError("Original replay changed its existing native source")
    context = next(c for c in original_plan.contexts if c["index"] == probe)
    if (original["probe"]["candidates"] != registered["candidates"]
            or original["probe"]["native_state_v2"] != registered["native_state_v2"]
            or {k: context[k] for k in source.context_source.OCCUPANCY_KEYS} != registered["occupancy_context"]
            or context["timetable_context"] != registered["timetable_context"]):
        raise ValueError("Original replay changed frozen causal inputs")
    print(f"FROZEN_CONFIRMATION_SOURCE_QUALIFIED {key}", flush=True)
    plan_class = partial(NativeContinuationValueAudit, continuation_seed=seed)
    baseline, plan = source.shared.episode(root, scenario, scene, raw/"observer", checkpoint, model,
        probe, reference=original_plan, plan_class=plan_class)
    source.qualify_observer(baseline)
    branches = []
    order = [registered["original_choice"], *[c for c in item["candidates"] if c != registered["original_choice"]]]
    for candidate in order:
        row, branch_plan = source.shared.episode(root, scenario, scene, raw/f"candidate_{candidate}", checkpoint,
            model, probe, candidate=candidate, reference=plan, plan_class=plan_class)
        source.shared.qualify(row, baseline, candidate=candidate)
        if branch_plan.env.scenario_tape.seed != seed or not branch_plan.env.scenario_tape._streams:
            raise ValueError("Native branch did not consume its registered future tape")
        row["policy_trace"] = source.shared.policy_trace(row.pop("supply"))
        branches.append(row)
        print(f"FROZEN_CONFIRMATION_DRAW {key} candidate={candidate}", flush=True)
    branches.sort(key=lambda b: b["probe"]["selected"])
    baseline["policy_trace"] = source.shared.policy_trace(baseline["supply"])
    source.validate_continuations(baseline, branches, candidates=item["candidates"])
    n = 2+len(branches)
    result = {"key": list(key), "software_qualified": True,
        "continuation_seed": seed, "original_choice": registered["original_choice"], "candidates": item["candidates"],
        "labels": labels_from_rollouts(baseline, branches, context["historical_peak"]),
        "original_source_reproductions": 1, "exact_resampled_identities": 1, "prefix_state_rng_matches": n-1,
        "frozen_weight_qualified_episodes": n, "window_ledger_qualified_episodes": n, "continuation_qualified": True,
        "native_clock_ticks": n*61380, "training_updates": 0,
        "wall_seconds": time.monotonic()-started, "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.}
    qualify_physical_draw(result, key, item, seed)
    write_json(raw/"paired_rollouts.json", {"original": original, "observer": baseline, "branches": branches,
        "context": context, "snapshot": original_plan.snapshot})
    return result


def collect(run, key):
    frozen = load_seal(run)
    item = next(a for a in frozen["active"] if tuple(a["key"]) == key[:4])
    output = draw_path(run, *key)
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        **replay_draw(key, item, continuation_seed(*key), output)}
    qualify_draw(result, key, item)
    raw = raw_directory(output)
    write_json(raw/"result.json", result)
    write_json(output, {k: v for k, v in result.items() if k != "labels"})
    print("FROZEN_CONFIRMATION_DRAW_COMPLETE", flush=True)


def qualify_draw(r, key, item):
    if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or key[-1] not in DRAWS:
        raise ValueError("Changed frozen replay protocol")
    qualify_physical_draw(r, key, item, continuation_seed(*key))


def qualify_physical_draw(r, key, item, seed):
    n = 2+len(item["candidates"])
    if not (tuple(r["key"]) == key and r["software_qualified"] and tuple(item["key"]) == key[:4]
            and r["continuation_seed"] == seed and r["candidates"] == item["candidates"]
            and r["original_source_reproductions"] == r["exact_resampled_identities"] == 1
            and r["prefix_state_rng_matches"] == n-1 and r["frozen_weight_qualified_episodes"] == n
            and r["window_ledger_qualified_episodes"] == n and r["continuation_qualified"]
            and r["native_clock_ticks"] == n*61380 and r["training_updates"] == 0):
        raise ValueError("Changed frozen replay coverage or physical budget")
    if [l["candidate"] for l in r["labels"]] != item["candidates"]: raise ValueError("Missing frozen candidate")
    labels = {l["candidate"]: l for l in r["labels"]}
    identity = labels[r["original_choice"]]
    for l in labels.values():
        if not (np.all(np.isfinite([l[f] for f in coverage.expected.LABEL_FIELDS])) and l["future_fleet_cost"] >= 0
                and np.isclose(l["actual_cost_delta"], l["fleet_cost_delta"]+l["nonfleet_cost_delta"], atol=1e-12, rtol=0)
                and np.isclose(l["physical_return_delta"], -100*l["actual_cost_delta"], atol=1e-12, rtol=0)
                and np.isclose(l["future_fleet_cost"]-identity["future_fleet_cost"], l["fleet_cost_delta"], atol=1e-12, rtol=0)):
            raise ValueError("Changed physical target units or fleet floor")
    if any(identity[f] != 0 for f in coverage.expected.LABEL_FIELDS if f != "future_fleet_cost"):
        raise ValueError("Original candidate failed exact identity")


def report(run, *, preflight=False):
    frozen = load_seal(run)
    by_state = {tuple(a["state"]): a for a in frozen["active"]}
    records = {}
    for key in cells(frozen, preflight=preflight):
        r = json.loads((raw_directory(draw_path(run, *key))/"result.json").read_text())
        item = next(a for a in frozen["active"] if tuple(a["key"]) == key[:4])
        qualify_draw(r, key, item)
        records[key] = r
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True,
        "draw_tasks": len(records), "native_episodes": sum(r["frozen_weight_qualified_episodes"] for r in records.values()),
        "wall_seconds_max": max(r["wall_seconds"] for r in records.values()),
        "peak_process_rss_mb_max": max(r["peak_process_rss_mb"] for r in records.values()), "training_updates": 0}
    if not preflight:
        panels = learning.sparse_panels(frozen["cells"], DRAWS,
            lambda state, draw: records[(*by_state[state]["key"], draw)]["labels"])
        def summaries(cs):
            return {name: {baseline: learning.summarize(cs, {k: {"values": p["values"][ds]} for k, p in panels.items()}, baseline=baseline)
                for baseline in ("raw", "all_mse_pooled_safe", "pair_gls_pooled_safe")}
                for name, ds in (("half0_32", slice(0, 32)), ("half1_32", slice(32, 64)), ("fresh64", slice(0, 64)))}
        mixed = [c for c in frozen["cells"] if c["key"][0].startswith("mixed")]
        result["mixed_pooled"] = summaries(mixed)
        result["views"] = {v: summaries([c for c in frozen["cells"] if c["key"][0] == v]) for v in previous.VIEWS}
        result["mixed_by_root"] = {str(r): summaries(learning.subset(mixed, r)) for r in ROOTS}
        result["mixed_by_scenario"] = {s: summaries(learning.subset(mixed, scenario=s)) for s in SCENARIOS}
        result["primary_confirmation"] = learning.confirm(result["mixed_pooled"]["fresh64"]["raw"]["variants"][PRIMARY],
            [result["mixed_pooled"][h]["raw"] for h in ("half0_32", "half1_32")])
        result["default_changed"] = result["native_deployment_eligible"] = False
    write_json(ROOT/"results"/run/("preflight_report" if preflight else "report")/"result.json", result)
    print("FROZEN_CONFIRMATION_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("seal", "collect", "preflight_report", "report"), required=True)
    p.add_argument("--seed", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--replica", type=int, choices=coverage.REPLICAS)
    p.add_argument("--probe", type=int, choices=coverage.PROBES)
    p.add_argument("--draw", type=int, choices=DRAWS)
    args = p.parse_args()
    if args.operation == "seal": seal(args.run_name)
    elif args.operation == "collect": collect(args.run_name, (args.seed, args.scenario, args.replica, args.probe, args.draw))
    else: report(args.run_name, preflight=args.operation == "preflight_report")


if __name__ == "__main__": main()
