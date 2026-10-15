#!/usr/bin/env python3
"""Confirm all frozen Stage252 selectors with new shared conditional futures."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_selector_information_stage252 as previous
from scripts import run_native_transit_frozen_confirmation_stage248 as replay
from freq_hrl.domains.transit import native_selector_confirmation as learning
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_selector_confirmation_stage253_v1"
SOURCE_RUN = "native_transit_selector_information_stage252_20261015_r1"
coverage, source = replay.coverage, replay.source
ROOTS, SCENARIOS, VIEWS = replay.ROOTS, replay.SCENARIOS, previous.VIEWS
DRAWS, CHUNKS = tuple(range(64)), tuple(range(16))
VARIANTS = learning.VARIANTS


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "variants": list(VARIANTS), "cohort": "same240_states40_prefixes32_cells4_views_both_model_banks_train60_held60",
        "frozen": "Stage252_full_actions_read_verbatim_all_six_protocols_and_cached_Stage247_controls_no_selection_or_inference",
        "draws": list(DRAWS), "halves": [list(range(32)), list(range(32, 64))],
        "future_seed": "1500000000+Stage195_state_index*64+draw_disjoint_from_previous_Transit_futures",
        "pairing": "same_new_state_draw_for_all_protocols_banks_scopes_and_views_shared_exposures_counted_once",
        "replay": "sparse_divergent_action_union_plus_original_CRN_identity_original_frozen_greedy_suffix",
        "zeros": "same_action_contrasts_exact_zero_full_cohort_denominators_no_absolute_outcome_imputation",
        "primary": "held_mixed_pooled_expanded8_minus_matched2_separately_within_free_old_and_shrunk_pooled_safe",
        "success": "conjunction_across_both_families_cost_wait_primary_and_expanded8_vs_raw_upper95_below_zero_both32_halves_and_both_banks_negative",
        "secondary": "all_four_views_train60_held60_both_banks_old_A2_Stage247_contrasts_prefix_attribution_and_directional_fleet_events",
        "uncertainty": "fixed_prefix_conditional_MC_normal95_not_training_seed_CI_or_rare_tail_guarantee; no_either_arm_winner_claim",
        "budget": {"draws_per_active_state": 64, "draws_per_chunk": 4,
            "native_episodes_per_draw": "2+sealed_action_union_including_original", "new_prefixes": 0,
            "new_raw_reference_fits": 0, "new_critic_fits": 0, "actor_updates": 0},
        "storage": "server_only_seal_rollouts_and_labels; sync_small_seal_preflight_final_report_only",
        "boundary": "new_future_confirmation_on_development_prefixes_one_action_intervention_not_complete_HRL_no_Stage248_outcomes_or_Stage169_reserves"}


def continuation_seed(*key):
    return 1500000000+coverage.states().index(key[:4])*64+key[4]


def seal_path(run):
    return ROOT/"results"/run/"seal_raw/choices.json"


def draw_path(run, *key):
    return coverage.draw_path(run, *key)


def chunk_path(run, key):
    r, s, i, p, c = key
    return ROOT/"results"/run/"chunks"/s/f"root_{r}"/f"replica_{i}"/f"probe_{p}"/f"chunk_{c}/receipt.json"


def seal(run):
    bank = json.loads(coverage.source.bank_path(coverage.SOURCE_RUN).read_text())
    rows = {replay.learning.state_key(r): r for r in bank["training_data"]}
    cells = []
    for key in previous.KEYS:
        raw = raw_directory(previous.result_path(SOURCE_RUN, *key))
        info = json.loads((raw/"aggregate.json").read_text())
        if info["protocol"] != previous.EXPERIMENT_PROTOCOL or info["contract"] != previous.contract() or not info["coverage_qualified"]:
            raise ValueError("Require qualified Stage252 actions")
        blocks = json.loads((raw/"actions.json").read_text())
        if len(blocks) != 4 or {(b["bank"], b["scope"]) for b in blocks} != {(b, s) for b in (0, 1) for s in ("train60", "held60")}:
            raise ValueError("Changed frozen model-bank scopes")
        for b in blocks:
            if len(b["states"]) != 60: raise ValueError("Changed frozen full-model state coverage")
            cells.append({"key": [*key, b["bank"], b["scope"]], "blocks": [{"states": [
                {**dict(zip(("root", "scene_seed", "probe_index"), st)), "scenario": rows[tuple(st)]["scenario"]}
                for st in b["states"]], "choices": b["choices"]}]})
    lookup = {(r, source.scene_seed(r, s, i), p): (r, s, i, p) for r, s, i, p in coverage.states()}
    states = {replay.learning.state_key(r) for c in cells for b in c["blocks"] for r in b["states"]}
    if states != lookup.keys(): raise ValueError("Changed existing 240-state cohort")
    active = [{"state": list(st), "key": list(lookup[st]), "candidates": sorted(set(cs)|{rows[st]["original_choice"]})}
        for st, cs in learning.frozen.action_union(cells, variants=VARIANTS).items()]
    for root in ROOTS:
        _, checkpoint = source.native.load_source(root)
        if not checkpoint.is_dir(): raise ValueError("Missing native source checkpoint")
    for a in active:
        if not coverage.noise.source_path(ROOT/"results"/coverage.noise.COLLECTION_RUN, *a["key"]).is_file():
            raise ValueError("Missing exact original-source replay")
    payload = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "cells": cells, "active": active}
    write_json(seal_path(run), payload)
    write_json(ROOT/"results"/run/"seal/result.json", {k: v for k, v in payload.items() if k != "cells"} | {
        "coverage_qualified": True, "fixed_states": len(states), "frozen_cells": len(cells), "active_states": len(active),
        "collection_draws": len(active)*64, "collection_chunks": len(active)*16,
        "native_episodes": sum(2+len(a["candidates"]) for a in active)*64, "training_updates": 0})
    print("SELECTOR_CONFIRMATION_ACTIONS_SEALED", flush=True)


def load_seal(run):
    r = json.loads(seal_path(run).read_text())
    if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract(): raise ValueError("Changed frozen confirmation")
    return r


def preflight_keys(sealed):
    active = [tuple(a["key"]) for a in sealed["active"]]
    return [(*next(k for k in active if k[:2] == (r, s)), 0) for r in ROOTS for s in SCENARIOS
        if any(k[:2] == (r, s) for k in active)]


def chunk_keys(sealed):
    return [(*a["key"], c) for a in sealed["active"] for c in CHUNKS]


def chunk_draws(sealed, key):
    initial = set(preflight_keys(sealed))
    return [(*key[:4], d) for d in range(4*key[4], 4*(key[4]+1)) if (*key[:4], d) not in initial]


def qualify_draw(r, key, item):
    if r["protocol"] != EXPERIMENT_PROTOCOL or r["contract"] != contract() or key[-1] not in DRAWS:
        raise ValueError("Changed selector confirmation protocol")
    replay.qualify_physical_draw(r, key, item, continuation_seed(*key))


def collect(run, key, *, preflight=False):
    sealed = load_seal(run)
    item = next(a for a in sealed["active"] if tuple(a["key"]) == key[:4])
    keys = [key] if preflight else chunk_draws(sealed, key)
    episodes = 0
    for k in keys:
        output = draw_path(run, *k)
        result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
            **replay.replay_draw(k, item, continuation_seed(*k), output)}
        qualify_draw(result, k, item)
        write_json(raw_directory(output)/"result.json", result)
        if preflight: write_json(output, {k: v for k, v in result.items() if k != "labels"})
        episodes += result["frozen_weight_qualified_episodes"]
        print(f"SELECTOR_CONFIRMATION_DRAW_COMPLETE {k}", flush=True)
    if not preflight:
        write_json(chunk_path(run, key), {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "key": list(key),
            "draws": [k[-1] for k in keys], "software_qualified": True, "native_episodes": episodes, "training_updates": 0})


def report(run, *, preflight=False):
    sealed = load_seal(run)
    by_state = {tuple(a["state"]): a for a in sealed["active"]}
    items = {tuple(a["key"]): a for a in sealed["active"]}
    keys = preflight_keys(sealed) if preflight else [(*st, d) for st in items for d in DRAWS]
    records = {}
    for key in keys:
        r = json.loads((raw_directory(draw_path(run, *key))/"result.json").read_text())
        qualify_draw(r, key, items[key[:4]])
        records[key] = r
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "coverage_qualified": True, "draws": len(records),
        "native_episodes": sum(r["frozen_weight_qualified_episodes"] for r in records.values()),
        "wall_seconds_max": max(r["wall_seconds"] for r in records.values()),
        "peak_process_rss_mb_max": max(r["peak_process_rss_mb"] for r in records.values()), "training_updates": 0}
    if not preflight:
        panels = learning.frozen.sparse_panels(sealed["cells"], DRAWS,
            lambda state, draw: records[(*by_state[state]["key"], draw)]["labels"], variants=VARIANTS)
        def summaries(cs):
            return {ds: learning.summarize(cs, {k: {"values": p["values"][sl]} for k, p in panels.items()})
                for ds, sl in (("half0_32", slice(0, 32)), ("half1_32", slice(32, 64)), ("fresh64", slice(0, 64)))}
        groups = {"mixed_pooled": lambda k: k[0].startswith("mixed"), **{v: lambda k, v=v: k[0] == v for v in VIEWS}}
        result["groups"] = {v: {scope: {label: summaries([c for c in sealed["cells"]
            if predicate(c["key"]) and c["key"][5] == scope and (bank is None or c["key"][4] == bank)])
            for label, bank in (("pooled", None), ("bank0", 0), ("bank1", 1))
            } for scope in ("train60", "held60")} for v, predicate in groups.items()}
        mixed = [c for c in sealed["cells"] if c["key"][0].startswith("mixed")]
        result["mixed_by_scenario"] = {scope: {s: summaries(learning.frozen.subset(cs, scenario=s))
            for s in sorted({r["scenario"] for c in cs for b in c["blocks"] for r in b["states"]})}
            for scope in ("train60", "held60") for cs in ([c for c in mixed if c["key"][5] == scope],)}
        result["attribution"] = {scope: {f"{m}_expanded8-{s}": learning.attribution(cs, panels, f"{m}_expanded8_pooled_safe", baseline)
            for m in learning.MODELS for s, baseline in (("matched2", f"{m}_matched2_pooled_safe"), ("raw", "raw"), ("Stage247", "old_pooled_safe"))}
            for scope in ("train60", "held60") for cs in ([c for c in mixed if c["key"][5] == scope],)}
        result["confirmation"] = learning.confirm(result["groups"]["mixed_pooled"]["held60"])
        result["default_changed"] = result["native_deployment_eligible"] = False
    write_json(ROOT/"results"/run/("preflight_report" if preflight else "report")/"result.json", result)
    print("SELECTOR_CONFIRMATION_REPORT_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--operation", choices=("seal", "preflight", "collect", "preflight_report", "report"), required=True)
    p.add_argument("--seed", type=int, choices=ROOTS)
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--replica", type=int, choices=coverage.REPLICAS)
    p.add_argument("--probe", type=int, choices=coverage.PROBES)
    p.add_argument("--draw", type=int, choices=DRAWS)
    p.add_argument("--chunk", type=int, choices=CHUNKS)
    args = p.parse_args()
    if args.operation == "seal": seal(args.run_name)
    elif args.operation in ("preflight", "collect"):
        collect(args.run_name, (args.seed, args.scenario, args.replica, args.probe,
            args.draw if args.operation == "preflight" else args.chunk), preflight=args.operation == "preflight")
    else: report(args.run_name, preflight=args.operation == "preflight_report")


if __name__ == "__main__": main()
