#!/usr/bin/env python3
"""Replace the remaining singleton targets with matched conditional future means."""

import argparse
import copy
from functools import partial
import json
import os
from pathlib import Path
import resource
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_bounded_wait_stage211 as previous
from scripts import run_native_transit_mean_coverage_stage195 as mean_source
from freq_hrl.domains.transit.native_continuation_value import NativeContinuationValueAudit
from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_training_means_stage212_v1"
SOURCE_RUN = "native_transit_bounded_wait_stage211_development_20261013_r1"
MASS_RUN = "native_transit_wait_feasibility_stage206_development_20261013_r1"
MODEL_RUN = "native_transit_mean_coverage_stage195_development_20261012_r1"
noise, source = mean_source.noise, mean_source.noise.source
ROOTS, SCENARIOS, PROBES, DRAWS = source.ROOTS, source.SCENARIOS, source.PROBES, noise.DRAWS
state_key = mean_source.state_key


def contract():
    return {"source_run": SOURCE_RUN, "source_protocol": previous.EXPERIMENT_PROTOCOL,
        "bank_run": mean_source.SOURCE_RUN, "mass_run": MASS_RUN,
        "data": "same600_causal_inputs_600_mean8_targets_original240_evaluation_states_and47_outer_folds",
        "new_targets": "120_existing_states_first_two_opened_scenes_plus240_fresh_states_replicas4_to7",
        "seeds": "Stage191_seed+continuation_slot*1000000_fresh_slot4_to7_existing_slot8_plus_scene_index",
        "intervention": noise.contract()["intervention"], "randomness": noise.contract()["randomness"],
        "retained": noise.contract()["retained"], "control": noise.contract()["control"],
        "weights": "preserve_original_240_count8_and360_count1_optimization_weights_separate_from_actual_draw_count8",
        "evaluation": "original240_mean8_states_future_panels_and47_outer_folds_unchanged_no_new_eval_labels",
        "next_fit": "refit_training_only_outer_and_inner_symmetric_GLM_then_same_B1_ridge0_or0.1_bounded_heads_inner_validation_on_original240_only",
        "preflight": "20_cells_two_origins_both_roots_all5regimes_probe8_draw0_before2860_remaining",
        "budget": {"collection_tasks": 2880, "new_native_episodes": 23040,
            "new_native_clock_ticks": 23040*61380, "new_prefix_scenes": 0, "training_updates": 0},
        "storage": "raw_rollouts_draw_labels_mean_targets_server_only_small_preflight_and_prepare_receipts",
        "gate": previous.contract()["gate"],
        "boundary": "conditional_training_label_precision_development_not_new_independent_CI_or_full_learned_HRL"}


def manifest(old):
    if old["protocol"] != previous.EXPERIMENT_PROTOCOL or old["contract"] != previous.contract():
        raise ValueError("Training means require the exact Stage211 cohort")
    evaluation = {state_key(r) for r in old["mean_states"]}
    training = []
    for group in old["groups"]:
        root, scenario, scene = group["root"], group["scenario"], group["scene_seed"]
        if group["origin"] == "existing":
            scenes = sorted(g["scene_seed"] for g in old["groups"]
                if (g["origin"], g["root"], g["scenario"]) == ("existing", root, scenario))
            if len(scenes) != 2:
                raise ValueError("Only the first two already-opened source scenes are allowed")
            index = scenes.index(scene)
            slot = 8+index
        else:
            index = next(i for i in source.REPLICAS if source.scene_seed(root, scenario, i) == scene)
            slot = index
        for probe in PROBES:
            if (root, scene, probe) not in evaluation:
                if group["origin"] == "fresh" and index not in range(4, 8):
                    raise ValueError("Changed the original mean8 evaluation cohort")
                training.append({**group, "scene_index": index, "continuation_slot": slot, "probe_index": probe})
    if (len(old["groups"]) != 100 or len(old["folds"]) != 47 or len(evaluation) != 240
            or len(training) != 360 or sum(r["origin"] == "existing" for r in training) != 120):
        raise ValueError("Changed the 600-state training/evaluation population")
    return {**old, "protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "training_states": training}


def cell_key(row, draw):
    return row["root"], row["scenario"], row["origin"], row["scene_index"], row["probe_index"], draw


def cells(registered, *, preflight=False):
    return [cell_key(r, d) for r in registered["training_states"] for d in DRAWS
        if not preflight or (r["scene_index"] == (0 if r["origin"] == "existing" else 4)
            and r["probe_index"] == 8 and d == 0)]


def continuation_seed(row, draw):
    return noise.continuation_seed(row["root"], row["scenario"], row["probe_index"], draw)+row["continuation_slot"]*1000000


def draw_path(run_name, key):
    root, scenario, origin, index, probe, draw = key
    return ROOT/"results"/run_name/"draws"/origin/scenario/f"root_{root}"/f"scene_{index}"/f"probe_{probe}"/f"draw_{draw}"/"receipt.json"


def qualify_original(original, row):
    if row["origin"] == "existing":
        prior = source.labels.source_record(row["root"], row["scenario"], scene_index=row["scene_index"])
        if prior["scene_seed"] != row["scene_seed"]:
            raise ValueError("Existing training scene differs from its original source")
        source.shared.qualify(original, {**prior, "probe": original["probe"]})
    else:
        prior = json.loads(noise.source_path(ROOT/"results"/noise.COLLECTION_RUN,
            row["root"], row["scenario"], row["scene_index"], row["probe_index"]).read_text())
        source.qualify_cell(prior)
        if ({k: original[k] for k in source.OUTCOME_KEYS} != prior["observer"]
                or source.shared.policy_trace(original["supply"]) != prior["policy_trace"]):
            raise ValueError("Fresh replay changed its original native outcome or policy trace")
    source.qualify_observer(original)


def replay_draw(row, draw, bank_row, model, output):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(source.native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = source.native.NativeRoutingTracker
    torch.set_num_threads(1)
    root, scenario, scene, probe = row["root"], row["scenario"], row["scene_seed"], row["probe_index"]
    seed, raw = continuation_seed(row, draw), raw_directory(output)
    source.shared.source.incumbent.source.distributional.validate_model(model)
    _, checkpoint = source.native.load_source(root)
    original, original_plan = source.shared.episode(root, scenario, scene, raw/"original", checkpoint,
        model, probe, plan_class=NativeTimetableContextAudit)
    qualify_original(original, row)
    context = next(c for c in original_plan.contexts if c["index"] == probe)
    if (original["probe"]["candidates"] != bank_row["candidates"]
            or original["probe"]["native_state_v2"] != bank_row["native_state_v2"]
            or {k: context[k] for k in source.context_source.OCCUPANCY_KEYS} != bank_row["occupancy_context"]
            or context["timetable_context"] != bank_row["timetable_context"]):
        raise ValueError("Original replay changed the frozen causal inputs")
    plan_class = partial(NativeContinuationValueAudit, continuation_seed=seed)
    baseline, plan = source.shared.episode(root, scenario, scene, raw/"observer", checkpoint, model,
        probe, reference=original_plan, plan_class=plan_class)
    source.qualify_observer(baseline)
    branches = []
    order = [bank_row["original_choice"], *[i for i in range(6) if i != bank_row["original_choice"]]]
    for candidate in order:
        result, branch_plan = source.shared.episode(root, scenario, scene, raw/f"candidate_{candidate}",
            checkpoint, model, probe, candidate=candidate, reference=plan, plan_class=plan_class)
        source.shared.qualify(result, baseline, candidate=candidate)
        if branch_plan.env.scenario_tape.seed != seed or not branch_plan.env.scenario_tape._streams:
            raise ValueError("Branch did not consume its registered independent future tape")
        result["policy_trace"] = source.shared.policy_trace(result.pop("supply"))
        branches.append(result)
        print(f"TRAIN_MEAN_DRAW {cell_key(row, draw)} candidate={candidate} "
            f"cost={result['service_cost_restricted']} wait={result['restricted_wait_horizon_min']}", flush=True)
    branches.sort(key=lambda b: b["probe"]["selected"])
    baseline["policy_trace"] = source.shared.policy_trace(baseline["supply"])
    source.validate_continuations(baseline, branches)
    context = next(c for c in plan.contexts if c["index"] == probe)
    record = source.training_record(root, scenario, probe, baseline, branches, plan.snapshot, context)
    if {k: v for k, v in record.items() if k != "labels"} != {k: v for k, v in bank_row.items() if k != "labels"}:
        raise ValueError("Future sampling changed preaction features or candidate plans")
    write_json(raw/"paired_rollouts.json", {"original": original, "snapshot": original_plan.snapshot,
        "observer": baseline, "branches": branches, "context": context})
    return {**row, "draw": draw, "protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "software_qualified": True, "continuation_seed": seed, "original_choice": bank_row["original_choice"],
        "existing_labels": copy.deepcopy(bank_row["labels"]), "labels": record["labels"],
        "original_source_reproductions": 1, "exact_resampled_identities": 1, "prefix_state_rng_matches": 7,
        "frozen_weight_qualified_episodes": 8, "window_ledger_qualified_episodes": 8, "continuation_qualified": True,
        "native_clock_ticks": 8*61380, "training_updates": 0,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.}


def qualify_draw(cell, row, draw):
    if not (all(cell[k] == v for k, v in row.items()) and draw in DRAWS and cell["draw"] == draw
            and cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract() and cell["software_qualified"]
            and cell["continuation_seed"] == continuation_seed(row, draw)
            and cell["original_source_reproductions"] == cell["exact_resampled_identities"] == 1
            and cell["prefix_state_rng_matches"] == 7 and cell["frozen_weight_qualified_episodes"] == 8
            and cell["window_ledger_qualified_episodes"] == 8 and cell["continuation_qualified"]
            and cell["native_clock_ticks"] == 8*61380 and cell["training_updates"] == 0):
        raise ValueError("Changed the registered conditional draw or physical budget")
    for labels in (cell["labels"], cell["existing_labels"]):
        if [l["candidate"] for l in labels] != list(range(6)):
            raise ValueError("Missing a paired action target")
        for label in labels:
            if not (np.all(np.isfinite([label[f] for f in mean_source.expected.LABEL_FIELDS]))
                    and label["future_fleet_cost"] >= 0
                    and np.isclose(label["actual_cost_delta"], label["fleet_cost_delta"]+label["nonfleet_cost_delta"], atol=1e-12, rtol=0)
                    and np.isclose(label["future_fleet_cost"]-labels[cell["original_choice"]]["future_fleet_cost"], label["fleet_cost_delta"], atol=1e-12, rtol=0)
                    and np.isclose(label["physical_return_delta"], -100*label["actual_cost_delta"], atol=1e-12, rtol=0)):
                raise ValueError("Changed physical target units or fleet floor")
        if any(labels[cell["original_choice"]][f] != 0 for f in mean_source.expected.LABEL_FIELDS if f != "future_fleet_cost"):
            raise ValueError("Same-tape original action is not an exact identity")


def aggregate_state(row, draws, bank_row, old_mass):
    if len(draws) != 8 or old_mass["draw_count"] != 1:
        raise ValueError("Only registered singleton targets can be upgraded")
    for draw, cell in enumerate(draws):
        qualify_draw(cell, row, draw)
        if cell["existing_labels"] != bank_row["labels"] or cell["original_choice"] != bank_row["original_choice"]:
            raise ValueError("Repeated target differs from its original bank state")
    labels = [{"candidate": i, **{f: float(np.mean([c["labels"][i][f] for c in draws]))
        for f in mean_source.expected.LABEL_FIELDS}} for i in range(6)]
    positive, negative = previous.source.feasibility.wait_masses(
        np.asarray([[l["wait_delta_min"] for l in c["labels"]] for c in draws]))
    mass = {**old_mass, "draw_count": 8, "sample_weight": 1,
        "positive_wait_min": positive.tolist(), "negative_wait_min": negative.tolist()}
    return {**{k: bank_row[k] for k in ("root", "scenario", "scene_seed", "probe_index")}, "labels": labels}, mass


def summarize_draws(registered, load, *, preflight=False):
    keys = cells(registered, preflight=preflight)
    rows = {cell_key(r, d): r for r in registered["training_states"] for d in DRAWS}
    rss = []
    for key in keys:
        cell = load(key)
        qualify_draw(cell, rows[key], key[-1])
        rss.append(cell["peak_process_rss_mb"])
    return {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "draws": len(keys), "native_episodes": len(keys)*8, "native_clock_ticks": len(keys)*8*61380,
        "by_origin": {o: sum(k[2] == o for k in keys) for o in ("existing", "fresh")},
        "peak_process_rss_mb_max": max(rss), "training_updates": 0}


def prepare(run_name, registered, load):
    bank = json.loads(mean_source.source.bank_path(mean_source.SOURCE_RUN).read_text())
    if (bank["protocol"] != mean_source.source.EXPERIMENT_PROTOCOL or bank["contract"] != mean_source.source.contract()
            or not bank["software_qualified"] or registered != manifest(
                json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text()))):
        raise ValueError("Changed the qualified bank or frozen split population")
    prepared = json.loads(previous.source.mass_path(MASS_RUN).read_text())
    if prepared["protocol"] != previous.source.EXPERIMENT_PROTOCOL or prepared["contract"] != previous.source.contract():
        raise ValueError("Changed the original mass targets")
    rows = {state_key(r): r for r in bank["training_data"]}
    masses = {state_key(r): r for r in prepared["targets"]}
    if len(rows) != 600 or set(rows) != set(masses):
        raise ValueError("Missing original states or wait masses")
    original_weights = {k: m["draw_count"] for k, m in masses.items()}
    targets, rss = [], []
    for row in registered["training_states"]:
        draws = [load(cell_key(row, d)) for d in DRAWS]
        key = state_key(row)
        target, masses[key] = aggregate_state(row, draws, rows[key], masses[key])
        targets.append(target)
        rss.extend(c["peak_process_rss_mb"] for c in draws)
    if any(m["draw_count"] != 8 or m.get("sample_weight", m["draw_count"]) != original_weights[k] for k, m in masses.items()):
        raise ValueError("Target precision changed the original optimization weights")
    output = raw_directory(ROOT/"results"/run_name/"prepare/receipt.json")
    write_json(output/"targets.json", {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(),
        "targets": list(masses.values()), "new_mean_labels": targets})
    report = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "draws": len(rss), "native_episodes": len(rss)*8, "native_clock_ticks": len(rss)*8*61380,
        "peak_process_rss_mb_max": max(rss), "training_updates": 0, "input_states": 600,
        "mean8_states": 600, "evaluation_states_unchanged": 240, "outer_folds_unchanged": 47,
        "sample_weight_sum": sum(original_weights.values()), "target_bytes": (output/"targets.json").stat().st_size}
    write_json(ROOT/"results"/run_name/"prepare/receipt.json", report)
    print("NATIVE_TRAINING_MEANS_PREPARED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--operation", choices=("collect", "preflight_report", "prepare"), required=True)
    parser.add_argument("--seed", type=int, choices=ROOTS)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--origin", choices=("existing", "fresh"))
    parser.add_argument("--scene-index", type=int)
    parser.add_argument("--probe", type=int, choices=PROBES)
    parser.add_argument("--draw", type=int, choices=DRAWS)
    args = parser.parse_args()
    registered = json.loads((ROOT/"results"/args.run_name/"inputs/manifest.json").read_text())
    old = json.loads((ROOT/"results"/SOURCE_RUN/"inputs/manifest.json").read_text())
    if registered != manifest(old):
        raise ValueError("Changed the registered training-target experiment")
    load = lambda key: json.loads((raw_directory(draw_path(args.run_name, key))/"result.json").read_text())
    if args.operation == "preflight_report":
        write_json(ROOT/"results"/args.run_name/"preflight_report/receipt.json", summarize_draws(registered, load, preflight=True))
        print("NATIVE_TRAINING_MEANS_PREFLIGHT_QUALIFIED", flush=True)
    elif args.operation == "prepare":
        prepare(args.run_name, registered, load)
    else:
        key = args.seed, args.scenario, args.origin, args.scene_index, args.probe, args.draw
        if key not in cells(registered): parser.error("Collect requires a registered singleton state/draw")
        row = next(r for r in registered["training_states"] if cell_key(r, args.draw) == key)
        bank = json.loads(mean_source.source.bank_path(mean_source.SOURCE_RUN).read_text())
        if bank["protocol"] != mean_source.source.EXPERIMENT_PROTOCOL or bank["contract"] != mean_source.source.contract() or not bank["software_qualified"]:
            raise ValueError("Collection requires the unchanged qualified bank")
        bank_row = next(r for r in bank["training_data"] if state_key(r) == state_key(row))
        model = json.loads((ROOT/"results"/args.run_name/"inputs/models.json").read_text())[str(args.seed)]
        output = draw_path(args.run_name, key)
        result = replay_draw(row, args.draw, bank_row, model, output)
        qualify_draw(result, row, args.draw)
        write_json(raw_directory(output)/"result.json", result)
        write_json(output, {k: v for k, v in result.items() if k not in ("labels", "existing_labels")})
        print("NATIVE_TRAINING_MEANS_DRAW_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
