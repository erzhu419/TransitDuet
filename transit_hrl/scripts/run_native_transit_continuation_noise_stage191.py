#!/usr/bin/env python3
"""Replicate matched action labels with independent post-probe innovations."""

import argparse
import copy
from functools import partial
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_native_transit_value_expansion_stage187 as source
from scripts import run_native_transit_coverage_value_stage188 as bank_source
from scripts import run_native_transit_calibrated_value_stage190 as calibration_source
from scripts.analyze_native_transit_value_expansion_stage187 import result_path as source_path
from scripts.submit_native_transit_coverage_value_stage188_scheduleurm import result_path as prediction_path
from freq_hrl.domains.transit.native_continuation_value import NativeContinuationValueAudit
from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json

EXPERIMENT_PROTOCOL = "native_transit_continuation_noise_stage191_v1"
BANK_RUN = calibration_source.SOURCE_RUN
CALIBRATION_RUN = "native_transit_calibrated_value_stage190_fit_20261012_r1"
ROOTS, SCENARIOS = source.ROOTS, source.SCENARIOS
PROBES, DRAWS = (8, 22, 36), tuple(range(8))
COLLECTION_RUN = bank_source.COLLECTION_RUN


def continuation_seed(root, scenario, probe, draw):
    return 920000000+root*100000+SCENARIOS.index(scenario)*10000+probe*100+draw


def cells(*, preflight=False):
    return [(r, s, p, d) for r in ROOTS for s in SCENARIOS for p in PROBES for d in DRAWS
        if not preflight or (p == 22 and d == 0)]


def contract():
    return {"bank_run": BANK_RUN, "collection_run": COLLECTION_RUN, "calibration_run": CALIBRATION_RUN,
        "roots": list(ROOTS), "scenarios": list(SCENARIOS), "probes": list(PROBES), "draws": list(DRAWS),
        "source_states": "Stage187_replica0_only_30_existing_states_in10_scene_groups",
        "future_seed": "920000000+root*100000+regime_index*10000+probe*100+draw",
        "intervention": "replace_ScenarioTape_after_exact_preaction_snapshot_then_one_candidate_and_original_greedy_suffix",
        "randomness": "independent_future_Poisson_route_speed_streams_shared_within_six_actions",
        "retained": "all_prefix_physics_cost_RNG_and_episode_demand_profile_peak_phase_OD_multipliers_lower_weights_duration_prior",
        "control": "one_exact_original_observer_plus_resampled_observer_six_actions_including_exact_resampled_identity",
        "frozen_choices": "Stage188_wait_feasible_and_Stage190_ensemble_mean_all_modes_splits_read_before_rollouts",
        "preflight": "both_roots_all5regimes_probe22_draw0_then_remaining230",
        "budget": {"tasks": 240, "native_episodes": 1920, "native_clock_ticks": 1920*61380,
            "new_prefix_scenes": 0, "training_updates": 0},
        "analysis": "paired_effect_power_minus_sampling_variance_over8_mean_noise_cross_half_rank_and_frozen_choice_transfer",
        "storage": "bank_full_context_snapshots_rollouts_and_draw_labels_server_only_compact_receipts_report_for_sync",
        "boundary": "conditional_innovation_diagnostic_not_new_performance_confirmation_or_learned_upper_training"}


def result_path(directory, root, scenario, probe, draw):
    return directory/"cells"/scenario/f"root_{root}"/f"probe_{probe}"/f"draw_{draw}"/"receipt.json"


def frozen_choices(row, bank):
    result = {}
    for split in bank_source.SPLITS:
        held = {"scene": [row["root"], row["scene_seed"]], "regime": row["scenario"], "lower_root": row["root"]}[split]
        fold = next(f for f in bank["manifest"]["folds"] if f["split"] == split and f["held_out"] == held)
        for mode in bank_source.MODES:
            cell = json.loads(prediction_path(ROOT/"results"/BANK_RUN, mode, fold).read_text())
            later = json.loads((raw_directory(calibration_source.result_path(CALIBRATION_RUN, mode, fold))/"result.json").read_text())
            for name, predictions in (("stage188", cell["predictions"]["wait_feasible"]),
                    ("stage190", later["predictions"]["ensemble_mean"])):
                prediction = next(p for p in predictions if (p["root"], p["scene_seed"], p["probe_index"])
                    == (row["root"], row["scene_seed"], row["probe_index"]))
                result[f"{name}:{mode}:{split}"] = prediction["selected"]
    return result


def collect(root, scenario, probe, draw, model, output):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(source.native.NATIVE))
    import frequency
    frequency.DemandFrequencyTracker = source.native.NativeRoutingTracker
    torch.set_num_threads(1)
    bank = json.loads(bank_source.bank_path(BANK_RUN).read_text())
    if not (bank["protocol"] == bank_source.EXPERIMENT_PROTOCOL and bank["contract"] == bank_source.contract()
            and bank["software_qualified"]):
        raise ValueError("Noise diagnostic requires the unchanged qualified bank")
    scene = source.scene_seed(root, scenario, 0)
    registered = next(r for r in bank["training_data"] if
        (r["root"], r["scene_seed"], r["probe_index"]) == (root, scene, probe))
    choices = frozen_choices(registered, bank)
    print(f"NOISE_CHOICES_SEALED root={root} regime={scenario} probe={probe} draw={draw}", flush=True)
    source.shared.source.incumbent.source.distributional.validate_model(model)
    _, checkpoint = source.native.load_source(root)
    raw = raw_directory(output)
    original, original_plan = source.shared.episode(root, scenario, scene, raw/"original", checkpoint,
        model, probe, plan_class=NativeTimetableContextAudit)
    prior = json.loads(source_path(ROOT/"results"/COLLECTION_RUN, root, scenario, 0, probe).read_text())
    source.qualify_cell(prior)
    source.qualify_observer(original)
    if ({k: original[k] for k in source.OUTCOME_KEYS} != prior["observer"]
            or source.shared.policy_trace(original["supply"]) != prior["policy_trace"]):
        raise ValueError("Original diagnostic replay changed its existing native outcome or policy trace")
    context = next(c for c in original_plan.contexts if c["index"] == probe)
    if (original["probe"]["candidates"] != registered["candidates"]
            or original["probe"]["native_state_v2"] != registered["native_state_v2"]
            or {k: context[k] for k in source.context_source.OCCUPANCY_KEYS} != registered["occupancy_context"]
            or context["timetable_context"] != registered["timetable_context"]):
        raise ValueError("Original replay changed the critic's causal inputs")
    seed = continuation_seed(root, scenario, probe, draw)
    plan_class = partial(NativeContinuationValueAudit, continuation_seed=seed)
    baseline, plan = source.shared.episode(root, scenario, scene, raw/"observer", checkpoint, model,
        probe, reference=original_plan, plan_class=plan_class)
    source.qualify_observer(baseline)
    branches = []
    order = [registered["original_choice"], *[i for i in range(6) if i != registered["original_choice"]]]
    for candidate in order:
        row, branch_plan = source.shared.episode(root, scenario, scene, raw/f"candidate_{candidate}",
            checkpoint, model, probe, candidate=candidate, reference=plan, plan_class=plan_class)
        source.shared.qualify(row, baseline, candidate=candidate)
        if branch_plan.env.scenario_tape.seed != seed or not branch_plan.env.scenario_tape._streams:
            raise ValueError("Native branch did not consume the registered future innovation tape")
        row["policy_trace"] = source.shared.policy_trace(row.pop("supply"))
        branches.append(row)
        print(f"noise_draw {root}/{scenario}/{probe}/{draw}/{candidate} "
            f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']}", flush=True)
    branches.sort(key=lambda r: r["probe"]["selected"])
    baseline["policy_trace"] = source.shared.policy_trace(baseline["supply"])
    source.validate_continuations(baseline, branches)
    repeated_context = next(c for c in plan.contexts if c["index"] == probe)
    record = source.training_record(root, scenario, probe, baseline, branches, plan.snapshot, repeated_context)
    if {k: v for k, v in record.items() if k != "labels"} != {k: v for k, v in registered.items() if k != "labels"}:
        raise ValueError("Future innovations changed preaction features or candidate plans")
    write_json(raw/"paired_rollouts.json", {"original": original, "snapshot": original_plan.snapshot,
        "observer": baseline, "branches": branches, "context": repeated_context})
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": contract(), "software_qualified": True,
        "root": root, "scenario": scenario, "scene_seed": scene, "probe_index": probe, "draw": draw,
        "continuation_seed": seed, "original_choice": registered["original_choice"],
        "existing_labels": copy.deepcopy(registered["labels"]), "labels": record["labels"], "frozen_choices": choices,
        "original_source_reproductions": 1, "exact_resampled_identities": 1,
        "prefix_state_rng_matches": 7, "frozen_weight_qualified_episodes": 8, "window_ledger_qualified_episodes": 8,
        "continuation_qualified": True, "native_clock_ticks": 8*61380, "training_updates": 0}
    qualify_cell(result)
    return result


def qualify_cell(cell):
    key = cell["root"], cell["scenario"], cell["probe_index"], cell["draw"]
    if not (key in cells() and cell["protocol"] == EXPERIMENT_PROTOCOL and cell["contract"] == contract()
            and cell["software_qualified"] and cell["scene_seed"] == source.scene_seed(key[0], key[1], 0)
            and cell["continuation_seed"] == continuation_seed(*key)
            and cell["original_source_reproductions"] == cell["exact_resampled_identities"] == 1
            and cell["prefix_state_rng_matches"] == 7 and cell["frozen_weight_qualified_episodes"] == 8
            and cell["window_ledger_qualified_episodes"] == 8 and cell["continuation_qualified"]
            and cell["native_clock_ticks"] == 8*61380 and cell["training_updates"] == 0):
        raise ValueError("Changed conditional continuation state, seed, qualification or budget")
    expected = {f"{stage}:{mode}:{split}" for stage in ("stage188", "stage190")
        for mode in bank_source.MODES for split in bank_source.SPLITS}
    if set(cell["frozen_choices"]) != expected or any(c not in range(6) for c in cell["frozen_choices"].values()):
        raise ValueError("Missing or changed frozen causal-choice references")
    for labels in (cell["labels"], cell["existing_labels"]):
        if [l["candidate"] for l in labels] != list(range(6)):
            raise ValueError("Missing repeated candidate labels")
        for label in labels:
            values = [label[k] for k in (*source.prior.value.HEADS, "actual_cost_delta", "episode_reward_delta")]
            if not (np.all(np.isfinite(values)) and np.isclose(label["actual_cost_delta"],
                    label["fleet_cost_delta"]+label["nonfleet_cost_delta"], rtol=0, atol=1e-12)):
                raise ValueError("Changed continuation label units")
        identity = labels[cell["original_choice"]]
        if any(identity[k] != 0 for k in (*source.prior.value.HEADS, "actual_cost_delta", "episode_reward_delta")):
            raise ValueError("Resampled identity is not exactly its same-tape baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--probe", type=int, choices=PROBES, required=True)
    parser.add_argument("--draw", type=int, choices=DRAWS, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.seed, args.scenario, args.probe, args.draw,
        json.loads(args.models.read_text())[str(args.seed)], args.output)
    write_json(raw_directory(args.output)/"result.json", result)
    write_json(args.output, {k: v for k, v in result.items() if k not in ("labels", "existing_labels", "frozen_choices")})
    print("NATIVE_CONTINUATION_NOISE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
