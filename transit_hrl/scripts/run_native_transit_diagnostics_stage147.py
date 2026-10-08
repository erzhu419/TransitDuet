#!/usr/bin/env python3
"""Diagnose frozen Stage146 actors without retraining or downloading weights."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_diagnostics import NativePolicyProbe
from freq_hrl.domains.transit.native_routing import NativeRoutingTracker
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_routing_stage146 as routing
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_diagnostics_stage147_v1"
SOURCE_RUN = "native_transit_routing_stage146_development_20261009_r1"
ROOTS = routing.ROOTS
METHODS = routing.METHODS


def conditions(method, *, preflight):
    return ("baseline",) if preflight or method != "correct" else (
        "baseline", "neutral_upper", "zero_holding")


def contract(preflight):
    return {
        "purpose": "frozen_mechanism_diagnosis_not_new_performance_claim",
        "source_protocol": routing.EXPERIMENT_PROTOCOL, "source_run": SOURCE_RUN,
        "checkpoint_ep": 299, "training_updates": 0,
        "scenarios": ["low_noise"] if preflight else list(routing.SCENARIOS),
        "scene_seeds": "first_registered_stage146_scene_per_root_and_regime",
        "conditions": {method: list(conditions(method, preflight=preflight)) for method in METHODS},
        "actor_dims": {"upper": 16, "lower": 33},
        "zero_input_probe": "offline_same_state_deterministic_action_sensitivity",
        "neutral_upper": "zero_target_headway_delta_same_fixed_dispatch_schedule",
        "zero_holding": "zero_learned_holding_native_boarding_dwell_unchanged",
        "baseline_gate": "exact_reproduction_of_source_metrics_and_exogenous_demand",
        "preflight": preflight, "artifacts": "compact_JSON_only_server_checkpoint_reuse",
    }


def check_episode(row, reference, *, baseline):
    for field in ("simulation_end_time_s", "passengers_generated", "N_fleet"):
        if row[field] != reference[field]:
            raise RuntimeError(f"Frozen diagnostic changed {field}")
    if baseline:
        for field in (*routing.METRICS, "passengers_unserved", "trips_completed", "ep_steps"):
            if row[field] != reference[field]:
                raise RuntimeError(f"Passive diagnostic failed source reproduction: {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(NATIVE))
    import torch
    import frequency
    frequency.DemandFrequencyTracker = NativeRoutingTracker
    from runner_v3 import TransitDuetV2Runner, load_config
    torch.set_num_threads(1)
    source_path = ROOT / "results" / SOURCE_RUN / "cells" / args.method / f"seed_{args.seed}" / "result.json"
    source = json.loads(source_path.read_text())
    if not (source["passed"] and source["method"] == args.method and source["seed"] == args.seed
            and source["protocol"] == routing.EXPERIMENT_PROTOCOL
            and source["contract"] == routing.contract(False)):
        raise RuntimeError("Source is not the completed Stage146 training cell")
    checkpoint = source_path.parent.with_name(source_path.parent.name + "_raw") / "training" / (
        f"F_freqduet_harmonic_hiro_seed{args.seed}") / "checkpoints"
    raw = raw_directory(args.output)
    spec = contract(args.preflight)
    evaluations, total_steps = [], 0
    for condition in conditions(args.method, preflight=args.preflight):
        for scenario in spec["scenarios"]:
            scene_seed = routing.evaluation_seeds(args.seed, scenario, preflight=False)[0]
            reference = next(r for r in source["evaluation"]
                             if r["scenario"] == scenario and r["scene_seed"] == scene_seed)
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            cfg = routing.configure(load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
                                    args.method, args.seed, preflight=False)
            cfg["env"].update(copy.deepcopy(routing.SCENARIOS[scenario]))
            cfg["logging"] = {"logs_dir": str(raw / condition / scenario)}
            runner = TransitDuetV2Runner(cfg, device="cpu")
            runner.load_checkpoint(checkpoint, 299, require_deployment_state=True)
            before = model_arrays(runner)
            probes = {}
            for level in ("upper", "lower"):
                policy = getattr(runner, f"{level}_trainer").policy_net
                probes[level] = NativePolicyProbe(policy, level, neutral=(
                    (level == "upper" and condition == "neutral_upper")
                    or (level == "lower" and condition == "zero_holding")))
                policy.get_action = probes[level].get_action
            row = runner.run_episode(300, training=False, N_fleet_override=12,
                                     scenario_seed=scene_seed, record_diagnostics=False)
            check_episode(row, reference, baseline=condition == "baseline")
            diagnostics = {level: probe.summarize() for level, probe in probes.items()}
            after = model_arrays(runner)
            if any(not np.array_equal(before[k], after[k]) for k in before):
                raise RuntimeError("Frozen diagnostic changed a network")
            evaluations.append({"condition": condition, "scenario": scenario, "scene_seed": scene_seed,
                **routing.compact_row(row), "actors": diagnostics,
                **{key: row[key] for key in ("lower_action_mean", "lower_action_std",
                    "upper_delta_mean", "upper_delta_std")}})
            total_steps += int(row["simulation_end_time_s"] / runner.env.time_step)
            print(f"{args.method} root={args.seed} {condition}/{scenario} "
                  f"cost={row['service_cost_restricted']} "
                  f"holding={row['lower_action_mean']} upper={row['upper_delta_mean']}", flush=True)
            for probe in probes.values():
                probe.policy.get_action = probe.original_get_action
            del runner, probes, before, after
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": spec,
        "source_run": SOURCE_RUN, "method": args.method, "seed": args.seed,
        "software_qualified": True, "baseline_reproduced": True,
        "training_updates": 0, "native_steps": total_steps, "evaluation": evaluations})
    print("NATIVE_DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
