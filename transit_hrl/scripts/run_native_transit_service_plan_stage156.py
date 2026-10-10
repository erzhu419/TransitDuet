#!/usr/bin/env python3
"""Qualify service allocation authority before training a new upper plan policy."""

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
from freq_hrl.domains.transit.native_service_plan import NativeServicePlan, CONDITIONS as PLAN_CONDITIONS
from freq_hrl.domains.transit.native_routing import NativeRoutingTracker
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_trace_credit_stage155 as source_spec
from scripts.run_native_transit_diagnostics_stage147 import check_episode
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_service_allocation_stage156_v1"
SOURCE_RUN = "native_transit_trace_credit_stage155_development_20261010_r1"
METHOD = "one_step"
ROOTS = source_spec.ROOTS
CONDITIONS = ("source_baseline", *PLAN_CONDITIONS)


def contract():
    source = source_spec.contract(False)
    return {"source_run": SOURCE_RUN, "source_protocol": source_spec.EXPERIMENT_PROTOCOL,
        "source_method": METHOD, "roots": list(ROOTS), "checkpoint_ep": 299,
        "training_updates": 0, "scenarios": source["scenarios"], "conditions": list(CONDITIONS),
        "scenes_per_regime": 4, "episodes_per_root": 120, "training_clock_s": source["training_clock_s"],
        "block": {"departures_per_direction": 6, "interval_bounds_s": [240, 480],
            "fixed_linear_interval_amplitude_s": 120, "endpoints": "fixed_nominal",
            "lower_goal": "committed_service_interval", "trip_count": "unchanged",
            "forecast": "current_causal_harmonic_inverse_sqrt_rate",
            "commitment": "each_block_once_before_first_departure_no_future_rewrite"},
        "baseline_gate": "all_twenty_source_baselines_then_all_twenty_source_neutral_episodes_exact",
        "checkpoint_choice": "one_step_both_registered_roots_no_performance_selection",
        "stage": "frozen_representation_authority_not_learned_policy_validation",
        "artifacts": "compact_JSON_server_only_checkpoint_reuse"}


def load_source(root):
    path = ROOT / "results" / SOURCE_RUN / "cells" / METHOD / f"seed_{root}/result.json"
    source = json.loads(path.read_text())
    if not (source["software_qualified"] and source["worker_preflight_passed"]
            and source["protocol"] == source_spec.EXPERIMENT_PROTOCOL
            and source["contract"] == source_spec.contract(False)
            and source["method"] == METHOD and source["seed"] == root
            and source["updates"] == source_spec.expected_updates(False)):
        raise ValueError("Service allocation requires the completed registered source checkpoint")
    return path, source


def reject_training(*args, **kwargs):
    raise RuntimeError("Frozen service-plan qualification attempted a training update")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(NATIVE))
    import torch
    import frequency
    frequency.DemandFrequencyTracker = NativeRoutingTracker
    from runner_v3 import TransitDuetV2Runner, load_config
    torch.set_num_threads(1)
    path, source = load_source(args.seed)
    checkpoint = path.parent.with_name(path.parent.name + "_raw") / "full/training" / (
        f"F_freqduet_harmonic_hiro_seed{args.seed}") / "checkpoints"
    references = {(r["condition"], r["scenario"], r["scene_seed"]): r for r in source["evaluation"]}
    spec, raw, rows = contract(), raw_directory(args.output), []
    for condition in CONDITIONS:
        for scenario in spec["scenarios"]:
            for scene in source_spec.scene_seeds(args.seed, scenario, preflight=False):
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                random.seed(args.seed)
                cfg = source_spec.configure(load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
                    METHOD, args.seed, preflight=False)
                cfg["env"].update(copy.deepcopy(source_spec.authority.routing.SCENARIOS[scenario]))
                cfg["logging"] = {"logs_dir": str(raw / condition / scenario / str(scene))}
                runner = TransitDuetV2Runner(cfg, device="cpu")
                runner.load_checkpoint(checkpoint, 299, require_deployment_state=True)
                before = model_arrays(runner)
                for level in ("upper", "lower"):
                    getattr(runner, f"{level}_trainer").update = reject_training
                plan = None
                if condition != "source_baseline":
                    plan = NativeServicePlan(runner.env, condition)
                    runner._upper_callback_v2 = plan
                row = runner.run_episode(300, training=False, N_fleet_override=12,
                    scenario_seed=scene, record_diagnostics=False)
                reference = references["baseline" if condition == "source_baseline" else "neutral_upper", scenario, scene]
                check_episode(row, reference, baseline=condition in {"source_baseline", "nominal_plan"})
                if len(runner.env.timetables) != reference["dispatch"]["queries"]:
                    raise RuntimeError("Service allocation changed the scheduled trip budget")
                if not all(np.isfinite(row[k]) for k in source_spec.authority.routing.METRICS):
                    raise RuntimeError("Nonfinite service-plan outcome")
                after = model_arrays(runner)
                if any(not np.array_equal(before[k], after[k]) for k in before):
                    raise RuntimeError("Frozen service-plan qualification changed a network")
                rows.append({"condition": condition, "scenario": scenario, "scene_seed": scene,
                    **source_spec.authority.routing.compact_row(row),
                    "lower_action_mean": row["lower_action_mean"], "scheduled_trips": len(runner.env.timetables),
                    "service_plan": plan.summarize() if plan else None,
                    "blocks": plan.blocks if plan else []})
                print(f"root={args.seed} {condition}/{scenario}/{scene} cost={row['service_cost_restricted']} "
                    f"wait={row['restricted_wait_horizon_min']} fleet={row['peak_fleet']}", flush=True)
                del runner, plan, before, after
        if condition in {"source_baseline", "nominal_plan"}:
            print(f"SERVICE_PLAN_SOURCE_REPRODUCED {condition}", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": spec, "method": METHOD,
        "seed": args.seed, "software_qualified": True, "baseline_reproduced": True,
        "neutral_reproduced": True, "training_updates": 0,
        "native_steps": len(rows) * spec["training_clock_s"], "evaluation": rows})
    print("NATIVE_SERVICE_PLAN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
