#!/usr/bin/env python3
"""Diagnose native upper opportunity on frozen Stage148 deployments."""

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
from freq_hrl.domains.transit.native_value_diagnostics import critic_action_curve, credit_ledger
from freq_hrl.domains.transit.native_routing import NativeRoutingTracker
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_authority_stage148 as source_spec
from scripts.run_native_transit_diagnostics_stage147 import check_episode
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_frontier_stage149_v1"
SOURCE_RUN = "native_transit_authority_stage148_development_20261009_r1"
METHODS = ("physical_lower", "physical_lower_service_credit")
ROOTS = source_spec.ROOTS
GOALS = (-60, -30, -15, 0, 15, 30, 60)
CACHED_GOALS = {-60: "upper_minus60", 0: "neutral_upper", 60: "upper_plus60"}
NEW_GOALS = (-30, -15, 15, 30)


def contract(preflight):
    return {"source_run": SOURCE_RUN, "source_protocol": source_spec.EXPERIMENT_PROTOCOL,
        "checkpoint_ep": 299, "training_updates": 0, "preflight": preflight,
        "methods": list(METHODS), "roots": list(ROOTS[:1] if preflight else ROOTS),
        "scenarios": ["low_noise"] if preflight else list(source_spec.routing.SCENARIOS),
        "new_conditions": ["baseline"] + ([] if preflight else list(NEW_GOALS)),
        "cached_goal_conditions": {str(goal): condition for goal, condition in CACHED_GOALS.items()},
        "critic_goals_s": list(GOALS), "scene_seeds": "reuse_registered_stage148_scenes",
        "baseline_gate": "exact_source_outcome_reproduction",
        "purpose": "post_result_control_opportunity_and_value_diagnosis_not_independent_confirmation",
        "oracle": "retrospective_grid_minimum_not_a_deployable_policy",
        "artifacts": "compact_JSON_only_no_checkpoint_download"}


def load_source(method, root):
    path = ROOT / "results" / SOURCE_RUN / "cells" / method / f"seed_{root}" / "result.json"
    source = json.loads(path.read_text())
    if not (source["software_qualified"] and source["method"] == method and source["seed"] == root
            and source["protocol"] == source_spec.EXPERIMENT_PROTOCOL
            and source["contract"] == source_spec.contract(False)):
        raise ValueError("Frozen frontier requires its completed Stage148 source cell")
    return path, source


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
    source_path, source = load_source(args.method, args.seed)
    checkpoint = source_path.parent.with_name(source_path.parent.name + "_raw") / "training" / (
        f"F_freqduet_harmonic_hiro_seed{args.seed}") / "checkpoints"
    raw = raw_directory(args.output)
    spec = contract(args.preflight)
    rows = []
    for scenario in spec["scenarios"]:
        reference = next(row for row in source["evaluation"]
                         if row["scenario"] == scenario and row["condition"] == "baseline")
        for condition in spec["new_conditions"]:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            cfg = source_spec.configure(load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
                args.method, args.seed, preflight=False)
            cfg["env"].update(copy.deepcopy(source_spec.routing.SCENARIOS[scenario]))
            cfg["logging"] = {"logs_dir": str(raw / str(condition) / scenario)}
            runner = TransitDuetV2Runner(cfg, device="cpu")
            runner.load_checkpoint(checkpoint, 299, require_deployment_state=True)
            before = model_arrays(runner)
            probe = NativePolicyProbe(runner.upper_trainer.policy_net, "upper",
                fixed_action_s=None if condition == "baseline" else float(condition))
            probe.policy.get_action = probe.get_action
            row = runner.run_episode(300, training=False, N_fleet_override=12,
                scenario_seed=reference["scene_seed"], record_diagnostics=False)
            check_episode(row, reference, baseline=condition == "baseline")
            for key in ("lower_action_mean", "upper_delta_mean"):
                if condition == "baseline" and row[key] != reference[key]:
                    raise RuntimeError(f"Passive frontier changed baseline {key}")
            result = {"condition": condition, "scenario": scenario, "scene_seed": reference["scene_seed"],
                **source_spec.routing.compact_row(row),
                **{key: row[key] for key in ("lower_action_mean", "upper_delta_mean")}}
            if condition == "baseline":
                result["critic_curve"] = critic_action_curve(runner.upper_trainer, probe.states, GOALS)
                result["credit_ledger"] = credit_ledger(runner._episode_upper_transitions)
            after = model_arrays(runner)
            if any(not np.array_equal(before[key], after[key]) for key in before):
                raise RuntimeError("Frozen frontier changed a network")
            rows.append(result)
            probe.policy.get_action = probe.original_get_action
            print(f"{args.method} root={args.seed} {condition}/{scenario} "
                  f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                  f"peak_fleet={row['peak_fleet']}", flush=True)
            del runner, probe, before, after
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": spec,
        "method": args.method, "seed": args.seed, "software_qualified": True,
        "baseline_reproduced": True, "training_updates": 0,
        "native_steps": len(rows) * 61380, "evaluation": rows})
    print("NATIVE_FRONTIER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
