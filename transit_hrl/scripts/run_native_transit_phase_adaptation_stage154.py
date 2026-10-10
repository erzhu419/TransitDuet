#!/usr/bin/env python3
"""Separate frozen learned dispatch adaptation from a constant phase shift."""

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
from scripts import run_native_transit_critic_regularization_stage153 as source_spec
from scripts.run_native_transit_diagnostics_stage147 import check_episode
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_phase_adaptation_stage154_v1"
SOURCE_RUN = "native_transit_critic_regularization_stage153_development_20261010_r1"
METHOD = "unit_physical"
ROOTS = source_spec.ROOTS
CONDITIONS = ("baseline", "constant_mean", "fixed5", "fixed7", "fixed9")
METRICS = (*source_spec.authority.routing.METRICS, "lower_action_mean", "upper_delta_mean")


def contract():
    source = source_spec.contract(False)
    return {"source_run": SOURCE_RUN, "source_protocol": source_spec.EXPERIMENT_PROTOCOL,
        "source_method": METHOD, "roots": list(ROOTS), "checkpoint_ep": 299,
        "training_updates": 0, "scenarios": source["scenarios"],
        "scene_seeds": "all_twenty_registered_stage153_scenes_per_root",
        "conditions": list(CONDITIONS), "constant_mean": "source_baseline_actor_call_weighted_mean_once_per_root",
        "fixed_commands_s": [5, 7, 9], "baseline_gate": "all_source_baselines_reproduced_before_interventions",
        "training_clock_s": source["training_clock_s"], "episodes_per_root": 100,
        "purpose": "post_result_phase_vs_adaptation_diagnosis_not_independent_confirmation",
        "unchanged": ["checkpoint", "lower_policy", "reward", "critic", "bounds", "dispatch_execution"],
        "artifacts": "compact_JSON_only_server_checkpoint_reuse"}


def validate_source(source, root):
    if not (source["software_qualified"] and source["worker_preflight_passed"]
            and source["protocol"] == source_spec.EXPERIMENT_PROTOCOL
            and source["contract"] == source_spec.contract(False)
            and source["method"] == METHOD and source["seed"] == root
            and source["updates"] == source_spec.expected_updates(False)
            and source["critic_action_units"] == "unit" and source["weight_reg_mode"] == "physical_sum"):
        raise ValueError("Phase diagnostic requires completed physical-L1 native training")
    required = {(condition, scenario, scene) for condition in source_spec.CONDITIONS
        for scenario in contract()["scenarios"]
        for scene in source_spec.scene_seeds(root, scenario, preflight=False)}
    keys = [(r["condition"], r["scenario"], r["scene_seed"]) for r in source["evaluation"]]
    if set(keys) != required or len(keys) != len(required):
        raise ValueError("Incomplete or duplicate source scene ledger")


def load_source(root):
    path = ROOT / "results" / SOURCE_RUN / "cells" / METHOD / f"seed_{root}" / "result.json"
    source = json.loads(path.read_text())
    validate_source(source, root)
    return path, source


def constant_actions(source):
    rows = [row["actors"]["upper"] for row in source["evaluation"] if row["condition"] == "baseline"]
    mean = float(np.average([r["proposal_mean_s"] for r in rows], weights=[r["actor_calls"] for r in rows]))
    if not np.isfinite(mean) or not -120 <= mean <= 120:
        raise ValueError("Invalid source-wide mean physical command")
    return {"constant_mean": mean, "fixed5": 5.0, "fixed7": 7.0, "fixed9": 9.0}


def check_row(row, reference, condition):
    check_episode(row, reference, baseline=condition == "baseline")
    if not all(np.isfinite(row[key]) for key in METRICS):
        raise RuntimeError("Nonfinite frozen phase outcome")
    if condition == "baseline":
        for field in ("lower_action_mean", "upper_delta_mean"):
            if row[field] != reference[field]:
                raise RuntimeError(f"Phase baseline failed source reproduction: {field}")


def reject_training(*args, **kwargs):
    raise RuntimeError("Frozen phase diagnostic attempted a training update")


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
    source_path, source = load_source(args.seed)
    # Stage153 stores qualification and fresh full training in separate raw trees.
    checkpoint = source_path.parent.with_name(source_path.parent.name + "_raw") / "full/training" / (
        f"F_freqduet_harmonic_hiro_seed{args.seed}") / "checkpoints"
    constants = constant_actions(source)
    references = {(r["scenario"], r["scene_seed"]): r for r in source["evaluation"] if r["condition"] == "baseline"}
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
                probe = NativePolicyProbe(runner.upper_trainer.policy_net, "upper",
                    fixed_action_s=constants.get(condition))
                probe.policy.get_action = probe.get_action
                row = runner.run_episode(300, training=False, N_fleet_override=12,
                    scenario_seed=scene, record_diagnostics=False)
                check_row(row, references[scenario, scene], condition)
                after = model_arrays(runner)
                if any(not np.array_equal(before[key], after[key]) for key in before):
                    raise RuntimeError("Frozen phase evaluation changed a network")
                rows.append({"condition": condition, "scenario": scenario, "scene_seed": scene,
                    "fixed_action_s": constants.get(condition),
                    **source_spec.authority.routing.compact_row(row),
                    **{key: row[key] for key in ("lower_action_mean", "upper_delta_mean")}})
                probe.policy.get_action = probe.original_get_action
                print(f"root={args.seed} {condition}/{scenario}/{scene} cost={row['service_cost_restricted']} "
                    f"wait={row['restricted_wait_horizon_min']} fleet={row['peak_fleet']}", flush=True)
                del runner, probe, before, after
        if condition == "baseline":
            print("PHASE_SOURCE_BASELINES_REPRODUCED", flush=True)
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": spec,
        "method": METHOD, "seed": args.seed, "software_qualified": True, "baseline_reproduced": True,
        "training_updates": 0, "constant_actions_s": constants,
        "native_steps": len(rows) * spec["training_clock_s"], "evaluation": rows})
    print("NATIVE_PHASE_ADAPTATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
