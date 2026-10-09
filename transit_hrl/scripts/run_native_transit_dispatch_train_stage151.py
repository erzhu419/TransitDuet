#!/usr/bin/env python3
"""Train native HIRO/signed-dispatch policies under a matched credit factorial."""

import argparse
import copy
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_diagnostics import NativePolicyProbe
from freq_hrl.domains.transit.native_routing import NativeRoutingTracker
from freq_hrl.domains.transit.native_value_diagnostics import credit_ledger
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_authority_stage148 as authority
from scripts.run_native_transit_dispatch_stage150 import dispatch_ledger
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_dispatch_training_stage151_v1"
ROOTS = (241, 257, 269, 281)
METHODS = {
    "hiro": ("hiro", "physical_lower"),
    "hiro_service_credit": ("hiro", "physical_lower_service_credit"),
    "dispatch": ("channels", "physical_lower"),
    "dispatch_service_credit": ("channels", "physical_lower_service_credit"),
}
CONDITIONS = ("baseline", "neutral_upper", "zero_holding")


def contract(preflight):
    native = authority.routing.contract(preflight)
    return {"methods": {key: {"coupling": mode, "authority_factor": factor}
                         for key, (mode, factor) in METHODS.items()},
        "roots": list(ROOTS), "preflight": preflight,
        **{key: native[key] for key in ("train_episodes", "upper_warmup", "training_clock_s",
            "demand_end_time_s", "preflight_overrides", "actor_dims", "evaluation_episodes_per_scenario")},
        "scenarios": native["evaluation_scenarios"], "conditions": list(CONDITIONS),
        "frequency_routing": "correct", "dispatch_commitment_s": 120,
        "decision_clock": {"hiro": "nominal_launch", "channels": "max_zero_nominal_minus_120s"},
        "initialization": "fresh_native_RE_SAC_same_seed_no_checkpoint_warmstart",
        "checkpoint": "last_training_episode_no_selection",
        "worker_preflight": "short_two_level_learning_then_fresh_seed_reset_for_full_training",
        "primary": "equal_regime_mean_service_cost_restricted",
        "upper_authority": "baseline_minus_same_checkpoint_zero_upper_physical_outcomes",
        "credit": {"wait_weight": 1, "fleet_weight": 1, "headway_weight": 0, "scale": 100},
        "unchanged": ["physical_lower_encoder", "RE_SAC", "networks", "action_bounds",
            "historical_priors", "frequency_estimator", "lower_reward", "fleet_cap", "leakage_penalties"],
        "statistics": "four_root_descriptive_factorial_before_independent_confirmation",
        "artifacts": "compact_JSON_server_only_last_checkpoints"}


def configure(base, method, root, *, preflight):
    mode, factor = METHODS[method]
    cfg = authority.configure(base, factor, root, preflight=preflight)
    cfg["coupling"]["coupling_mode"] = mode
    return cfg


def scene_seeds(root, scenario, *, preflight):
    return authority.routing.evaluation_seeds(root, scenario, preflight=preflight)


def expected_updates(preflight):
    spec = contract(preflight)
    return {"upper": (spec["train_episodes"] - spec["upper_warmup"]) * (2 if preflight else 10),
            "lower": spec["train_episodes"] * (2 if preflight else 30)}


def run_cell(method, root, output, *, preflight):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(NATIVE))
    import torch
    import frequency
    frequency.DemandFrequencyTracker = NativeRoutingTracker
    from runner_v3 import TransitDuetV2Runner, load_config
    torch.set_num_threads(1)

    def seed_runtime():
        torch.manual_seed(root)
        np.random.seed(root)
        random.seed(root)

    seed_runtime()
    spec = contract(preflight)
    mode, factor = METHODS[method]
    cfg = configure(load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
                    method, root, preflight=preflight)
    raw = raw_directory(output) / ("qualification" if preflight else "full")
    cfg["logging"] = {"logs_dir": str(raw / "training")}
    runner = TransitDuetV2Runner(cfg, device="cpu")
    dims = {"upper": runner.upper_state_dim, "lower": runner.lower_state_dim}
    if dims != spec["actor_dims"] or runner.env.frequency_tracker.routing != "correct" or runner.coupling_mode != mode:
        raise RuntimeError("Changed native geometry or registered control mode")
    initial = model_arrays(runner)
    parameter_counts = {level: {name: sum(p.numel() for p in
        getattr(getattr(runner, f"{level}_trainer"), name).parameters()) for name in ("policy_net", "q_net")}
        for level in ("upper", "lower")}
    updates = {"upper": 0, "lower": 0}
    for level in updates:
        trainer = getattr(runner, f"{level}_trainer")
        update = trainer.update
        def counted(*pos, _level=level, _update=update, **kw):
            updates[_level] += 1
            return _update(*pos, **kw)
        trainer.update = counted
    curve, demand, fleets = [], [], []
    for ep in range(spec["train_episodes"]):
        row = runner.run_episode(ep, training=True)
        if row["simulation_end_time_s"] != spec["training_clock_s"]:
            raise RuntimeError("Policy-dependent training clock")
        if not all(np.isfinite(row[key]) for key in authority.routing.METRICS):
            raise RuntimeError("Nonfinite native training outcome")
        credit = authority.credit_row(row, factor)
        demand.append(row["passengers_generated"])
        fleets.append(row["N_fleet"])
        if ep % 10 == 0 or ep in (spec["upper_warmup"] - 1, spec["train_episodes"] - 1):
            curve.append({**authority.routing.compact_row(row), **credit})
            print(f"{method} root={root} preflight={preflight} train={ep+1}/{spec['train_episodes']} "
                  f"cost={row['service_cost_restricted']} updates={updates}", flush=True)
    final = model_arrays(runner)
    actor_change = {level: max(float(np.max(np.abs(final[key] - initial[key])))
        for key in final if key.startswith(f"{level}/policy_net/")) for level in updates}
    if updates != expected_updates(preflight) or not all(value > 0 for value in actor_change.values()):
        raise RuntimeError(f"Incomplete two-level learning: {updates}, {actor_change}")
    training_credit = credit_ledger(runner._episode_upper_transitions)
    runner._save_checkpoint(spec["train_episodes"] - 1)
    checkpoint = Path(runner.log_dir) / "checkpoints"
    del runner, initial, final

    evaluations = []
    for scenario in spec["scenarios"]:
        for scene_seed in scene_seeds(root, scenario, preflight=preflight):
            reference = None
            for condition in CONDITIONS:
                seed_runtime()
                eval_cfg = copy.deepcopy(cfg)
                eval_cfg["env"].update(copy.deepcopy(authority.routing.SCENARIOS[scenario]))
                eval_cfg["logging"] = {"logs_dir": str(raw / condition / scenario / str(scene_seed))}
                evaluator = TransitDuetV2Runner(eval_cfg, device="cpu")
                evaluator.load_checkpoint(checkpoint, spec["train_episodes"] - 1, require_deployment_state=True)
                before = model_arrays(evaluator)
                probes = {level: NativePolicyProbe(getattr(evaluator, f"{level}_trainer").policy_net, level,
                    neutral=(condition == "neutral_upper" and level == "upper")
                        or (condition == "zero_holding" and level == "lower")) for level in ("upper", "lower")}
                for probe in probes.values():
                    probe.policy.get_action = probe.get_action
                callback, queries = evaluator._upper_callback_v2, []
                def record_query(state, trip):
                    queries.append((trip.launch_turn, float(evaluator.env.current_time)))
                    return callback(state, trip)
                evaluator._upper_callback_v2 = record_query
                row = evaluator.run_episode(spec["train_episodes"], training=False, N_fleet_override=12,
                    scenario_seed=scene_seed, record_diagnostics=False)
                if row["simulation_end_time_s"] != spec["training_clock_s"]:
                    raise RuntimeError("Policy-dependent evaluation clock")
                if reference is None:
                    reference = row
                for key in ("simulation_end_time_s", "passengers_generated", "N_fleet"):
                    if row[key] != reference[key]:
                        raise RuntimeError(f"Unpaired physical intervention: {key}")
                if not all(np.isfinite(row[key]) for key in authority.routing.METRICS):
                    raise RuntimeError("Nonfinite frozen outcome")
                diagnostics = {level: probe.summarize() for level, probe in probes.items()}
                ledger = dispatch_ledger(evaluator.env, queries)
                if ledger["advance_count"] and (mode == "hiro" or condition == "neutral_upper"):
                    raise RuntimeError("Unexpected advance without a signed dispatch command")
                commands = np.asarray(evaluator._ep_upper_deltas)
                after = model_arrays(evaluator)
                if any(not np.array_equal(before[key], after[key]) for key in before):
                    raise RuntimeError("Frozen evaluation changed a network")
                evaluations.append({"condition": condition, "scenario": scenario, "scene_seed": scene_seed,
                    **authority.routing.compact_row(row), **authority.credit_row(row, factor),
                    "actors": diagnostics, "dispatch": ledger,
                    "command_abs_mean_s": float(np.mean(np.abs(commands))),
                    "subsecond_command_fraction": float(np.mean(np.abs(commands) < 1)),
                    "credit_ledger": credit_ledger(evaluator._episode_upper_transitions),
                    **{key: row[key] for key in ("lower_action_mean", "upper_delta_mean")}})
                print(f"{method} root={root} {condition}/{scenario}/{scene_seed} "
                      f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                      f"advance={ledger['advance_count']} delay={ledger['delay_count']}", flush=True)
                for probe in probes.values():
                    probe.policy.get_action = probe.original_get_action
                del evaluator, probes, before, after
    result = {"protocol": EXPERIMENT_PROTOCOL, "contract": spec, "method": method, "seed": root,
        "software_qualified": True, "actor_dims": dims, "parameter_counts": parameter_counts,
        "updates": updates, "actor_change_max_abs": actor_change,
        "native_steps": (spec["train_episodes"] + len(evaluations)) * spec["training_clock_s"],
        "training_demand_counts": demand, "training_fleets": fleets, "training_curve": curve,
        "last_training_credit_ledger": training_credit, "evaluation": evaluations}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=ROOTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Worker qualification is separate from, and never warmstarts, full training.
    preflight = run_cell(args.method, args.seed, args.output.with_name("preflight.json"), preflight=True)
    result = run_cell(args.method, args.seed, args.output, preflight=False)
    result.update(worker_preflight_passed=True, worker_preflight_native_steps=preflight["native_steps"])
    write_json(args.output, result)
    print("NATIVE_DISPATCH_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
