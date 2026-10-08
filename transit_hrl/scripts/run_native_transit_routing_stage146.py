#!/usr/bin/env python3
"""Train matched native frequency-routing controls with the preserved RE-SAC."""

import argparse
import copy
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from freq_hrl.domains.transit.native_routing import METHODS, NativeRoutingTracker
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_routing_stage146_v1"
ROOTS = (101, 113, 127, 139, 151, 163, 179, 191)
METRICS = (
    "service_cost_restricted", "restricted_wait_horizon_min", "ep_reward",
    "avg_wait_observed_min", "headway_cv", "peak_fleet",
    "passenger_unserved_rate", "trip_completion_rate",
    "upper_hf_power_ratio", "lower_lf_drift_ratio",
)
SCENARIOS = {
    "low_noise": {"demand_noise": 0.0, "route_sigma": 1.5, "peak_shift_choices": [0]},
    "high_noise": {"demand_noise": 0.3, "route_sigma": 2.0, "peak_shift_choices": [0]},
    "hour_burst": {"demand_noise": 0.15, "route_sigma": 1.5, "peak_shift_choices": [0],
                   "demand_hourly_multipliers": {10: 2.0}},
    "persistent_shift": {"demand_noise": 0.15, "route_sigma": 1.5, "peak_shift_choices": [0],
                         "demand_hourly_multipliers": {hour: 1.5 for hour in range(13, 20)}},
    "ood_period": {"demand_noise": 0.15, "route_sigma": 1.5, "peak_shift_choices": [2]},
}


def contract(preflight):
    return {
        "methods": list(METHODS), "train_episodes": 2 if preflight else 300,
        "upper_warmup": 1 if preflight else 30,
        "training_clock_s": 5400 if preflight else 60000,
        "demand_end_time_s": 3600 if preflight else 50400,
        "preflight": preflight,
        "preflight_overrides": {"effective_trip_num": 24, "service_end_hour": 7,
            "upper_batch_size": 8, "lower_batch_size": 32, "updates_per_episode": 2} if preflight else {},
        "evaluation_scenarios": ["low_noise"] if preflight else list(SCENARIOS),
        "evaluation_episodes_per_scenario": 1 if preflight else 4,
        "evaluation_fleet": 12, "checkpoint": "last_training_episode_no_selection",
        "shared_upper_slots": ["LF_forecast", "HF_energy", "OD_entropy", "OD_HF_energy"],
        "actor_dims": {"upper": 16, "lower": 33},
        "primary": "equal_scenario_mean_service_cost_restricted_correct_minus_control",
        "primary_controls": ["raw_history_common", "swapped_common"],
        "statistics": "paired_optimizer_root_bootstrap_10000_two_contrasts_Bonferroni",
        "independence_unit": "optimizer_root_with_disjoint_paired_eval_scenarios",
        "stage": "development_not_independent_confirmation",
        "artifacts": "compact_JSON_and_server_only_final_inference_checkpoint",
    }


def configure(base, method, seed, *, preflight):
    cfg = copy.deepcopy(base)
    spec = contract(preflight)
    cfg["seed"] = seed
    cfg["frequency"]["routing"] = method
    cfg["env"].update(allow_early_finish=False,
                      demand_end_time_s=spec["demand_end_time_s"],
                      evaluation_end_time_s=spec["training_clock_s"])
    if preflight:
        cfg["env"].update(effective_trip_num=24, service_end_hour=7)
        cfg["coupling"]["upper_warmup_eps"] = 1
        for level, batch in (("upper", 8), ("lower", 32)):
            cfg[level].update(batch_size=batch, updates_per_episode=2)
    return cfg


def evaluation_seeds(root, scenario, *, preflight):
    index = list(SCENARIOS).index(scenario)
    return [900000000 + root * 100 + index * 10 + i
            for i in range(contract(preflight)["evaluation_episodes_per_scenario"])]


def compact_row(row):
    return {key: row[key] for key in (
        "ep", "N_fleet", "simulation_end_time_s", "done_reason", "passengers_generated",
        "passengers_unserved", "trips_completed", "ep_steps", *METRICS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(NATIVE))
    import torch
    import frequency
    frequency.DemandFrequencyTracker = NativeRoutingTracker
    from runner_v3 import TransitDuetV2Runner, load_config
    torch.set_num_threads(1)

    def seed_runtime():
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    seed_runtime()
    cfg = configure(load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
                    args.method, args.seed, preflight=args.preflight)
    raw = raw_directory(args.output)
    cfg["logging"] = {"logs_dir": str(raw / "training")}
    runner = TransitDuetV2Runner(cfg, device="cpu")
    spec = contract(args.preflight)
    dims = {"upper": runner.upper_state_dim, "lower": runner.lower_state_dim}
    if dims != spec["actor_dims"]:
        raise RuntimeError(f"Native actor geometry changed: {dims}")
    if runner.env.frequency_tracker.routing != args.method:
        raise RuntimeError("Native simulator did not bind the registered routing control")
    initial = model_arrays(runner)
    parameter_counts = {
        level: {name: sum(p.numel() for p in getattr(getattr(runner, f"{level}_trainer"), name).parameters())
                for name in ("policy_net", "q_net")}
        for level in ("upper", "lower")
    }
    updates = {"upper": 0, "lower": 0}
    for level in updates:
        trainer = getattr(runner, f"{level}_trainer")
        original_update = trainer.update
        def counted(*pos, _level=level, _update=original_update, **kw):
            updates[_level] += 1
            return _update(*pos, **kw)
        trainer.update = counted
    training_curve, training_demand = [], []
    total_steps = 0
    for ep in range(spec["train_episodes"]):
        row = runner.run_episode(ep, training=True)
        if row["simulation_end_time_s"] != spec["training_clock_s"]:
            raise RuntimeError("Policy-dependent training episode clock")
        if not all(np.isfinite(row[k]) for k in METRICS):
            raise RuntimeError("Nonfinite native training metric")
        training_demand.append(row["passengers_generated"])
        total_steps += row["simulation_end_time_s"] // runner.env.time_step
        if ep % 10 == 0 or ep in (spec["upper_warmup"] - 1, spec["train_episodes"] - 1):
            training_curve.append(compact_row(row))
            print(f"{args.method} seed={args.seed} train={ep + 1}/{spec['train_episodes']} "
                  f"cost={row['service_cost_restricted']} updates={updates}", flush=True)
    final = model_arrays(runner)
    actor_change = {
        level: max(float(np.max(np.abs(final[k] - initial[k])))
                   for k in final if k.startswith(f"{level}/policy_net/"))
        for level in updates
    }
    expected_updates = {
        "upper": (spec["train_episodes"] - spec["upper_warmup"]) * cfg["upper"]["updates_per_episode"],
        "lower": spec["train_episodes"] * cfg["lower"]["updates_per_episode"],
    }
    if updates != expected_updates or not all(v > 0 for v in actor_change.values()):
        raise RuntimeError(f"Native learning budget was not met: {updates}, {actor_change}")
    runner._save_checkpoint(spec["train_episodes"] - 1)
    checkpoint_dir = Path(runner.log_dir) / "checkpoints"
    del runner, initial, final

    evaluation = []
    for scenario in spec["evaluation_scenarios"]:
        for scene_seed in evaluation_seeds(args.seed, scenario, preflight=args.preflight):
            # Each episode starts from the same trained deployment state.
            seed_runtime()
            eval_cfg = copy.deepcopy(cfg)
            eval_cfg["env"].update(SCENARIOS[scenario])
            eval_cfg["logging"] = {"logs_dir": str(raw / "evaluation")}
            evaluator = TransitDuetV2Runner(eval_cfg, device="cpu")
            evaluator.load_checkpoint(checkpoint_dir, spec["train_episodes"] - 1,
                                      require_deployment_state=True)
            before = model_arrays(evaluator)
            row = evaluator.run_episode(spec["train_episodes"], training=False,
                                        N_fleet_override=12, scenario_seed=scene_seed,
                                        record_diagnostics=False)
            after = model_arrays(evaluator)
            if any(not np.array_equal(before[k], after[k]) for k in before):
                raise RuntimeError("Frozen native evaluation updated a network")
            if row["simulation_end_time_s"] != spec["training_clock_s"]:
                raise RuntimeError("Policy-dependent evaluation episode clock")
            if not all(np.isfinite(row[k]) for k in METRICS):
                raise RuntimeError("Nonfinite native evaluation metric")
            evaluation.append({"scenario": scenario, "scene_seed": scene_seed, **compact_row(row)})
            total_steps += row["simulation_end_time_s"] // evaluator.env.time_step
            del evaluator, before, after
    write_json(args.output, {
        "protocol": EXPERIMENT_PROTOCOL, "contract": spec, "method": args.method,
        "seed": args.seed, "passed": True, "actor_dims": dims, "parameter_counts": parameter_counts,
        "updates": updates, "actor_change_max_abs": actor_change,
        "native_steps": int(total_steps), "training_demand_counts": training_demand,
        "training_curve": training_curve, "evaluation": evaluation,
    })
    print("NATIVE_ROUTING_CELL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
