#!/usr/bin/env python3
"""Cross native lower goal conditioning with physical upper interval credit."""

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
from freq_hrl.experiments.pointmaze_root_response import raw_directory, write_json
from scripts import run_native_transit_routing_stage146 as routing
from scripts.run_native_transit_preservation_stage145 import NATIVE, model_arrays

EXPERIMENT_PROTOCOL = "native_transit_authority_stage148_v1"
METHODS = ("legacy", "physical_lower", "service_credit", "physical_lower_service_credit")
ROOTS = (217, 229)
CONDITIONS = ("baseline", "neutral_upper", "upper_minus60", "upper_plus60", "zero_holding")
CREDIT_FIELDS = ("upper_interval_reward_sum", "upper_interval_wait_cost_sum",
                "upper_interval_fleet_cost_sum", "upper_interval_coverage_mean")


def contract(preflight):
    native = routing.contract(preflight)
    return {
        "methods": list(METHODS), "roots": list(ROOTS[:1] if preflight else ROOTS),
        **{key: native[key] for key in ("train_episodes", "upper_warmup", "training_clock_s",
            "demand_end_time_s", "preflight_overrides", "actor_dims")},
        "preflight": preflight, "routing": "correct", "coupling": "hiro",
        "conditions": list(CONDITIONS),
        "scenarios": ["low_noise"] if preflight else list(routing.SCENARIOS),
        "scene_seeds": "stage148_disjoint_roots_one_paired_scene_per_regime",
        "lower_factor": "physical_dimensionless_v1_explicit_target_v2_not_scale_only",
        "credit_factor": {"assignment": "additive_global_intervals", "wait_weight": 1.0,
            "fleet_weight": 1.0, "headway_weight": 0.0, "reward_scale": 100.0,
            "system_reward_mode": "none", "gap_credit_mode": "none"},
        "unchanged": ["RE_SAC", "action_bounds", "networks", "frequency_estimator",
            "HF_and_drift_penalties", "lower_reward", "fixed_dispatch_schedule"],
        "checkpoint": "last_training_episode_no_selection",
        "decision": "physical_upper_authority_and_service_tradeoff_before_frequency_sweep",
        "statistics": "two_root_descriptive_factorial_no_supported_performance_claim",
        "artifacts": "compact_JSON_server_only_final_checkpoint",
    }


def configure(base, method, seed, *, preflight):
    cfg = routing.configure(base, "correct", seed, preflight=preflight)
    if method in ("physical_lower", "physical_lower_service_credit"):
        cfg["lower"]["state_encoder"] = {
            "enable": True, "mode": "physical_dimensionless_v1", "input_schema": "explicit_target_v2"}
    if method in ("service_credit", "physical_lower_service_credit"):
        # Keep one global stream: no directional double-counting or stream change.
        cfg["upper"]["credit_assignment"] = {"system_reward_mode": "none", "gap_credit_mode": "none"}
        cfg["upper"]["interval_credit"] = {
            "enable": True, "assignment_mode": "additive", "reward_scale": 100.0,
            "weights": {"wait": 1.0, "headway": 0.0, "fleet": 1.0}}
    return cfg


def credit_row(row, method):
    values = {key: row[key] for key in CREDIT_FIELDS}
    if method in ("service_credit", "physical_lower_service_credit"):
        expected = -100.0 * (values["upper_interval_wait_cost_sum"] + values["upper_interval_fleet_cost_sum"])
        if not np.isclose(values["upper_interval_reward_sum"], expected, atol=0.00011, rtol=0):
            raise RuntimeError("Upper interval reward does not match physical wait/fleet credit")
    return values


def scene_seed(root, scenario):
    return routing.evaluation_seeds(root, scenario, preflight=False)[0]


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

    def seed_runtime():
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    seed_runtime()
    spec = contract(args.preflight)
    cfg = configure(load_config(str(NATIVE / "configs_freqduet/F_freqduet_harmonic_hiro.yaml")),
                    args.method, args.seed, preflight=args.preflight)
    raw = raw_directory(args.output)
    cfg["logging"] = {"logs_dir": str(raw / "training")}
    runner = TransitDuetV2Runner(cfg, device="cpu")
    dims = {"upper": runner.upper_state_dim, "lower": runner.lower_state_dim}
    if dims != spec["actor_dims"] or runner.env.frequency_tracker.routing != "correct":
        raise RuntimeError("Native geometry or fixed frequency routing changed")
    initial = model_arrays(runner)
    parameter_counts = {
        level: {name: sum(p.numel() for p in getattr(getattr(runner, f"{level}_trainer"), name).parameters())
                for name in ("policy_net", "q_net")}
        for level in ("upper", "lower")}
    updates = {"upper": 0, "lower": 0}
    for level in updates:
        trainer = getattr(runner, f"{level}_trainer")
        update = trainer.update
        def counted(*pos, _level=level, _update=update, **kw):
            updates[_level] += 1
            return _update(*pos, **kw)
        trainer.update = counted
    curve, demand, total_steps = [], [], 0
    for ep in range(spec["train_episodes"]):
        row = runner.run_episode(ep, training=True)
        if row["simulation_end_time_s"] != spec["training_clock_s"]:
            raise RuntimeError("Policy-dependent training clock")
        if not all(np.isfinite(row[key]) for key in routing.METRICS):
            raise RuntimeError("Nonfinite native training metric")
        credit = credit_row(row, args.method)
        demand.append(row["passengers_generated"])
        total_steps += int(row["simulation_end_time_s"] / runner.env.time_step)
        if ep % 10 == 0 or ep in (spec["upper_warmup"] - 1, spec["train_episodes"] - 1):
            curve.append({**routing.compact_row(row), **credit})
            print(f"{args.method} root={args.seed} train={ep+1}/{spec['train_episodes']} "
                  f"cost={row['service_cost_restricted']} updates={updates} "
                  f"interval_reward={credit['upper_interval_reward_sum']}", flush=True)
    final = model_arrays(runner)
    actor_change = {level: max(float(np.max(np.abs(final[key] - initial[key])))
        for key in final if key.startswith(f"{level}/policy_net/")) for level in updates}
    expected_updates = {
        "upper": (spec["train_episodes"] - spec["upper_warmup"]) * cfg["upper"]["updates_per_episode"],
        "lower": spec["train_episodes"] * cfg["lower"]["updates_per_episode"]}
    if updates != expected_updates or not all(value > 0 for value in actor_change.values()):
        raise RuntimeError(f"Learning budget not met: {updates}, {actor_change}")
    runner._save_checkpoint(spec["train_episodes"] - 1)
    checkpoint = Path(runner.log_dir) / "checkpoints"
    del runner, initial, final

    evaluations = []
    for scenario in spec["scenarios"]:
        reference = None
        for condition in CONDITIONS:
            seed_runtime()
            eval_cfg = copy.deepcopy(cfg)
            eval_cfg["env"].update(copy.deepcopy(routing.SCENARIOS[scenario]))
            eval_cfg["logging"] = {"logs_dir": str(raw / condition / scenario)}
            evaluator = TransitDuetV2Runner(eval_cfg, device="cpu")
            evaluator.load_checkpoint(checkpoint, spec["train_episodes"] - 1, require_deployment_state=True)
            before = model_arrays(evaluator)
            probes = {level: NativePolicyProbe(getattr(evaluator, f"{level}_trainer").policy_net, level,
                neutral=(condition == "neutral_upper" and level == "upper") or (
                    condition == "zero_holding" and level == "lower"),
                fixed_action_s=({"upper_minus60": -60.0, "upper_plus60": 60.0}.get(condition)
                                if level == "upper" else None)) for level in ("upper", "lower")}
            for probe in probes.values():
                probe.policy.get_action = probe.get_action
            row = evaluator.run_episode(spec["train_episodes"], training=False,
                N_fleet_override=12, scenario_seed=scene_seed(args.seed, scenario), record_diagnostics=False)
            if row["simulation_end_time_s"] != spec["training_clock_s"]:
                raise RuntimeError("Policy-dependent evaluation clock")
            if reference is None:
                reference = row
            for key in ("simulation_end_time_s", "passengers_generated", "N_fleet"):
                if row[key] != reference[key]:
                    raise RuntimeError(f"Unpaired physical intervention: {key}")
            if not all(np.isfinite(row[key]) for key in routing.METRICS):
                raise RuntimeError("Nonfinite frozen metric")
            diagnostics = {level: probe.summarize() for level, probe in probes.items()}
            after = model_arrays(evaluator)
            if any(not np.array_equal(before[key], after[key]) for key in before):
                raise RuntimeError("Frozen evaluation changed a network")
            evaluations.append({"scenario": scenario, "condition": condition,
                "scene_seed": scene_seed(args.seed, scenario), **routing.compact_row(row),
                **credit_row(row, args.method), "actors": diagnostics,
                **{key: row[key] for key in ("lower_action_mean", "upper_delta_mean")}})
            total_steps += int(row["simulation_end_time_s"] / evaluator.env.time_step)
            print(f"{args.method} root={args.seed} {condition}/{scenario} "
                  f"cost={row['service_cost_restricted']} wait={row['restricted_wait_horizon_min']} "
                  f"holding={row['lower_action_mean']} upper={row['upper_delta_mean']}", flush=True)
            for probe in probes.values():
                probe.policy.get_action = probe.original_get_action
            del evaluator, probes, before, after
    write_json(args.output, {"protocol": EXPERIMENT_PROTOCOL, "contract": spec,
        "method": args.method, "seed": args.seed, "software_qualified": True,
        "actor_dims": dims, "parameter_counts": parameter_counts, "updates": updates,
        "actor_change_max_abs": actor_change, "native_steps": total_steps,
        "training_demand_counts": demand, "training_curve": curve, "evaluation": evaluations})
    print("NATIVE_AUTHORITY_CELL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
