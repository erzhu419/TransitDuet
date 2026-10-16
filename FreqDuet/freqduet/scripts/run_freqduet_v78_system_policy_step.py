#!/usr/bin/env python3
"""Small closed-loop test of a finite-day, constrained on-policy adjustment."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import freqduet_system_policy_step as step
from freqduet_system_credit import observable_decision
from freqduet_system_learner import ArrivalMoments, SystemStateEncoder
import run_freqduet_v60_deployment_factorial as deployment
import run_freqduet_v69_event_coupling as event
from run_freqduet_v58_system_credit import CreditObserver, reference_trace
from run_freqduet_v59_system_learning import csv_episode
from run_freqduet_v73_fleet_state import write_json


joint = deployment.joint
SEEDS = joint.SEEDS
TRAIN_DAYS = tuple(range(1780001, 1780033))
VALID_DAYS = tuple(range(1781001, 1781017))
TRAIN_ITEMS = list(itertools.product(SEEDS, TRAIN_DAYS))
MODES = (("deterministic", 0), ("stochastic", 0), ("stochastic", 1), ("stochastic", 2))
EVAL_ITEMS = list(itertools.product(SEEDS, VALID_DAYS, MODES))
METRICS = ("passenger_person_s", "journey_min", "cv_squared", "readiness_delay_s",
           "holding_s_per_trip", "trips_unlaunched", "trips_incomplete")


def due_at_return(env, direction):
    now, count = float(env.current_time), 0
    for trip in env.timetables:
        if trip.launched or bool(trip.direction) != direction:
            continue
        published = float(getattr(trip, "_freqduet_scheduled_launch", trip.launch_time))
        offset = not bool(getattr(trip, "_freqduet_terminal_dispatch", False)) and hasattr(trip, "_delta_t")
        release = max(published, float(trip._original_launch + trip._delta_t)) if offset else published
        count += release <= now
    return count


def rollout(root, seed, scenario, mode, sampler_seed, theta, label, save_training=False):
    _, Runner, load_config, runtime_torch = joint.optimized_runtime(root)
    observer, moments = CreditObserver(), ArrivalMoments()
    uniforms = event.EventUniforms(sampler_seed)
    spent, records = defaultdict(float), []
    policy, encoder = None, None

    class PolicyRunner(Runner):
        def _lower_policy_action(self, obs, last_action=0., deterministic=False):
            nonlocal policy, encoder
            if encoder is None:
                env = self.env
                encoder = SystemStateEncoder(self.lower_state_dim, env.max_agent_num,
                    len(env.timetables), len(env.stations), env.protocol.evaluation_end_time_s)
                policy = deployment.make_policy(self.lower_trainer.policy_net, encoder, sampler_seed)
            key = self._pilot_agent
            bus = self._bus_for_agent(key)
            self._capture_lower_action_context(key, bus)
            decision = observable_decision(self, obs, key, last_action)
            decision = replace(decision, pending_commands=tuple(observer.stream.pending_commands))
            moments.observe(self.env.headway_events.headways_s)
            encoded = encoder.encode(decision, moments)
            f = step.action_features(self._pending_lower_action_context[int(key)],
                len(bus.passengers) / bus.capacity, self.env._ready_vehicle_count(not bool(bus.direction)),
                due_at_return(self.env, not bool(bus.direction)), spent[bus.trip_id])
            probabilities = step.probabilities(policy, torch.from_numpy(encoded)[None, :], f, theta)[0].numpy().astype(float)
            probabilities /= probabilities.sum()
            if mode == "deterministic":
                selected = int(np.argmax(probabilities))
            else:
                identity, uniform = uniforms.draw(bus.trip_id, bus.last_station.station_id, bus.direction)
                selected = event.categorical_index(probabilities, uniform)
                uniforms.records[identity] = dict(uniform=uniform, order=len(uniforms.records),
                    time_s=float(self.env.current_time), action_s=float(step.ACTIONS[selected]))
            records.append(dict(time_s=float(self.env.current_time), features=f,
                                probabilities=probabilities, action=selected))
            return np.asarray([step.ACTIONS[selected]], dtype=np.float32)

        def _lower_action_for_agent(self, obs, key, last_action=0., deterministic=False):
            self._pilot_agent = key
            result = super()._lower_action_for_agent(obs, key, last_action, deterministic)
            action = float(np.asarray(result).item())
            if action != step.ACTIONS[records[-1]["action"]]:
                raise ValueError("returned action differs from the sampled categorical action")
            decision = observable_decision(self, obs, key, last_action)
            spent[decision.trip_id] += action
            observer.stream.decide(decision, action)
            return result

    joint.seed_runtime(runtime_torch, seed)
    config = load_config(str(joint.MODEL_ROOT / "configs_freqduet" / f"{joint.CURRENT}.yaml"))
    runner = PolicyRunner(joint.runtime_config(config, seed, root / "cells" / label / "logs", 40), device="cpu")
    runner.load_checkpoint(joint.TRAIN_ROOT / "logs" / f"{joint.CURRENT}_seed{seed}/checkpoints",
                           ep=joint.BASE_EP, require_deployment_state=True)
    if runner.lower_terminal_action_mode != "transition" or runner.lower_causal_guard_policy_mask_enabled:
        raise ValueError("V78 requires the original seven-bin, unmasked B200 action contract")
    upper_before = copy.deepcopy(runner.upper_trainer.policy_net.state_dict())
    lower_before = copy.deepcopy(runner.lower_trainer.policy_net.state_dict())
    observer.attach(runner)
    episode = runner.run_episode(joint.BASE_EP, training=False, scenario_seed=scenario, record_diagnostics=False)
    for network, before in ((runner.upper_trainer.policy_net, upper_before), (runner.lower_trainer.policy_net, lower_before)):
        for name, value in before.items():
            torch.testing.assert_close(network.state_dict()[name], value, atol=0, rtol=0)
    audit = observer.stream.audit()
    if len(records) != len(observer.stream.transitions) or len(runner.replay_buffer):
        raise ValueError("on-policy records do not align with the once-only system stream")
    moments.observe(runner.env.headway_events.headways_s)
    cv = float(runner.env.headway_events.summary()["headway_cv"])
    np.testing.assert_allclose(moments.cv_squared, cv ** 2, atol=1e-10, rtol=0)
    row = dict(train_seed=seed, scenario_seed=scenario, mode=mode, sampler_seed=sampler_seed,
        passenger_person_s=float(observer.stream.total[0] + observer.stream.total[1]),
        journey_min=float(episode["restricted_total_journey_horizon_min"]), cv_squared=moments.cv_squared,
        readiness_delay_s=float(episode["fleet_readiness_delay_mean_s"]),
        holding_s_per_trip=float(episode["holding_vehicle_seconds_per_launched_trip"]),
        trips_unlaunched=int(runner.env._measurement_details["trips_unlaunched"]),
        trips_incomplete=int(runner.env._measurement_details["trips_incomplete"]),
        action_counts={str(int(a)): sum(h["action_s"] == a for h in observer.holds) for a in step.ACTIONS},
        stream_audit=audit, theta=list(theta), base_actors_unchanged=True,
        training_updates=0, dual_updates=0, main_policy_changed=False)
    if save_training:
        exposures = np.asarray([t.exposure[0] + t.exposure[1] for t in observer.stream.transitions])
        remaining = np.cumsum(exposures[::-1])[::-1]
        np.testing.assert_allclose(remaining[0] + observer.stream.prefix[:2].sum(), row["passenger_person_s"], rtol=0, atol=1e-7)
        path = root / "training" / f"seed{seed}_day{scenario}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, times=[r["time_s"] for r in records],
            features=[r["features"] for r in records], probabilities=[r["probabilities"] for r in records],
            actions=[r["action"] for r in records], remaining_passenger=remaining,
            scenario_seed=scenario, horizon_s=runner.env.protocol.evaluation_end_time_s,
            cv_squared=row["cv_squared"], readiness_delay_s=row["readiness_delay_s"])
    write_json(root / "ready" / (label + ".json"), row)
    print("V78_ROLLOUT_DONE " + json.dumps(dict(label=label, mode=mode)), flush=True)
    return row, uniforms.records, episode, observer.holds


def train_item(root, index):
    seed, day = TRAIN_ITEMS[index]
    label = f"train_seed{seed}_day{day}"
    if not (root / "ready" / (label + ".json")).exists():
        rollout(root, seed, day, "stochastic", 178000000 + seed + 100 * (day - TRAIN_DAYS[0]),
                [0.] * len(step.FEATURES), label, save_training=True)
    return index


def fit(root, seed):
    days = []
    for day in TRAIN_DAYS:
        with np.load(root / "training" / f"seed{seed}_day{day}.npz") as data:
            days.append(dict(data))
    result = step.fit_step(days)
    result.update(train_seed=seed, original_actor="B200 warm start, not the V59 post-64 actor",
                  base_actor_updates=0, dual_updates=0, main_policy_changed=False,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    write_json(root / "adapters" / f"seed{seed}.json", result)
    print("V78_ADAPTER_FROZEN " + json.dumps({k: result[k] for k in
          ("train_seed", "theta", "update_identified", "training_mean_kl", "directional_cost_changes")}), flush=True)


def eval_item(root, index):
    seed, day, (mode, rep) = EVAL_ITEMS[index]
    path = root / "pairs" / f"p{index}.json"
    if path.exists():
        return index
    adapters = {s: json.loads((root / "adapters" / f"seed{s}.json").read_text()) for s in SEEDS}
    sampler = 178100000 + seed + 100 * (day - VALID_DAYS[0]) + rep
    reference = rollout(root, seed, day, mode, sampler, [0.] * len(step.FEATURES), f"eval_p{index}_reference")
    theta = adapters[seed]["theta"]
    if adapters[seed]["update_identified"]:
        candidate = rollout(root, seed, day, mode, sampler, theta, f"eval_p{index}_candidate")
    else:
        candidate = reference
    alignment = event.alignment(reference[1], candidate[1]) if mode == "stochastic" else None
    write_json(path, dict(index=index, train_seed=seed, scenario_seed=day, mode=mode, rep=rep,
        reference=reference[0], candidate=candidate[0], alignment=alignment,
        reused_identical_reference=not adapters[seed]["update_identified"],
        effects={m: candidate[0][m] - reference[0][m] for m in METRICS}))
    return index


def interval(rows, metric):
    values = [float(np.mean([r["effects"][metric] for r in rows if r["scenario_seed"] == d])) for d in VALID_DAYS]
    result = event.labels.expanded.paired_interval(values, 78000)
    return {k: result[k] for k in ("n_demand_days", "mean", "ci95")}


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(EVAL_ITEMS))]
    adapters = {str(s): json.loads((root / "adapters" / f"seed{s}.json").read_text()) for s in SEEDS}
    effects, gates = {}, {}
    for mode in ("stochastic", "deterministic"):
        selected = [r for r in rows if r["mode"] == mode]
        effects[mode] = {m: interval(selected, m) for m in METRICS}
        gates[mode] = effects[mode]["passenger_person_s"]["ci95"][1] < 0 and all(
            effects[mode][m]["ci95"][1] <= 0 for m in ("cv_squared", "readiness_delay_s", "trips_unlaunched", "trips_incomplete"))
    write_json(root / "summary" / "status.json", dict(status="finite_day_policy_step_pilot_complete",
        adapters=adapters, effects=effects, joint_gates=gates, primary_mode="stochastic",
        primary_gate_passed=gates["stochastic"], deterministic_gate_passed=gates["deterministic"],
        pairs=len(rows), original_training_seeds=len(SEEDS), training_days_per_seed=len(TRAIN_DAYS),
        heldout_demand_days=len(VALID_DAYS), paired_sampling_repeats=3,
        new_rollouts=len(TRAIN_ITEMS) + sum(1 + (not r["reused_identical_reference"]) for r in rows),
        zero_update_cases=[int(s) for s, a in adapters.items() if not a["update_identified"]],
        base_actor_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        by_seed={str(s): {mode: {m: interval([r for r in rows if r["train_seed"] == s and r["mode"] == mode], m)
            for m in METRICS} for mode in effects} for s in SEEDS},
        alignment={k: sum(r["alignment"][k] for r in rows if r["alignment"] is not None)
                   for k in next(r["alignment"] for r in rows if r["alignment"] is not None)},
        diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))
    print("V78_PILOT_COMPLETE " + json.dumps(gates), flush=True)


def smoke(root, v56_root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v78_*.py", "-v"], check=True)
    reference = reference_trace(v56_root, SEEDS[0], joint.SCENARIOS[0])
    zero = rollout(root, SEEDS[0], joint.SCENARIOS[0], "deterministic", 178990000,
                   [0.] * len(step.FEATURES), "smoke_zero")
    joint.check_unchanged(csv_episode(zero[2]), reference["snapshot"]["episode"])
    pd.testing.assert_frame_equal(pd.DataFrame(zero[3]), reference["holds"], check_exact=True)
    changed = rollout(root, SEEDS[0], 1789901, "stochastic", 178990001, [1.] * len(step.FEATURES), "smoke_features", True)
    write_json(root / "smoke" / "done.json", dict(status="tests_native_identity_and_feature_rollout_passed",
        exact_zero_adapter_path=True, decisions=changed[0]["stream_audit"]["decisions"],
        formal_validation_days_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "train", "fit", "evaluate", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v56-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command in ("train", "evaluate"):
        items, function = (TRAIN_ITEMS, train_item) if args.command == "train" else (EVAL_ITEMS, eval_item)
        end = len(items) if args.end is None else args.end
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(function, args.out_root, i) for i in range(args.start, end)]
            for task in pending:
                task.result()
        write_json(args.out_root / "shards" / args.command / f"{args.start}-{end}.json", dict(status="shard_complete"))
    elif args.command == "smoke":
        smoke(args.out_root, args.v56_root)
    elif args.command == "fit":
        fit(args.out_root, args.seed)
    else:
        aggregate(args.out_root)


if __name__ == "__main__":
    main()
