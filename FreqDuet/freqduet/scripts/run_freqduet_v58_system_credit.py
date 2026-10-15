#!/usr/bin/env python3
"""Verify a shadow cross-trip credit stream without changing the frozen controller."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import pandas as pd

from freqduet_system_credit import Command, SystemCreditStream, observable_decision
import run_freqduet_v37_joint_lower as joint
from run_freqduet_v56_benefit_timing import exposure_at, load_plan, load_trace


CASES = list(itertools.product(joint.SEEDS, joint.SCENARIOS))


class CreditObserver:
    def __init__(self):
        self.stream = SystemCreditStream()
        self.holds = []
        self.legacy_terminals = 0
        self.last_rates = None

    def attach(self, runner):
        env = runner.env
        tracker = env._upper_interval_outcome_tracker
        original_record, original_step = tracker.record_step, env.step

        def record_step(**kwargs):
            original_record(**kwargs)
            self.last_rates = [sum(kwargs[name].values()) for name in (
                "waiting_by_direction", "onboard_by_direction", "waiting_low_by_direction",
                "onboard_low_by_direction", "dispatch_backlog_by_direction")]
            self.stream.advance(float(env.current_time), float(kwargs["dt_s"]), *self.last_rates)

        def step(action, *args, **kwargs):
            now = float(env.current_time)
            before = {b.bus_id: (bool(b.on_route), int(b.trip_id)) for b in env.bus_all}
            result = original_step(action, *args, **kwargs)
            executed = []
            for bus in env.bus_all:
                previous = before.get(bus.bus_id)
                if previous and previous[0] and not bus.on_route:
                    self.stream.returned(bus.bus_id, previous[1], bus.back_to_terminal_time)
                if bus.last_action_time == now:
                    hold = dict(time_s=now, bus_id=int(bus.bus_id), trip_id=int(bus.trip_id),
                                station_id=int(bus.last_action_station_id), action_s=float(bus.last_action_s))
                    self.holds.append(hold)
                    if not runner._lower_terminal_action_masked(bus):
                        executed.append(Command(**hold))
            self.stream.acknowledge(executed)
            if env.done:
                details = env._measurement_details
                cleared = (env._done_reason == "service_cleared"
                           and details["trips_unlaunched"] == 0 and details["trips_incomplete"] == 0)
                if self.last_rates[0] != details["passengers_unserved"]:
                    raise ValueError("terminal waiting count differs from censored passenger ledger")
                self.stream.finish(env.protocol.evaluation_end_time_s, service_cleared=cleared,
                                   unserved=self.last_rates[0], lf_unserved=self.last_rates[2])
            return result

        tracker.record_step, env.step = record_step, step


def case_name(seed, scenario):
    return f"seed{seed}_eval{scenario}"


def reference_trace(v56_root, seed, scenario):
    plan = load_plan(v56_root)
    context = next(c for c in plan["contexts"] if
                   (c["seed"], c["scenario"], c["target_time_s"]) == (seed, scenario, 7200))
    item = next(i for i in plan["items"] if i["label"] == context["arms"]["reference"])
    return load_trace(v56_root, item)


def collect(root, v56_root, seed, scenario):
    label = case_name(seed, scenario)
    marker = root / "ready" / f"{label}.json"
    if marker.exists():
        return label
    _, Runner, load_config, torch = joint.optimized_runtime(root)
    observer = CreditObserver()

    class ShadowRunner(Runner):
        def _lower_action_for_agent(self, obs, key, last_action=0.0, deterministic=False):
            result = super()._lower_action_for_agent(obs, key, last_action, deterministic)
            bus = self._bus_for_agent(key)
            if not self._lower_terminal_action_masked(bus):
                observer.stream.decide(observable_decision(self, obs, key, last_action),
                                       float(np.asarray(result).item()))
            return result

        def _record_lower_transition(self, **event):
            result = super()._record_lower_transition(**event)
            observer.legacy_terminals += int(event["transition_done"])
            return result

    cell = root / "cells" / label
    joint.seed_runtime(torch, seed)
    config = load_config(str(joint.MODEL_ROOT / "configs_freqduet" / f"{joint.CURRENT}.yaml"))
    runner = ShadowRunner(joint.runtime_config(config, seed, cell / "logs", 40), device="cpu")
    runner.load_checkpoint(joint.TRAIN_ROOT / "logs" / f"{joint.CURRENT}_seed{seed}/checkpoints",
                           ep=joint.BASE_EP, require_deployment_state=True)
    observer.attach(runner)
    try:
        runner.evaluate([scenario], output_dir=cell / "evaluation", policy_ep=joint.BASE_EP)
        episode = pd.read_csv(cell / "evaluation/evaluation.csv").iloc[0].to_dict()
        reference = reference_trace(v56_root, seed, scenario)
        joint.check_unchanged(episode, reference["snapshot"]["episode"])
        pd.testing.assert_frame_equal(pd.DataFrame(observer.holds), reference["holds"], check_exact=True)
        stream = observer.stream
        np.testing.assert_array_equal(stream.total[[0, 1, 4]], exposure_at(reference, stream.clock_s))
        summary = dict(label=label, train_seed=seed, eval_seed=scenario, **stream.audit(),
                       legacy_trip_terminals=observer.legacy_terminals,
                       exact_episode=True, exact_holding_path=True, exact_reference_exposure=True,
                       training_runs=0, main_policy_changed=False,
                       diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
        records = root / "records" / label
        records.mkdir(parents=True, exist_ok=True)
        with (records / "transitions.jsonl").open("w") as file:
            for transition in stream.transitions:
                row = asdict(transition)
                row.update(reward=transition.reward, done=transition.done,
                           bootstrap_discount=transition.bootstrap_discount)
                file.write(json.dumps(row, default=lambda v: v.tolist()) + "\n")
        (records / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(summary, indent=2) + "\n")
    except Exception as error:
        failures = root / "failures"
        failures.mkdir(parents=True, exist_ok=True)
        with (failures / f"{label}.jsonl").open("a") as file:
            file.write(json.dumps(dict(error=repr(error))) + "\n")
        raise
    print("V58_CASE_DONE " + json.dumps(summary), flush=True)
    return label


def batch(root, v56_root, start, end, workers):
    if not 0 <= start < end <= len(CASES):
        raise ValueError("case batch is outside the frozen four-day smoke")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        done = list(pool.map(collect, itertools.repeat(root), itertools.repeat(v56_root),
                             [s for s, _ in CASES[start:end]], [d for _, d in CASES[start:end]]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "done.json").write_text(json.dumps(dict(completed=done)) + "\n")


def aggregate(root):
    summaries = [json.loads((root / "ready" / f"{case_name(*case)}.json").read_text()) for case in CASES]
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    (summary / "cases.json").write_text(json.dumps(summaries, indent=2) + "\n")
    result = dict(status="shadow_system_credit_contract_pass", cases=len(summaries),
                  decisions=sum(s["decisions"] for s in summaries),
                  same_clock_bootstraps=sum(s["same_clock_bootstraps"] for s in summaries),
                  cross_trip_bootstraps=sum(s["cross_trip_bootstraps"] for s in summaries),
                  return_crossing_bootstraps=sum(s["return_crossing_bootstraps"] for s in summaries),
                  legacy_trip_terminals=sum(s["legacy_trip_terminals"] for s in summaries),
                  system_terminals=sum(s["terminal_transitions"] for s in summaries),
                  conservation_max_error=max(s["conservation_max_error"] for s in summaries),
                  all_original_episodes_and_actions_exact=True, training_runs=0, main_policy_changed=False)
    if result["system_terminals"] != len(summaries) or result["cross_trip_bootstraps"] == 0:
        raise ValueError("shadow stream did not retain finite-day cross-trip continuation")
    (summary / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v56-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(CASES))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        label = collect(args.out_root, args.v56_root, *CASES[0])
        export = args.out_root / "smoke_export"
        export.mkdir(exist_ok=True)
        (export / "done.json").write_text(json.dumps(dict(case=label, status="pass")) + "\n")
    elif args.command == "batch":
        batch(args.out_root, args.v56_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root)
