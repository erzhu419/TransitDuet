#!/usr/bin/env python3
"""Matched single-action grids with closed-loop or recorded lower continuation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v82_local_global_regularity as trace


pilot, step = trace.pilot, trace.step
STRATA = ("missing_bilateral", "bilateral_active", "bilateral_inactive")
REFERENCES = list(itertools.product(pilot.SEEDS, trace.DAYS))
CONTEXTS = list(itertools.product(range(len(REFERENCES)), STRATA))
MODES = ("closed_loop", "recorded_lower")
ITEMS = list(itertools.product(range(len(CONTEXTS)), MODES, range(len(step.ACTIONS))))
TARGET_S = 7200.


def command_rows(records):
    return [[r["time_s"], r["trip_id"], r["station_id"], int(r["direction"]), int(r["action"])] for r in records]


def adjacent_pair(events, trip, station, direction, target):
    ordered = sorted(((float(t), int(k)) for k, s, d, t in events
                      if (s, bool(d)) == (station, bool(direction))), key=lambda r: r[0])
    index = next((i for i, (_, k) in enumerate(ordered) if k == trip), None)
    if index is None or index == 0 or index == len(ordered) - 1:
        return None
    current, previous, follower = ordered[index], ordered[index - 1], ordered[index + 1]
    forward, backward = current[0] - previous[0], follower[0] - current[0]
    return dict(time_s=current[0], forward_gap_s=forward, follower_gap_s=backward,
        follower_trip_id=follower[1], loss=((forward - target) ** 2 + (backward - target) ** 2) / (2. * target ** 2))


def expected_effect(law, base, values):
    values = np.asarray(values, dtype=float)
    # Center before weighting so the large common full-day prefix cancels numerically.
    return float((np.asarray(law) - np.asarray(base)) @ (values - values[0]))


class PropagationTrace(trace.LocalGlobalAudit):
    def __init__(self, reference=None, stratum=None, mode="closed_loop", action=None):
        super().__init__("projected")
        self.reference, self.stratum, self.mode, self.action = reference, stratum, mode, action
        self.captures, self.fired, self.result = {}, False, None
        self.target = reference["contexts"][stratum] if reference else None
        self.tape = {tuple(r[1:4]): r[4] for r in reference["commands"]} if reference else None

    def attach(self, runner):
        super().attach(runner)
        self.runner = runner

    def __call__(self, policy, states, features, theta):
        law = super().__call__(policy, states, features, theta)
        runner = self.runner
        bus = runner._bus_for_agent(runner._pilot_agent)
        key = (int(bus.trip_id), int(bus.last_station.station_id), int(bus.direction))
        context = runner._pending_lower_action_context[int(runner._pilot_agent)]
        base, projected, active = self.laws[-1]
        category = ("bilateral_active" if active else "bilateral_inactive") if (
            context.evidence_valid and context.follower_evidence_valid) else "missing_bilateral"
        eligible = (self.target is None and runner.env.current_time >= TARGET_S and category not in self.captures)
        matched = self.target is not None and key == tuple(self.target["key"])
        if eligible or matched:
            if not bool(policy.feasible_action_mask(states).all()):
                raise AssertionError("the registered context must permit all seven physical bins")
            stations = [int(s.station_id) for s in bus.effective_station]
            position = stations.index(key[1])
            raw = step.probabilities(policy, states, features, theta)[0].numpy().astype(float)
            capture = dict(key=list(key), ordinal=len(self.laws) - 1, time_s=float(runner.env.current_time),
                ready_time_s=self.ready[key], bus_id=int(bus.bus_id), encoded_state=states[0].numpy().tolist(),
                context=asdict(context), features=features.tolist(), base=base.tolist(), raw=(raw / raw.sum()).tolist(),
                projected=projected.tolist(), downstream=stations[position + 1:position + 6],
                predicted_follower_trip=self.followers[key], load=int(len(bus.passengers)))
            self.captures[category] = capture
            if matched:
                for name in capture:
                    pilot.joint.assert_same_state(self.target[name], capture[name], "pre_action_" + name)
                self.fired = True
        selected = None
        if matched:
            selected = int(self.action)
        elif self.fired and self.mode == "recorded_lower":
            selected = int(self.tape[key])
        if selected is None:
            return law
        actual = np.eye(len(step.ACTIONS))[selected]
        self.laws[-1] = (base, actual, active)
        return torch.from_numpy(actual[None, :])

    def finish(self, runner, records, holds):
        commands = command_rows(records)
        arrivals = [[e["trip_id"], e["station_id"], int(e["direction"]), e["arrival_time_s"]]
                    for e in runner.env.headway_events.events]
        departures = [[k[0], k[1], int(k[2]), t] for k, t in self.departures.items()]
        physical_holds = {(h["trip_id"], h["station_id"]): h["action_s"] for h in holds}
        if self.target is not None:
            if not self.fired:
                raise AssertionError("the registered logical intervention was not reached")
            ordinal, time = self.target["ordinal"], self.target["time_s"]
            np.testing.assert_array_equal(commands[:ordinal], self.reference["commands"][:ordinal])
            np.testing.assert_array_equal([e for e in arrivals if e[3] < time],
                                          [e for e in self.reference["arrivals"] if e[3] < time])
            if commands[ordinal][-1] != self.action:
                raise AssertionError("physical intervention differs from requested bin")
            if self.mode == "recorded_lower":
                for r in commands[ordinal + 1:]:
                    if r[-1] != self.tape[tuple(r[1:4])]:
                        raise AssertionError("a later lower command differs from its recorded tape")
        outcomes = {}
        for category, capture in self.captures.items():
            trip, station, direction = capture["key"]
            target = float(capture["context"]["target_headway_s"])
            witness = dict(departure=adjacent_pair(departures, trip, station, direction, target))
            for hop in (1, 3, 5):
                witness[f"arrival_h{hop}"] = adjacent_pair(arrivals, trip, capture["downstream"][hop - 1], direction, target) \
                    if len(capture["downstream"]) >= hop else None
            follower = capture["predicted_follower_trip"]
            key = (follower, station, bool(direction))
            witness.update(predicted_follower_ready_s=self.ready.get(key),
                predicted_follower_departure_s=self.departures.get(key),
                predicted_follower_holding_s=physical_holds.get((follower, station)))
            outcomes[category] = witness
        self.result = dict(contexts=self.captures, commands=commands, arrivals=arrivals, witnesses=outcomes)


def reference(root, v81_root, v78_root, index):
    path = root / "references" / f"r{index}.json"
    if path.exists():
        return
    seed, day = REFERENCES[index]
    original_index = trace.projection.ITEMS.index((seed, day, ("stochastic", 0)))
    original = json.loads((v81_root / "pairs" / f"p{original_index}.json").read_text())["results"]["projected"]
    theta = json.loads((v78_root / "adapters" / f"seed{seed}.json").read_text())["theta"]
    sampler = 181000000 + seed + 100 * (day - trace.projection.DAYS[0])
    observer = PropagationTrace()
    row, events, _, _ = pilot.rollout(root, seed, day, "stochastic", sampler, theta, f"reference{index}",
                                     probability_rule=observer, trace_observer=observer)
    for metric in pilot.METRICS:
        np.testing.assert_equal(row[metric], original[metric])
    if set(observer.captures) != set(STRATA):
        raise AssertionError("not all registered causal evidence strata were observed")
    for capture in observer.captures.values():
        capture["native_index"] = observer.result["commands"][capture["ordinal"]][-1]
    pilot.write_json(path, dict(index=index, seed=seed, day=day, theta=theta, sampler=sampler,
        metrics={m: row[m] for m in pilot.METRICS}, events=[[list(k), v] for k, v in events.items()], **observer.result))
    print("V83_REFERENCE_DONE " + str(index), flush=True)


def arm(root, index, force_native=False):
    path = root / "arms" / f"a{index}.json"
    if path.exists() and not force_native:
        return
    context_index, mode, action = ITEMS[index]
    ref_index, category = CONTEXTS[context_index]
    ref = json.loads((root / "references" / f"r{ref_index}.json").read_text())
    capture = ref["contexts"][category]
    native = action == capture["native_index"]
    if native and not force_native:
        metrics, witness, alignment, changes = ref["metrics"], ref["witnesses"][category], None, 0
    else:
        observer = PropagationTrace(ref, category, mode, action)
        row, events, _, _ = pilot.rollout(root, ref["seed"], ref["day"], "stochastic", ref["sampler"], ref["theta"],
            f"arm{index}" + ("_forced_native" if force_native else ""), probability_rule=observer, trace_observer=observer)
        metrics, witness = {m: row[m] for m in pilot.METRICS}, observer.result["witnesses"][category]
        alignment = pilot.event.alignment({tuple(k): v for k, v in ref["events"]}, events)
        tape = {tuple(r[1:4]): r[-1] for r in ref["commands"]}
        changes = sum(r[-1] != tape[tuple(r[1:4])] for r in observer.result["commands"][capture["ordinal"] + 1:])
        if native:
            pilot.joint.assert_same_state(ref["metrics"], metrics, "forced_native_metrics")
            np.testing.assert_array_equal(ref["commands"], observer.result["commands"])
            np.testing.assert_array_equal(ref["arrivals"], observer.result["arrivals"])
    pilot.write_json(path, dict(index=index, context_index=context_index, reference_index=ref_index, stratum=category,
        mode=mode, action_index=action, action_s=float(step.ACTIONS[action]), metrics=metrics, witness=witness,
        alignment=alignment, later_lower_action_changes=changes, reused_native=native and not force_native,
        forced_native_verified=force_native, exact_prefix=True))
    print("V83_ARM_DONE " + str(index), flush=True)


def aggregate(root):
    arms = [json.loads((root / "arms" / f"a{i}.json").read_text()) for i in range(len(ITEMS))]
    refs = [json.loads((root / "references" / f"r{i}.json").read_text()) for i in range(len(REFERENCES))]
    contexts = []
    for ci, (ri, category) in enumerate(CONTEXTS):
        capture = refs[ri]["contexts"][category]
        effects = {}
        for mode in MODES:
            rows = sorted([r for r in arms if (r["context_index"], r["mode"]) == (ci, mode)], key=lambda r: r["action_index"])
            values = {m: [r["metrics"][m] for r in rows] for m in pilot.METRICS}
            for stage in ("departure", "arrival_h1", "arrival_h3", "arrival_h5"):
                values[stage + "_loss"] = [r["witness"][stage]["loss"] if r["witness"][stage] else None for r in rows]
            effects[mode] = {label + "_vs_base": {m: expected_effect(capture[label], capture["base"], v)
                if all(x is not None for x in v) else None for m, v in values.items()} for label in ("raw", "projected")}
        contexts.append(dict(reference_index=ri, seed=refs[ri]["seed"], day=refs[ri]["day"], stratum=category,
            time_s=capture["time_s"], predicted_projected_gain=expected_effect(capture["projected"], capture["base"],
                [g[0] for g in capture["features"]]), effects=effects))
    groups = {}
    for category in STRATA:
        selected = [c for c in contexts if c["stratum"] == category]
        groups[category] = {mode: {contrast: {m: dict(observed=sum(c["effects"][mode][contrast][m] is not None for c in selected),
            mean=float(np.mean([c["effects"][mode][contrast][m] for c in selected if c["effects"][mode][contrast][m] is not None]))
                if any(c["effects"][mode][contrast][m] is not None for c in selected) else None)
            for m in selected[0]["effects"][mode][contrast]} for contrast in ("raw_vs_base", "projected_vs_base")} for mode in MODES}
    pilot.write_json(root / "summary/status.json", dict(status="single_action_holding_propagation_complete",
        contexts=contexts, by_stratum=groups, labelled_arms=len(arms), references=len(refs),
        forced_native_checks=sum(r["forced_native_verified"] for r in arms),
        max_recorded_lower_changes=max(r["later_lower_action_changes"] for r in arms if r["mode"] == "recorded_lower"),
        actors_updated=False, main_policy_changed=False, controller_fits=0,
        scope="consumed four-day bank, one fixed future per prefix; causal interventions, not a promotion gate",
        full_policy_effects_are_not_additive_single_action_effects=True))
    print("V83_HOLDING_PROPAGATION_COMPLETE", flush=True)


def smoke(root, v81_root, v78_root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v83_*.py", "-v"], check=True)
    reference(root, v81_root, v78_root, 0)
    ref = json.loads((root / "references/r0.json").read_text())
    native = ref["contexts"]["bilateral_active"]["native_index"]
    ci = CONTEXTS.index((0, "bilateral_active"))
    far = 0 if step.ACTIONS[native] >= 20 else len(step.ACTIONS) - 1
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        pending = [pool.submit(arm, root, ITEMS.index((ci, mode, native)), True) for mode in MODES]
        pending.append(pool.submit(arm, root, ITEMS.index((ci, "recorded_lower", far))))
        for task in pending:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="tests_native_identity_and_changed_tape_replay_passed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "references", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v81-root", type=Path)
    parser.add_argument("--v78-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v81_root, args.v78_root)
    elif args.command in ("references", "run"):
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            pending = ([pool.submit(reference, args.out_root, args.v81_root, args.v78_root, i) for i in range(len(REFERENCES))]
                if args.command == "references" else [pool.submit(arm, args.out_root, i) for i in range(args.start, args.end)])
            for task in pending:
                task.result()
        marker = "reference_ready/done.json" if args.command == "references" else f"shards/{args.start}-{args.end}.json"
        pilot.write_json(args.out_root / marker, dict(status="complete"))
    else:
        aggregate(args.out_root)


if __name__ == "__main__":
    main()
