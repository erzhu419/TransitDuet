#!/usr/bin/env python3
"""Passive time accounting on the already evaluated V85 holding interventions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v85_feedback_recovery as recovery


repeated, pilot = recovery.repeated, recovery.pilot
MODES = (*recovery.propagation.MODES, "recover_900s")
ITEMS = list(itertools.product(range(len(repeated.REFERENCES)), MODES))
PERIOD_S = 300.
EXPOSURES = ("waiting_person_s", "onboard_person_s", "dispatch_backlog_trip_s")


def cleared_tail_at(time, last_time, last_total, horizon, final_total):
    # The existing credit stream extends cleared service by constant terminal rates.
    return last_total + (final_total - last_total) * ((time - last_time) / (horizon - last_time))


def headway_prefix(events, time):
    x = np.asarray([e["headway_s"] for e in events
                    if e["headway_s"] is not None and e["arrival_time_s"] <= time], dtype=float)
    total, square, n = float(x.sum()), float(x @ x), len(x)
    mean = total / n if n else 0.
    cv = max(square / n - mean * mean, 0.) / max(mean, 1.) ** 2 if n else 0.
    return dict(count=n, total_s=total, square_total_s2=square, cv_squared=cv)


class TemporalCredit(pilot.CreditObserver):
    def __init__(self, target):
        super().__init__()
        self.target, self.samples, self.launches = target, [], []
        self.last_time, self.last_total = 0., np.zeros(5)
        self.times, self.cursor = None, 0

    def attach(self, runner):
        super().attach(runner)
        env = runner.env
        tracker = env._upper_interval_outcome_tracker
        original_record, original_launch = tracker.record_step, env.launch_bus

        def record_step(**kwargs):
            original_record(**kwargs)
            # run_episode initializes the protocol in reset, after observer attachment.
            if self.times is None:
                self.horizon = float(env.protocol.evaluation_end_time_s)
                self.times = list(np.arange(self.target, self.horizon, PERIOD_S)) + [self.horizon]
            self.last_time, self.last_total = self.stream.clock_s, self.stream.total.copy()
            if self.cursor < len(self.times) and self.last_time >= self.times[self.cursor]:
                if self.last_time != self.times[self.cursor]:
                    raise AssertionError("the locked simulator clock missed a registered accounting boundary")
                self.capture(env, self.last_time, self.last_total)

        def launch(trip, actual_launch=None):
            bus = original_launch(trip, actual_launch=actual_launch)
            if bus is not None:
                now = float(env.current_time)
                self.launches.append(dict(time_s=now, trip_id=int(trip.launch_turn), bus_id=int(bus.bus_id),
                    direction=bool(trip.direction),
                    scheduled_s=float(getattr(trip, "_freqduet_scheduled_launch", trip.launch_time)),
                    readiness_s=max(0., now - float(trip._freqduet_first_eligible_time))))
            return bus

        tracker.record_step, env.launch_bus = record_step, launch

    def capture(self, env, time, total):
        self.samples.append(dict(time_s=float(time), **{k: float(total[i]) for k, i in zip(EXPOSURES, (0, 1, 4))},
            holding_vehicle_s=float(sum(b.episode_hold_vehicle_seconds for b in env.bus_all)),
            holding_person_s=float(sum(b.episode_hold_person_seconds for b in env.bus_all)),
            launched=len(self.launches), readiness_delay_sum_s=float(sum(x["readiness_s"] for x in self.launches))))
        self.cursor += 1

    def finish_timeline(self, runner):
        env = runner.env
        while self.cursor < len(self.times):
            time = self.times[self.cursor]
            total = cleared_tail_at(time, self.last_time, self.last_total, self.horizon, self.stream.total)
            self.capture(env, time, total)
        for sample in self.samples:
            sample["headways"] = headway_prefix(env.headway_events.events, sample["time_s"])
        last = self.samples[-1]
        np.testing.assert_allclose([last[k] for k in EXPOSURES], self.stream.total[[0, 1, 4]], atol=1e-7, rtol=0)
        np.testing.assert_equal(last["holding_vehicle_s"], env._measurement_details["holding_vehicle_seconds"])
        np.testing.assert_equal(last["launched"], env._measurement_details["trips_launched"])
        np.testing.assert_allclose(last["readiness_delay_sum_s"] / max(last["launched"], 1),
            env._measurement_details["fleet_readiness_delay_mean_s"], atol=1e-10, rtol=0)
        return dict(samples=self.samples, launches=self.launches, returns=self.stream.returns)


def arm(root, v84_root, v85_root, index, check_native=False):
    path = root / ("native_checks" if check_native else "arms") / f"a{index}.json"
    if path.exists():
        return
    ri, mode = ITEMS[index]
    ref = json.loads((v84_root / "references" / f"r{ri}.json").read_text())
    capture = ref["contexts"][ref["stratum"]]
    action = capture["native_index"] if check_native else 6
    uniforms = pilot.event.EventUniforms(ref["sampler"])
    credit = TemporalCredit(capture["time_s"])
    observer = (recovery.RecoveryTrace(ref, action, uniforms) if mode == "recover_900s" else
        repeated.FutureTrace(ref, ref["stratum"], mode, action, ref["future_seed"], uniforms))
    row, events, _, _ = pilot.rollout(root, ref["seed"], ref["day"], "stochastic", ref["sampler"], ref["theta"],
        f"v86_{index}" + ("_native" if check_native else ""), probability_rule=observer,
        trace_observer=observer, event_uniforms=uniforms, credit_observer=credit)
    metrics = {m: row[m] for m in pilot.METRICS}
    if check_native:
        pilot.joint.assert_same_state(ref["metrics"], metrics, "temporal_native_metrics")
        for key in ("commands", "arrivals"):
            pilot.joint.assert_same_state(ref[key], observer.result[key], "temporal_native_" + key)
        pilot.joint.assert_same_state(ref["events"], [[list(k), v] for k, v in events.items()], "temporal_native_events")
    else:
        old_index = recovery.ITEMS.index((ri, action)) if mode == "recover_900s" else repeated.ITEMS.index((ri, mode, action))
        old_root = v85_root if mode == "recover_900s" else v84_root
        old = json.loads((old_root / "arms" / f"a{old_index}.json").read_text())
        pilot.joint.assert_same_state(old["metrics"], metrics, "temporal_original_endpoints")
        pilot.joint.assert_same_state(old["witness"], observer.result["witnesses"][ref["stratum"]], "temporal_original_witness")
    alignment = pilot.event.alignment({tuple(k): v for k, v in ref["events"]}, events)
    timeline = credit.finish_timeline(observer.runner)
    np.testing.assert_allclose(timeline["samples"][-1]["headways"]["cv_squared"], metrics["cv_squared"], atol=1e-10, rtol=0)
    pilot.write_json(path, dict(index=index, reference_index=ri, context_index=ref["context_index"],
        future=ref["future"], seed=ref["seed"], day=ref["day"], stratum=ref["stratum"], mode=mode,
        target_key=capture["key"], target_bus=capture["bus_id"], target_time_s=capture["time_s"],
        action_s=float(repeated.step.ACTIONS[action]), metrics=metrics, alignment=alignment,
        exact_original_endpoints=True, forced_native_verified=check_native, **timeline))
    print("V86_TIMELINE_DONE " + str(index), flush=True)


def window_values(row, start, end):
    samples = {s["time_s"]: s for s in row["samples"]}
    origin = row["target_time_s"]
    left = samples[origin + start]
    right = samples[origin + end] if end is not None else row["samples"][-1]
    result = {k: right[k] - left[k] for k in (*EXPOSURES, "holding_vehicle_s", "holding_person_s", "readiness_delay_sum_s")}
    result["passenger_person_s"] = result["waiting_person_s"] + result["onboard_person_s"]
    result["cumulative_cv_squared"] = right["headways"]["cv_squared"]
    result["launched_trips"] = right["launched"] - left["launched"]
    result["cumulative_headway_count"] = right["headways"]["count"]
    return result


def vehicle_chain(row):
    returned = next((r for r in row["returns"] if r["trip_id"] == row["target_key"][0]), None)
    launched = next((r for r in row["launches"] if r["bus_id"] == row["target_bus"]
        and returned is not None and r["time_s"] >= returned["observed_time_s"]), None)
    return dict(returned=returned, next_launch=launched)


def aggregate(root):
    rows = {(ri, mode): json.loads((root / "arms" / f"a{i}.json").read_text()) for i, (ri, mode) in enumerate(ITEMS)}
    windows = ((0, 900), (900, 1800), (1800, 3600), (3600, None))
    contexts = []
    for ci in range(len(recovery.propagation.CONTEXTS)):
        refs = [ri for ri, (c, _) in enumerate(repeated.REFERENCES) if c == ci]
        effects, chains = {}, {}
        for control in recovery.propagation.MODES:
            effects[control] = {}
            for start, end in windows:
                left = [window_values(rows[ri, control], start, end) for ri in refs]
                right = [window_values(rows[ri, "recover_900s"], start, end) for ri in refs]
                values = {m: [b[m] - a[m] for a, b in zip(left, right)] for m in left[0]}
                if control == "recorded_lower" and start == 0:
                    for m in (*EXPOSURES, "holding_vehicle_s", "holding_person_s", "readiness_delay_sum_s"):
                        np.testing.assert_array_equal(values[m], np.zeros(len(refs)))
                effects[control][f"{start}-{end}"] = {m: repeated.describe(x) for m, x in values.items()}
            for ri in refs:
                total = sum(window_values(rows[ri, "recover_900s"], a, b)["passenger_person_s"] -
                            window_values(rows[ri, control], a, b)["passenger_person_s"] for a, b in windows)
                np.testing.assert_allclose(total, rows[ri, "recover_900s"]["metrics"]["passenger_person_s"] -
                    rows[ri, control]["metrics"]["passenger_person_s"], atol=1e-7, rtol=0)
            pairs = [(vehicle_chain(rows[ri, control]), vehicle_chain(rows[ri, "recover_900s"])) for ri in refs]
            returns = [b["returned"]["physical_time_s"] - a["returned"]["physical_time_s"]
                       for a, b in pairs if a["returned"] is not None and b["returned"] is not None]
            launches = [b["next_launch"]["time_s"] - a["next_launch"]["time_s"]
                        for a, b in pairs if a["next_launch"] is not None and b["next_launch"] is not None]
            chains[control] = dict(paired_returns=len(returns), paired_next_launches=len(launches),
                return_time_s=repeated.describe(returns) if len(returns) == len(refs) else None,
                same_bus_next_dispatch_s=repeated.describe(launches) if len(launches) == len(refs) else None,
                same_next_trip=sum(a["next_launch"] is not None and b["next_launch"] is not None and
                    a["next_launch"]["trip_id"] == b["next_launch"]["trip_id"] for a, b in pairs))
        first = rows[refs[0], "recover_900s"]
        contexts.append(dict(context_index=ci, seed=first["seed"], day=first["day"], stratum=first["stratum"],
                             effects=effects, target_vehicle=chains))
    pilot.write_json(root / "summary/status.json", dict(status="feedback_timing_complete", contexts=contexts,
        labelled_arms=len(ITEMS), action_s=45., actors_updated=False, main_policy_changed=False, controller_fits=0,
        scope="existing consumed prefixes/futures; fixed-45s regime differences, not a new controller",
        cv_is_cumulative_not_additive=True, raw_return_and_launch_ledgers="server arms only"))
    print("V86_FEEDBACK_TIMING_COMPLETE", flush=True)


def smoke(root, v84_root, v85_root):
    tests = Path(__file__).resolve().parents[1] / "tests"
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(tests), "-p", "test_v86_*.py", "-v"], check=True)
    first, second = [repeated.REFERENCES.index((ci, 0)) for ci in (1, 16)]
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        pending = [pool.submit(arm, root, v84_root, v85_root, ITEMS.index((first, mode)), True) for mode in MODES]
        pending.append(pool.submit(arm, root, v84_root, v85_root, ITEMS.index((second, "recover_900s")), True))
        pending.extend(pool.submit(arm, root, v84_root, v85_root, ITEMS.index((first, mode))) for mode in MODES)
        for task in pending:
            task.result()
    rows = {mode: json.loads((root / "arms" / f"a{ITEMS.index((first, mode))}.json").read_text()) for mode in MODES}
    for key in (*EXPOSURES, "holding_vehicle_s", "holding_person_s", "readiness_delay_sum_s"):
        np.testing.assert_equal(window_values(rows["recorded_lower"], 0, 900)[key],
                                window_values(rows["recover_900s"], 0, 900)[key])
    pilot.write_json(root / "smoke/done.json", dict(status="native_paths_original_endpoints_and_temporal_accounting_passed"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v84-root", type=Path)
    parser.add_argument("--v85-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v84_root, args.v85_root)
    elif args.command == "run":
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(arm, args.out_root, args.v84_root, args.v85_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))
    else:
        aggregate(args.out_root)


if __name__ == "__main__":
    main()
