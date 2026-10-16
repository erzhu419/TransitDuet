#!/usr/bin/env python3
"""Matched temporal accounting of fixed V70 development interventions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
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

from audit_freqduet_v38_mechanism import contrast_events
from freqduet_system_credit import SystemCreditStream
import run_freqduet_v70_independent_labels as source


labels, joint, event = source.labels, source.joint, source.event
PLAN = {15: (5, (45,)), 123: (0, (20, 45)), 51: (0, (5, 45)),
        87: (0, (30,)), 125: (30, (0,))}
ITEMS = list(itertools.product(PLAN, source.FUTURES))
SMOKE_ITEMS = (0, 32, 128)
OUTCOMES = ("passenger_person_s", "journey_min", "cv_squared", "readiness_delay_s",
            "holding_total_s", "target_return_delta_s")


class TimelineStream(SystemCreditStream):
    def __init__(self):
        super().__init__()
        self.timeline = [[self.clock_s, *self.total]]

    def advance(self, *args, **kwargs):
        super().advance(*args, **kwargs)
        self.timeline.append([self.clock_s, *self.total])


class TraceObserver(labels.deployment.CreditObserver):
    def __init__(self):
        super().__init__()
        self.stream = TimelineStream()
        self.launches, self.denied = [], {}
        self.holding = [[0., 0., 0.]]

    def attach(self, runner):
        super().attach(runner)
        self.env = env = runner.env
        original_launch, original_step = env.launch_bus, env.step

        def launch(trip, actual_launch=None):
            result = original_launch(trip, actual_launch=actual_launch)
            now, tid = float(env.current_time), int(trip.launch_turn)
            if result is None:
                entry = self.denied.setdefault(tid, dict(trip_id=tid, first_time_s=now, last_time_s=now, attempts=0))
                entry["last_time_s"], entry["attempts"] = now, entry["attempts"] + 1
            else:
                self.launches.append(dict(time_s=now, trip_id=tid, bus_id=int(result.bus_id),
                                          direction=bool(trip.direction),
                                          scheduled_s=float(getattr(trip, "_freqduet_scheduled_launch", trip.launch_time)),
                                          readiness_s=max(0., now - float(trip._freqduet_first_eligible_time))))
            return result

        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.holding.append([float(env.current_time),
                                 float(sum(b.episode_hold_vehicle_seconds for b in env.bus_all)),
                                 float(sum(b.episode_hold_person_seconds for b in env.bus_all))])
            return result

        env.launch_bus, env.step = launch, step

    def payload(self, result, target):
        horizon = self.stream.clock_s
        if self.holding[-1][0] < horizon:
            self.holding.append([horizon, *self.holding[-1][1:]])
        capture = result["captures"][target]
        np.testing.assert_allclose(self.holding[-1][1], result["summary"]["holding_total_s"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(sum(self.stream.total[:2]), result["summary"]["passenger_person_s"], atol=0, rtol=0)
        return dict(summary=result["summary"], identity=capture["identity"], observation=capture["observation"],
                    timeline=np.asarray(self.stream.timeline), holding=np.asarray(self.holding),
                    holds=result["holds"], returns=self.stream.returns, launches=self.launches,
                    denied=list(self.denied.values()), headways=list(self.env.headway_events.events))


def cumulative(timeline, time):
    return np.asarray([np.interp(time, timeline[:, 0], timeline[:, i]) for i in range(1, timeline.shape[1])])


def trip_return(data, trip):
    returns = [r for r in data["returns"] if r["trip_id"] == trip]
    if len(returns) != 1:
        raise ValueError("intervention trip lacks its unique physical terminal return")
    return returns[0]["physical_time_s"]


def common_cuts(reference):
    time, bus, trip, _, _ = reference["identity"]
    returned = trip_return(reference, trip)
    next_launches = [r["time_s"] for r in reference["launches"] if r["bus_id"] == bus and r["time_s"] >= returned]
    cuts = dict(before=time, min15=time + 900, min30=time + 1800, reference_return=returned,
                day_end=float(reference["timeline"][-1, 0]))
    if next_launches:
        cuts["reference_next_launch"] = min(next_launches)
    if any(t > cuts["day_end"] for t in cuts.values()):
        raise ValueError("mechanism cut exceeds the fixed evaluation horizon")
    return cuts


def cv_squared_at(data, time):
    values = [r["headway_s"] for r in data["headways"] if r["arrival_time_s"] < time and r["headway_s"] is not None]
    moments = labels.deployment.ArrivalMoments()
    moments.observe(values)
    return moments.cv_squared


def timing_effects(reference, alternative, action):
    joint.assert_same_state(reference["identity"], alternative["identity"], "intervention_identity")
    cuts = common_cuts(reference)
    first = cuts["before"]
    np.testing.assert_array_equal(cumulative(reference["timeline"], first), cumulative(alternative["timeline"], first))
    np.testing.assert_array_equal(cumulative(reference["holding"], first), cumulative(alternative["holding"], first))
    rows = {}
    for name, time in cuts.items():
        delta = cumulative(alternative["timeline"], time) - cumulative(reference["timeline"], time)
        holding = cumulative(alternative["holding"], time) - cumulative(reference["holding"], time)
        commands = sum(h["action_s"] for h in alternative["holds"] if h["time_s"] < time)
        commands -= sum(h["action_s"] for h in reference["holds"] if h["time_s"] < time)
        ready = sum(r["readiness_s"] for r in alternative["launches"] if r["time_s"] <= time)
        ready -= sum(r["readiness_s"] for r in reference["launches"] if r["time_s"] <= time)
        rows[name] = dict(time_s=time, passenger_person_s=float(delta[0] + delta[1]),
                          waiting_person_s=float(delta[0]), onboard_person_s=float(delta[1]),
                          dispatch_backlog_trip_s=float(delta[4]), executed_holding_s=float(holding[0]),
                          holding_person_s=float(holding[1]), commanded_holding_s=float(commands),
                          other_command_holding_s=float(commands - (action - reference["native_action_s"] if time > first else 0)),
                          readiness_delay_sum_s=float(ready),
                          cv_squared=cv_squared_at(alternative, time) - cv_squared_at(reference, time))
    np.testing.assert_allclose(rows["day_end"]["passenger_person_s"],
        alternative["summary"]["passenger_person_s"] - reference["summary"]["passenger_person_s"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(rows["day_end"]["executed_holding_s"],
        alternative["summary"]["holding_total_s"] - reference["summary"]["holding_total_s"], atol=1e-6, rtol=0)
    return rows


def group_path(root, index):
    context, future = ITEMS[index]
    return root / "groups" / f"c{context}_f{future}.json"


def run_group(root, v59_root, v70_root, index, check_identity=False):
    path = group_path(root, index)
    if path.exists():
        return path.stem
    context, future = ITEMS[index]
    spec = source.context_spec(context)
    native_action, alternatives = PLAN[context]
    old = json.loads(source.group_path(v70_root, context * 32 + future).read_text())
    original = torch.load(source.reference_path(v70_root, spec["reference_index"]), map_location="cpu", weights_only=False)
    original = original["captures"][spec["target_time_s"]]
    joint.assert_same_state(original["actor_action_s"], native_action, "frozen_V70_native_action")
    seed = source.future_seed(context, future)
    joint.assert_same_state(old["future_seed"], seed, "frozen_V70_future_seed")

    def replay(action, suffix, observer):
        uniforms = event.EventUniforms(seed + 10000000)

        def continuation(runner, policy):
            startup = source.repeated.fork_future(runner, policy, seed)
            uniforms.install(runner, policy)
            return dict(startup, event_uniform_seed=uniforms.seed)

        result = labels.run_spec(root, v59_root, source.reference_spec(spec["reference_index"]),
                                 (spec["target_time_s"],), f"c{context}_f{future}_{suffix}", action,
                                 continuation, observer=observer)
        labels.same_prefix(original, result["captures"][spec["target_time_s"]])
        expected = next(r for r in old["rows"] if r["action_s"] == (native_action if action is None else action))
        for metric in labels.OUTCOMES:
            joint.assert_same_state(expected[metric], result["summary"][metric], "V70_endpoint_" + metric)
        print("V71_ARM_DONE " + json.dumps(dict(context_index=context, future_index=future, arm=suffix)), flush=True)
        return result, uniforms.records

    observer = TraceObserver()
    native, native_records = replay(None, "native", observer)
    reference = observer.payload(native, spec["target_time_s"])
    reference["native_action_s"] = native_action
    traces = root / "traces"
    traces.mkdir(exist_ok=True)
    torch.save(reference, traces / f"c{context}_f{future}_native.pt")
    rows = []
    for action in alternatives:
        observer = TraceObserver()
        result, records = replay(action, f"a{action}", observer)
        joint.assert_same_state(native["captures"][spec["target_time_s"]]["continuation"],
                                result["captures"][spec["target_time_s"]]["continuation"], "common_future_start")
        candidate = observer.payload(result, spec["target_time_s"])
        torch.save(candidate, traces / f"c{context}_f{future}_a{action}.pt")
        phases = timing_effects(reference, candidate, action)
        events = contrast_events({k: pd.DataFrame(reference[k]) for k in ("launches", "holds")},
                                 {k: pd.DataFrame(candidate[k]) for k in ("launches", "holds")})
        rows.append(dict(action_s=action, **{m: result["summary"][m] - native["summary"][m] for m in labels.OUTCOMES},
                         target_return_delta_s=trip_return(candidate, reference["identity"][2])
                             - trip_return(reference, reference["identity"][2]),
                         phases=phases, events=events, alignment=event.alignment(native_records, records)))
    if check_identity:
        forced_observer = TraceObserver()
        forced, forced_records = replay(native_action, "forced_native", forced_observer)
        plain, _ = replay(None, "uninstrumented_native", None)
        for key in ("summary", "holds", "upper_path"):
            joint.assert_same_state(native[key], forced[key], "forced_native_" + key)
            joint.assert_same_state(native[key], plain[key], "trace_does_not_change_" + key)
        joint.assert_same_state(native_records, forced_records, "forced_native_event_uniforms")
        joint.assert_same_state(reference, dict(forced_observer.payload(forced, spec["target_time_s"]),
                                                native_action_s=native_action), "forced_native_full_trace")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(dict(**spec, future_index=future, future_seed=seed, native_action_s=native_action,
                                    identity=reference["identity"], observation=reference["observation"], rows=rows,
                                    cached_endpoints_exact=True, exact_prefix=True, instrument_identity=check_identity,
                                    diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")), indent=2) + "\n")
    print("V71_GROUP_DONE " + json.dumps(dict(context_index=context, future_index=future)), flush=True)
    return path.stem


def future_interval(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(71000).integers(len(values), size=(10000, len(values)))
    return dict(n_futures=len(values), mean=float(values.mean()), ci95=np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist())


def phase_summary(rows, cut):
    values = [r["phases"][cut] for r in rows if cut in r["phases"]]
    return dict(passenger_person_s=future_interval([v["passenger_person_s"] for v in values]),
                executed_holding_s=future_interval([v["executed_holding_s"] for v in values]),
                means={m: float(np.mean([v[m] for v in values])) for m in
                       ("waiting_person_s", "onboard_person_s", "other_command_holding_s",
                        "readiness_delay_sum_s", "cv_squared")})


def aggregate(root):
    panels, all_groups = [], []
    for context, (native, actions) in PLAN.items():
        groups = [json.loads(group_path(root, ITEMS.index((context, f))).read_text()) for f in source.FUTURES]
        all_groups.extend(groups)
        if [g["future_index"] for g in groups] != list(source.FUTURES):
            raise ValueError("V71 panel lacks all 32 predeclared continuations")
        for action in actions:
            rows = [next(r for r in g["rows"] if r["action_s"] == action) for g in groups]
            cuts = sorted({cut for r in rows for cut in r["phases"] if cut != "before"})
            panels.append(dict(context_index=context, native_action_s=native, action_s=action,
                endpoint={m: future_interval([r[m] for r in rows]) for m in OUTCOMES},
                phases={cut: phase_summary(rows, cut) for cut in cuts},
                reference_next_launch_futures=sum("reference_next_launch" in r["phases"] for r in rows),
                passenger_sign_reversals={cut: sum(r["phases"][cut]["passenger_person_s"] * r["passenger_person_s"] < 0 for r in rows)
                                          for cut in ("min15", "min30", "reference_return")},
                event_means={m: float(np.mean([r["events"][m] for r in rows]))
                             for m in ("launches_changed", "published_plan_changes", "vehicles_reassigned", "holding_commands_changed")},
                reduced_other_commands_futures=sum(r["phases"]["day_end"]["other_command_holding_s"] < 0 for r in rows)))
    result = dict(status="fixed_V70_temporal_mechanism_complete", contexts=len(PLAN), paired_groups=len(all_groups),
                  formal_rollouts=32 * sum(1 + len(a) for _, a in PLAN.values()), futures_per_context=32,
                  cached_endpoints_exact=all(g["cached_endpoints_exact"] for g in all_groups),
                  exact_prefixes=all(g["exact_prefix"] for g in all_groups),
                  instrument_identity_groups=sum(g["instrument_identity"] for g in all_groups),
                  uniform_mismatches=sum(r["alignment"]["uniform_mismatches"] for g in all_groups for r in g["rows"]),
                  panels=panels, conditional_consumed_future_intervals=True, causal_mediation_identified=False,
                  training_runs=0, actor_updates=0, dual_updates=0, value_updates=0, main_policy_changed=False)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V71_AGGREGATE " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v70-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
        return
    if args.command == "smoke":
        tests = str(Path(__file__).resolve().parents[1] / "tests")
        for pattern in ("test_v70_*.py", "test_v71_*.py"):
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", tests, "-p", pattern, "-v"], check=True)
        indices = SMOKE_ITEMS
    else:
        if not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("batch is outside the fixed V71 intervention panel")
        indices = range(args.start, args.end)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        done = list(pool.map(run_group, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                             itertools.repeat(args.v70_root), indices,
                             itertools.repeat(args.command == "smoke")))
    output = args.out_root / ("smoke_export" if args.command == "smoke" else f"exports/batch_{args.start}_{args.end}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(completed_groups=len(done), smoke_identity_pass=args.command == "smoke")) + "\n")


if __name__ == "__main__":
    main()
