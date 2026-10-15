#!/usr/bin/env python3
"""Read-only matched stop clocks and segment-error diagnosis of V43 forecasts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from inspect import signature
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

import run_freqduet_v42_distribution as distribution


def traced_event(original, stage, record):
    call_signature = signature(original)
    clock = "arrival_time_s" if stage == "arrival" else "departure_time_s"

    def wrapped(*args, **kwargs):
        values = call_signature.bind(*args, **kwargs).arguments
        record(stage, values["station_id"], values["direction"], values[clock], values["trip_id"])
        return original(*args, **kwargs)

    return wrapped


class SegmentObserver(distribution.DistributionObserver):
    def __init__(self, *args):
        super().__init__(*args)
        self.actual_events, self.forecast_events, self.requested_actions = [], [], []
        self.restorers = []

    def attach(self, runner):
        super().attach(runner)
        original_reset = runner.env.reset

        def reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            self.attach_recorder(runner)
            return result

        runner.env.reset = reset

    def attach_recorder(self, runner):
        env, recorder = runner.env, runner.env.headway_events

        def record(stage, station, direction, time, trip, **values):
            bus = next(bus for bus in env.bus_all if bus.trip_id == trip)
            departure = recorder.previous_departure_event(station, direction)
            valid = departure is not None and departure["trip_id"] == bus.forward_predecessor_trip_id
            self.actual_events.append(dict(stage=stage, station_id=station, direction=direction,
                time_s=float(time), trip_id=int(trip), bus_id=int(bus.bus_id), load=len(bus.passengers),
                progress_m=float(bus.travel_distance), target_headway_s=float(bus._target_headway),
                forward_gap_s=float(time - departure["time_s"]) if valid else None,
                follower_gap_s=bus.pre_action_follower_departure_gap if stage == "action" else None,
                evidence_valid=valid, **values))

        recorder.record = traced_event(recorder.record, "arrival", record)
        recorder.record_departure = traced_event(recorder.record_departure, "departure", record)
        bus_class = vars(sys.modules[type(env).__module__])["Bus"]
        original_prepare, original_start = bus_class._prepare_for_action, bus_class._start_dwelling

        def prepare(bus, time, buses, debug):
            result = original_prepare(bus, time, buses, debug)
            record("ready", int(bus.last_station.station_id), bool(bus.direction), time, bus.trip_id)
            if bus.state.name == "DWELLING":
                record("action", int(bus.last_station.station_id), bool(bus.direction), time, bus.trip_id, action_s=0.0)
            return result

        def start(bus, action, current_time=None):
            result = original_start(bus, action, current_time)
            record("action", int(bus.last_station.station_id), bool(bus.direction), current_time,
                   bus.trip_id, action_s=float(bus.dwelling_time or 0.0))
            return result

        bus_class._prepare_for_action, bus_class._start_dwelling = prepare, start
        self.restorers.append((bus_class, original_prepare, original_start))

    def close(self):
        for bus_class, prepare, start in reversed(self.restorers):
            bus_class._prepare_for_action, bus_class._start_dwelling = prepare, start

    def forecast_event_sink(self, label, horizon):
        if horizon != 1800:
            return None

        def record(draw, event):
            if draw < 8:
                self.forecast_events.append(dict(origin=label, draw=draw, **event))
        return record

    def observe(self, runner, obs, key, previous, action):
        super().observe(runner, obs, key, previous, action)
        context = runner._pending_lower_action_context[int(key)]
        self.requested_actions.append(dict(time_s=float(runner.env.current_time),
            action_s=float(np.asarray(action).item()), evidence_valid=context.evidence_valid,
            masked=runner._lower_terminal_action_masked(runner._bus_for_agent(key))))


def run_day(arguments):
    root, seed, scenario, cohort = arguments
    original, captured = distribution.DistributionObserver, []

    class Capture(SegmentObserver):
        def __init__(self, *args):
            super().__init__(*args)
            captured.append(self)

    distribution.DistributionObserver = Capture
    try:
        label = distribution.run_day(arguments)
    finally:
        distribution.DistributionObserver = original
        for observer in captured:
            observer.close()
    cell = root / "observed" / label
    if {row["stage"] for row in captured[0].actual_events} != {"arrival", "ready", "action", "departure"}:
        raise ValueError("reset lost the actual stop-clock instrumentation")
    for name in ("actual_events", "forecast_events", "requested_actions"):
        pd.DataFrame(getattr(captured[0], name)).to_csv(cell / f"{name}.csv", index=False)
    print(f"SEGMENT_DAY_DONE {label}", flush=True)
    return label


def batch(root, start, end, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        completed = list(pool.map(run_day, [(root, *item) for item in distribution.items()[start:end]]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "done.json").write_text(json.dumps(dict(completed=completed)) + "\n")


def segment_rows(predicted, actual, routes, now):
    keys = ["trip_id", "station_id", "stage"]
    matched = predicted.merge(actual, on=keys, suffixes=("_pred", "_true"), validate="one_to_one")
    rows = []
    for row in matched.itertuples():
        if now < row.time_s_true <= now + 1800:
            lead = "0-300" if row.time_s_true <= now + 300 else "300-900" if row.time_s_true <= now + 900 else "900-1800"
            rows.append(dict(metric=row.stage + "_clock_s", lead=lead, error=row.time_s_pred - row.time_s_true))
            if row.stage == "action":
                for name in ("forward_gap_s", "follower_gap_s", "load", "action_s", "progress_m", "target_headway_s"):
                    left, right = getattr(row, name + "_pred"), getattr(row, name + "_true")
                    if pd.notna(left) and pd.notna(right):
                        rows.append(dict(metric=name, lead=lead, error=left - right))
    pp = predicted.pivot(index=["trip_id", "station_id"], columns="stage", values="time_s")
    aa = actual.pivot(index=["trip_id", "station_id"], columns="stage", values="time_s")
    for route in routes.values():
        ids = route["station_ids"]
        for (trip, station), truth in aa.iterrows():
            if station not in ids or pd.isna(truth.get("arrival")) or not now < truth.arrival <= now + 1800:
                continue
            if (trip, station) not in pp.index:
                continue
            forecast = pp.loc[(trip, station)]
            for name, start, finish in (("service_s", "arrival", "ready"), ("holding_pipeline_s", "ready", "departure")):
                values = [forecast.get(start), forecast.get(finish), truth.get(start), truth.get(finish)]
                if all(pd.notna(value) for value in values):
                    rows.append(dict(metric=name, lead="all", error=(values[1] - values[0]) - (values[3] - values[2])))
            index = ids.index(station)
            if index > 0 and (trip, ids[index - 1]) in pp.index and (trip, ids[index - 1]) in aa.index:
                pdep, adep = pp.loc[(trip, ids[index - 1])].get("departure"), aa.loc[(trip, ids[index - 1])].get("departure")
                if pd.notna(pdep) and pd.notna(adep) and pd.notna(forecast.get("arrival")) and adep >= now:
                    rows.append(dict(metric="travel_s", lead="all", error=(forecast.arrival - pdep) - (truth.arrival - adep)))
    in_window = (matched.time_s_true > now) & (matched.time_s_true <= now + 1800)
    return rows, int(in_window.sum()), len(predicted.loc[(predicted.time_s > now) & (predicted.time_s <= now + 1800)]), len(
        actual.loc[(actual.time_s > now) & (actual.time_s <= now + 1800)])


def aggregate(root):
    rows, counts, requests = [], [], []
    for seed, scenario, cohort in distribution.items():
        cell = root / "observed" / f"seed{seed}_eval{scenario}"
        actual = pd.read_csv(cell / "actual_events.csv")
        predicted = pd.read_csv(cell / "forecast_events.csv")
        snapshots = json.loads((cell / "forecasts.json").read_text())
        request = pd.read_csv(cell / "requested_actions.csv")
        request["unmatched_departure_hold_s"] = np.where(~request.evidence_valid & ~request.masked, request.action_s, 0)
        requests.append(dict(cohort=cohort, train_seed=seed, eval_seed=scenario,
            total_requested_hold_s=request.action_s.sum(), unmatched_departure_hold_s=request.unmatched_departure_hold_s.sum(),
            invalid_context_actions=int((~request.evidence_valid & ~request.masked).sum())))
        for (origin, draw), group in predicted.groupby(["origin", "draw"]):
            pool = snapshots[origin]["pool"]
            errors, matched, n_pred, n_actual = segment_rows(group, actual, pool["routes"], pool["time_s"])
            rows.extend(dict(cohort=cohort, train_seed=seed, eval_seed=scenario, origin=origin, draw=draw, **row) for row in errors)
            counts.append(dict(cohort=cohort, matched=matched, predicted=n_pred, actual=n_actual))
    frame = pd.DataFrame(rows)
    frame["abs_error"] = frame.error.abs()
    out = root / "segment_summary"
    out.mkdir(parents=True, exist_ok=True)
    result = frame.groupby(["cohort", "metric", "lead"]).agg(n=("error", "size"),
        mae=("abs_error", "mean"), bias=("error", "mean"))
    result.to_csv(out / "errors.csv")
    pd.DataFrame(counts).groupby("cohort").mean().to_csv(out / "event_counts.csv")
    pd.DataFrame(requests).to_csv(out / "invalid_departure_holding.csv", index=False)
    frame.to_csv(root / "segment_records.csv", index=False)
    (out / "status.json").write_text(json.dumps(dict(days=8, draws_per_origin=8, policy_changed=False,
        source_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"), matched_v35=True)) + "\n")
    print(result.to_string(), flush=True)
    print(pd.DataFrame(requests).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "batch", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] / "tests"), pattern="test_v4[124]*.py")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        run_day((args.out_root, *distribution.items()[0]))
        (args.out_root / "done.json").write_text('{"matched_v35": true}\n')
    elif args.command == "batch":
        batch(args.out_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root)
