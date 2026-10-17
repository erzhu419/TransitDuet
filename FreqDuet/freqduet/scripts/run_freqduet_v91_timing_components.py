#!/usr/bin/env python3
"""Exact release/realized-holding accounting for the unchanged V89 primary."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v90_conditional_regularity as prior


source, pilot = prior.source, prior.pilot
ITEMS, MODES = prior.ITEMS, prior.MODES
COMPONENTS = ("release", "realized_holding", "other_elapsed")
EVENT_FIELDS = ("trip_id", "station_id", "direction", "time_s", "release", "realized_holding", "other_elapsed")


def day_interval(rows, getter):
    values = [float(np.mean([getter(r) for r in rows if r["day"] == day])) for day in source.DAYS]
    stats = pilot.event.labels.expanded.paired_interval(values, 91000)
    return {k: stats[k] for k in ("n_demand_days", "mean", "ci95")}


def review(root, v90_root):
    rows = [json.loads((v90_root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    def moment(r, mode, metric):
        a = r["audits"][mode]["arrival"]["stop_direction"]
        return a["mean_headway_s"] if metric == "mean_headway_s" else a["cv_squared"] * a["mean_headway_s"] ** 2
    def local(r, mode):
        return r["audits"][mode]["early_minus_late_local_formula"]["more_hold_gain_not_up"]
    result = dict(status="v90_absolute_moments_review_complete",
        absolute={mode: {m: day_interval(rows, lambda r: moment(r, mode, m)) for m in
            ("mean_headway_s", "variance_s2")} for mode in MODES},
        effects={m: day_interval(rows, lambda r: moment(r, "no_gain_early", m) - moment(r, "late", m))
            for m in ("mean_headway_s", "variance_s2")},
        primary_local={mode: dict(
            means={m: day_interval(rows, lambda r: local(r, mode)["means"][m]) for m in
                ("predicted_gain_delta", "completed_ready_gain_delta", "completed_departure_gain_delta", "follower_gap_error_s")},
            ready_reversal_fraction=day_interval(rows, lambda r:
                local(r, mode)["predicted_nonharm_but_completed_ready_harm"] / local(r, mode)["decisions"])) for mode in MODES},
        new_rollouts=0, main_policy_changed=False, original_gate_changed=False,
        scope="day-weighted diagnostics on consumed V89/V90 paths; local formulas are not causal action values")
    pilot.write_json(root / "review/status.json", result)
    print("V91_ABSOLUTE_MOMENTS_REVIEW " + json.dumps(result), flush=True)


def timing_event(launch, holding_counter, time, trip, station, direction):
    held = float(holding_counter) - launch["holding_counter_at_launch"]
    if held < 0:
        raise AssertionError("realized holding counter reset within a trip")
    return dict(trip_id=int(trip), station_id=int(station), direction=bool(direction), time_s=float(time),
        release=launch["time_s"], realized_holding=held,
        other_elapsed=float(time) - launch["time_s"] - held)


def moments(headways, components):
    h, c = np.asarray(headways, dtype=float), np.asarray(components, dtype=float)
    np.testing.assert_allclose(c.sum(1), h, atol=1e-9, rtol=0)
    mean, var = float(h.mean()), float(h.var())
    covariance = ((c - c.mean(0)) * (h - mean)[:, None]).mean(0)
    np.testing.assert_allclose(covariance.sum(), var, atol=1e-8, rtol=1e-12)
    return dict(samples=len(h), mean_s=mean, variance_s2=var, cv_squared=var / mean ** 2,
        component_means_s=dict(zip(COMPONENTS, c.mean(0).tolist())),
        covariance_s2=dict(zip(COMPONENTS, covariance.tolist())))


def timing_series(events):
    groups, headways, parts, bands = defaultdict(list), [], [], defaultdict(lambda: ([], []))
    for row in events:
        group = groups[str((row["station_id"], row["direction"]))]
        if group:
            previous = group[-1]
            h = row["time_s"] - previous["time_s"]
            c = [row[k] - previous[k] for k in COMPONENTS]
            headways.append(h)
            parts.append(c)
            band = bands[str(int(row["time_s"] // 14400))]
            band[0].append(h)
            band[1].append(c)
        group.append(row)
    first = sum(g[0]["time_s"] for g in groups.values())
    last = sum(g[-1]["time_s"] for g in groups.values())
    result = moments(headways, parts)
    np.testing.assert_allclose(last - first, np.sum(headways), atol=1e-8, rtol=0)
    return dict(**result, first_times_sum_s=first, last_times_sum_s=last,
        four_hour_bands={b: moments(*v) for b, v in bands.items()},
        orders={g: [r["trip_id"] for r in rows] for g, rows in groups.items()})


def component_effect(base, candidate):
    mb, mc = base["mean_s"], candidate["mean_s"]
    nb, nc = base["samples"], candidate["samples"]
    result = {k + "_variance_at_base_mean": (candidate["covariance_s2"][k] - base["covariance_s2"][k]) / mb ** 2
              for k in COMPONENTS}
    result["mean_normalization"] = candidate["variance_s2"] * (1. / mc ** 2 - 1. / mb ** 2)
    np.testing.assert_allclose(sum(result.values()), candidate["cv_squared"] - base["cv_squared"], atol=1e-10, rtol=0)
    result["cv_squared"] = candidate["cv_squared"] - base["cv_squared"]
    result["mean_headway_s"] = mc - mb
    result["raw_variance_s2"] = candidate["variance_s2"] - base["variance_s2"]
    result["last_boundary_mean_s"] = (candidate["last_times_sum_s"] - base["last_times_sum_s"]) / nb
    result["first_boundary_mean_s"] = -(candidate["first_times_sum_s"] - base["first_times_sum_s"]) / nb
    result["count_normalization_mean_s"] = mc * nc * (1. / nc - 1. / nb)
    np.testing.assert_allclose(sum(result[k] for k in
        ("last_boundary_mean_s", "first_boundary_mean_s", "count_normalization_mean_s")), mc - mb, atol=1e-9, rtol=0)
    return result


def order_changes(base, candidate):
    def neighbours(orders):
        return {(g, trip): rows[i - 1] if i else None for g, rows in orders.items() for i, trip in enumerate(rows)}
    b, c = neighbours(base), neighbours(candidate)
    common = b.keys() & c.keys()
    return dict(matched=len(common), unmatched_base=len(b.keys() - c.keys()), unmatched_candidate=len(c.keys() - b.keys()),
        changed_predecessor=sum(b[k] != c[k] for k in common))


class TimingAudit(source.ConditionalCross):
    def __init__(self, seed, mode):
        super().__init__(seed, mode)
        self.launches, self.buses, self.arrivals, self.departures = {}, {}, [], []

    def attach(self, runner):
        super().attach(runner)
        self.env = runner.env
        self.recorder = type(self.env.headway_events)
        self.original_arrival = self.recorder.record
        self.original_departure = self.recorder.record_departure
        self.original_launch = self.env.launch_bus

        def launch(trip, actual_launch=None):
            bus = self.original_launch(trip, actual_launch=actual_launch)
            if bus is not None:
                key = int(trip.launch_turn)
                self.buses[key] = bus
                self.launches[key] = dict(time_s=float(self.env.current_time), bus_id=int(bus.bus_id),
                    holding_counter_at_launch=float(bus.episode_hold_vehicle_seconds),
                    readiness_s=max(0., float(self.env.current_time) - trip._freqduet_first_eligible_time))
            return bus

        def capture(trip, station, direction, time):
            bus = self.buses[int(trip)]
            if int(bus.trip_id) != int(trip):
                raise AssertionError("event was not captured on its actual vehicle trip")
            return timing_event(self.launches[int(trip)], bus.episode_hold_vehicle_seconds,
                                time, trip, station, direction)

        def arrival(recorder, station_id, direction, arrival_time_s, trip_id, target_headway_s=None):
            result = self.original_arrival(recorder, station_id, direction, arrival_time_s, trip_id, target_headway_s)
            self.arrivals.append(capture(trip_id, station_id, direction, arrival_time_s))
            return result

        def departure(recorder, station_id, direction, departure_time_s, trip_id):
            result = self.original_departure(recorder, station_id, direction, departure_time_s, trip_id)
            self.departures.append(capture(trip_id, station_id, direction, departure_time_s))
            return result

        self.env.launch_bus = launch
        self.recorder.record, self.recorder.record_departure = arrival, departure

    def detach(self):
        self.env.launch_bus = self.original_launch
        self.recorder.record, self.recorder.record_departure = self.original_arrival, self.original_departure

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        def key(r):
            return r["trip_id"], r["station_id"], r["direction"]
        observed = {key(r) for r in self.arrivals}
        departures = [r for r in self.departures if key(r) in observed]
        if {key(r) for r in departures} != observed:
            raise AssertionError("cleared-service arrival events lack matched departures")
        self.summary = dict(arrival=timing_series(self.arrivals), departure=timing_series(departures),
            departure_support_matches_arrivals=True,
            mean_readiness_s=float(np.mean([v["readiness_s"] for v in self.launches.values()])))
        self.env, self.buses = None, {}

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, event_fields=EVENT_FIELDS,
            arrivals=[[r[k] for k in EVENT_FIELDS] for r in self.arrivals],
            departures=[[r[k] for k in EVENT_FIELDS] for r in self.departures],
            launch_fields=("trip_id", "bus_id", "time_s", "readiness_s"),
            launches=[[trip, r["bus_id"], r["time_s"], r["readiness_s"]] for trip, r in self.launches.items()])


def replay(root, seed, day, rep, mode):
    rule, credit = TimingAudit(seed, mode), pilot.CreditObserver()
    row, events, _, holds = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day + rep,
        [0.] * 4, f"v91_seed{seed}_day{day}_r{rep}_{mode}", probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s)
    np.testing.assert_allclose(rule.summary["arrival"]["cv_squared"], row["cv_squared"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(rule.summary["mean_readiness_s"], row["readiness_delay_s"], atol=1e-10, rtol=0)
    return row, events, holds, rule


def run_cell(root, v89_root, index, check_paths=False):
    path = root / "pairs" / f"p{index}.json"
    if path.exists() and not check_paths:
        return
    seed, day, rep = ITEMS[index]
    old = json.loads((v89_root / "pairs" / f"p{index}.json").read_text())
    summaries = {}
    for mode in MODES:
        row, events, holds, rule = replay(root, seed, day, rep, mode)
        for metric in (*source.experiment.METRICS, "action_counts"):
            pilot.joint.assert_same_state(old["results"][mode][metric], row[metric], "v91_original_v89_endpoints")
        pilot.joint.assert_same_state(old["contexts"][mode], rule.context_summary, "v91_original_v89_contexts")
        if check_paths:
            native = source.replay(root, seed, day, rep, mode)
            pilot.joint.assert_same_state(native[1], events, "v91_passive_events")
            pilot.joint.assert_same_state(native[2], holds, "v91_passive_holding")
        summaries[mode] = rule.summary
        rule.save(root / "records" / f"p{index}_{mode}.npz")
    pilot.write_json(path, dict(index=index, seed=seed, day=day, rep=rep, summaries=summaries,
        effects={kind: component_effect(summaries["late"][kind], summaries["no_gain_early"][kind])
                 for kind in ("arrival", "departure")},
        order_changes={kind: order_changes(summaries["late"][kind]["orders"], summaries["no_gain_early"][kind]["orders"])
                       for kind in ("arrival", "departure")}, exact_original_endpoints=True, full_paths_checked=check_paths))
    print("V91_TIMING_COMPONENTS_CELL_DONE " + str(index), flush=True)


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    def describe(chosen):
        return {kind: dict(
            effects={k: day_interval(chosen, lambda r: r["effects"][kind][k]) for k in chosen[0]["effects"][kind]},
            order_changes={k: sum(r["order_changes"][kind][k] for r in chosen) for k in chosen[0]["order_changes"][kind]},
            four_hour_effects={band: {metric: day_interval(chosen, lambda r:
                r["summaries"]["no_gain_early"][kind]["four_hour_bands"][band][metric] -
                r["summaries"]["late"][kind]["four_hour_bands"][band][metric])
                for metric in ("mean_s", "variance_s2", "cv_squared", "samples")} for band in ("0", "1", "2", "3")})
            for kind in ("arrival", "departure")}
    result = dict(status="timing_components_diagnosis_complete", combined=describe(rows),
        by_seed={str(seed): describe([r for r in rows if r["seed"] == seed]) for seed in pilot.SEEDS},
        rollouts=2 * len(rows), exact_original_endpoints=True, actor_updates=0, critic_updates=0, dual_updates=0,
        main_policy_changed=False, original_gate_changed=False, controller_admitted=False,
        raw_records="compressed timing primitives stay server-side for subsequent analysis",
        scope="consumed V89 days; covariance-allocated variance and finite-span identities, not causal mediation; other elapsed includes travel/service/clock")
    pilot.write_json(root / "summary/status.json", result)
    print("V91_TIMING_COMPONENTS_COMPLETE " + json.dumps(result["combined"]["arrival"]["effects"]), flush=True)


def smoke(root, v89_root, v90_root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v91_*.py", "-v"], check=True)
    review(root, v90_root)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(run_cell, root, v89_root, ITEMS.index((seed, source.DAYS[0], 0)), True)
                     for seed in pilot.SEEDS]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="exact_paths_actual_holding_timing_variance_and_boundary_accounting_passed"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v89-root", type=Path)
    parser.add_argument("--v90-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v89_root, args.v90_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_cell, args.out_root, args.v89_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))
