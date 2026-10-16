#!/usr/bin/env python3
"""Replay a fixed V81 subset to diagnose local versus full-day regularity."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

from freqduet_regularity_projection import RegularityProjection
import run_freqduet_v81_regularity_projection as projection


pilot, step = projection.pilot, projection.pilot.step
DAYS = projection.DAYS[::8]
ITEMS = [i for i, (_, day, _) in enumerate(projection.ITEMS) if day in DAYS]
LOCAL_METRICS = ("predicted_gain_delta", "expected_holding_delta_s", "completed_ready_gain_delta",
    "completed_departure_gain_delta", "follower_gap_error_s", "future_follower_holding_s",
    "modal_gain_delta", "modal_holding_delta_s", "completed_ready_modal_gain_delta")


def completed_gain(context, gap):
    # Post-episode replay of the local formula, not a causal alternative-action label.
    return step.action_features(replace(context, follower_departure_gap_s=max(float(gap), 0.)),
                                0., 0, 0, 0.)[:, 0]


def local_effect(context, features, base, actual, forecast):
    delta = actual - base
    row = dict(predicted_gain_delta=float(delta @ features[:, 0]),
               expected_holding_delta_s=float(delta @ step.ACTIONS),
               completed_ready_gain_delta=None, completed_departure_gain_delta=None,
               follower_gap_error_s=None, future_follower_holding_s=None,
               modal_gain_delta=float(features[np.argmax(actual), 0] - features[np.argmax(base), 0]),
               modal_holding_delta_s=float(step.ACTIONS[np.argmax(actual)] - step.ACTIONS[np.argmax(base)]),
               completed_ready_modal_gain_delta=None)
    if context.evidence_valid and context.follower_evidence_valid and forecast and forecast["resolved"]:
        row["completed_ready_gain_delta"] = float(delta @ completed_gain(context, forecast["actual_follower_gap_s"]))
        gains = completed_gain(context, forecast["actual_follower_gap_s"])
        row["completed_ready_modal_gain_delta"] = float(gains[np.argmax(actual)] - gains[np.argmax(base)])
        row["follower_gap_error_s"] = float(forecast["raw_gap_prediction_error_s"])
        row["future_follower_holding_s"] = forecast.get("follower_action_s")
        if forecast["follower_departure_time_s"] is not None:
            gap = forecast["follower_departure_time_s"] - forecast["decision_time_s"]
            row["completed_departure_gain_delta"] = float(delta @ completed_gain(context, gap))
    return row


def arrival_decomposition(events, key):
    events = [e for e in events if e["headway_s"] is not None]
    values = np.asarray([e["headway_s"] for e in events], dtype=float)
    mean, groups = float(values.mean()), defaultdict(list)
    for event in events:
        groups[str(key(event))].append(float(event["headway_s"]))
    within = sum(len(v) * np.var(v) for v in groups.values()) / len(values) / mean ** 2
    between = sum(len(v) * (np.mean(v) - mean) ** 2 for v in groups.values()) / len(values) / mean ** 2
    cv = float(values.var() / mean ** 2)
    np.testing.assert_allclose(within + between, cv, rtol=1e-12, atol=1e-12)
    return dict(samples=len(values), mean_headway_s=mean, cv_squared=cv,
        within_group_cv_squared=float(within), between_group_cv_squared=float(between),
        group_cv_contributions={k: float(np.square((np.asarray(v) - mean) / mean).sum() / len(values))
                                for k, v in groups.items()})


def summarize_local(rows):
    return dict(decisions=len(rows),
        observed={m: sum(r[m] is not None for r in rows) for m in LOCAL_METRICS},
        means={m: float(np.mean([r[m] for r in rows if r[m] is not None]))
               if any(r[m] is not None for r in rows) else None for m in LOCAL_METRICS},
        predicted_nonharm_but_completed_ready_harm=sum(r["predicted_gain_delta"] >= -2e-12
            and r["completed_ready_gain_delta"] is not None and r["completed_ready_gain_delta"] < -1e-10 for r in rows))


class LocalGlobalAudit:
    def __init__(self, label):
        self.label, self.laws = label, []
        self.projection = RegularityProjection()
        self.summary = None
        self.ready, self.departures, self.followers = {}, {}, {}

    def attach(self, runner):
        from env.bus import Bus

        self.bus_type = Bus
        self.original_prepare = Bus._prepare_for_action
        self.recorder_type = type(runner.env.headway_events)
        self.original_departure = self.recorder_type.record_departure

        def prepare(bus, now, buses, debug):
            result = self.original_prepare(bus, now, buses, debug)
            key = (int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction))
            self.ready[key] = float(now)
            follower = next((b for b in buses if b.bus_id == bus.pre_action_follower_bus_id), None)
            self.followers[key] = int(follower.trip_id) if follower is not None else None
            return result

        def depart(recorder, station_id, direction, departure_time_s, trip_id):
            self.departures[(int(trip_id), int(station_id), bool(direction))] = float(departure_time_s)
            return self.original_departure(recorder, station_id, direction, departure_time_s, trip_id)

        Bus._prepare_for_action = prepare
        self.recorder_type.record_departure = depart

    def detach(self):
        self.bus_type._prepare_for_action = self.original_prepare
        self.recorder_type.record_departure = self.original_departure

    def __call__(self, policy, states, features, theta):
        before = self.projection.active
        law = (self.projection(policy, states, features, theta) if self.label == "projected"
               else step.probabilities(policy, states, features, theta))
        actual = law[0].numpy().astype(float)
        base = step.probabilities(policy, states, features, np.zeros(4))[0].numpy().astype(float)
        self.laws.append((base / base.sum(), actual / actual.sum(), self.projection.active > before))
        return law

    def finish(self, runner, records, holds):
        recorder, groups = runner.env.headway_events, defaultdict(list)
        holding = {(r["trip_id"], r["station_id"]): r["action_s"] for r in holds}
        if len(records) != len(self.laws):
            raise AssertionError("trace decisions and probability laws do not align")
        for record, (base, actual, active) in zip(records, self.laws):
            context = record["context"]
            key = (record["trip_id"], record["station_id"], record["direction"])
            follower = (self.followers[key], record["station_id"], record["direction"])
            ready, forecast = self.ready.get(follower), None
            if ready is not None and key in self.departures:
                gap = ready - self.ready[key]
                forecast = dict(resolved=True, actual_follower_gap_s=gap,
                    raw_gap_prediction_error_s=float(context.follower_departure_gap_s) - gap
                        if context.follower_evidence_valid else None,
                    follower_action_s=holding.get(follower[:2]),
                    follower_departure_time_s=self.departures.get(follower), decision_time_s=self.ready[key])
            effect = local_effect(context, record["features"], base, actual, forecast)
            bucket = ("bilateral_active" if active else "bilateral_inactive") if (
                context.evidence_valid and context.follower_evidence_valid) else "missing_bilateral"
            groups[bucket].append(effect)
        self.summary = dict(local={name: summarize_local(rows) for name, rows in groups.items()},
            arrival={"stop_direction": arrival_decomposition(recorder.events,
                lambda e: (e["station_id"], e["direction"])),
                "four_hour_band": arrival_decomposition(recorder.events, lambda e: int(e["arrival_time_s"] // 14400))})


def run_item(root, v81_root, v78_root, index):
    path = root / "pairs" / f"p{index}.json"
    if path.exists():
        return index
    original_index = ITEMS[index]
    seed, day, (mode, rep) = projection.ITEMS[original_index]
    original = json.loads((v81_root / "pairs" / f"p{original_index}.json").read_text())
    theta = json.loads((v78_root / "adapters" / f"seed{seed}.json").read_text())["theta"]
    sampler = 181000000 + seed + 100 * (day - projection.DAYS[0]) + rep
    audits = {}
    for label in ("base", "raw", "projected"):
        observer = LocalGlobalAudit(label)
        row, _, _, _ = pilot.rollout(root, seed, day, mode, sampler,
            [0.] * 4 if label == "base" else theta, f"p{index}_{label}",
            probability_rule=observer, trace_observer=observer)
        for metric in pilot.METRICS:
            np.testing.assert_equal(row[metric], original["results"][label][metric])
        if row["action_counts"] != original["results"][label]["action_counts"]:
            raise AssertionError("trace instrumentation changed an action histogram")
        np.testing.assert_allclose(observer.summary["arrival"]["stop_direction"]["cv_squared"],
                                   row["cv_squared"], rtol=1e-12, atol=1e-12)
        if not any(g["observed"]["completed_departure_gain_delta"] for g in observer.summary["local"].values()):
            raise AssertionError("full-day trace contains no completed bilateral follower labels")
        audits[label] = observer.summary
    pilot.write_json(path, dict(original_v81_index=original_index, train_seed=seed, scenario_seed=day,
        mode=mode, rep=rep, audits=audits, all_v81_metrics_and_action_counts_exact=True))
    print("V82_LOCAL_GLOBAL_CELL_DONE " + str(index), flush=True)
    return index


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    result = {}
    for seed in pilot.SEEDS:
        result[str(seed)] = {}
        for mode in ("stochastic", "deterministic"):
            chosen = [r for r in rows if (r["train_seed"], r["mode"]) == (seed, mode)]
            policies = {}
            for label in ("base", "raw", "projected"):
                local = {}
                for group in ("bilateral_active", "bilateral_inactive", "missing_bilateral"):
                    entries = [r["audits"][label]["local"][group] for r in chosen if group in r["audits"][label]["local"]]
                    if not entries:
                        continue
                    counts = {m: sum(e["observed"][m] for e in entries) for m in LOCAL_METRICS}
                    local[group] = dict(decisions=sum(e["decisions"] for e in entries), observed=counts,
                        means={m: sum(e["means"][m] * e["observed"][m] for e in entries if e["observed"][m]) / counts[m]
                               if counts[m] else None for m in LOCAL_METRICS},
                        predicted_nonharm_but_completed_ready_harm=sum(e["predicted_nonharm_but_completed_ready_harm"] for e in entries))
                arrival = {}
                for partition in ("stop_direction", "four_hour_band"):
                    entries = [r["audits"][label]["arrival"][partition] for r in chosen]
                    arrival[partition] = {m: float(np.mean([e[m] for e in entries])) for m in
                        ("mean_headway_s", "cv_squared", "within_group_cv_squared", "between_group_cv_squared")}
                    arrival[partition]["group_cv_contributions"] = {k: float(np.mean([
                        e["group_cv_contributions"].get(k, 0.) for e in entries]))
                        for k in sorted(set().union(*(e["group_cv_contributions"] for e in entries)))}
                policies[label] = dict(local=local, arrival=arrival)
            result[str(seed)][mode] = policies
    pilot.write_json(root / "summary" / "status.json", dict(status="local_global_regularity_diagnosis_complete",
        days=DAYS, cells=len(rows), rollouts=3 * len(rows), by_seed=result,
        original_endpoints_and_action_counts_exact=True, new_controller_fits=0, actor_updates=0,
        dual_updates=0, main_policy_changed=False, v81_gate_changed=False,
        scope="four evenly spaced consumed V81 days, descriptive mechanism evidence only",
        completed_gap_replay_is_not_an_action_counterfactual=True))
    print("V82_LOCAL_GLOBAL_DIAGNOSIS_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v81-root", type=Path)
    parser.add_argument("--v78-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
            str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v82_*.py", "-v"], check=True)
        run_item(args.out_root, args.v81_root, args.v78_root, 0)
        pilot.write_json(args.out_root / "smoke" / "done.json", dict(status="tests_and_three_policy_exact_replay_passed"))
    elif args.command == "run":
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(run_item, args.out_root, args.v81_root, args.v78_root, i)
                       for i in range(args.start, args.end)]
            for task in pending:
                task.result()
        pilot.write_json(args.out_root / "shards" / f"{args.start}-{args.end}.json", dict(status="complete"))
    else:
        aggregate(args.out_root)


if __name__ == "__main__":
    main()
