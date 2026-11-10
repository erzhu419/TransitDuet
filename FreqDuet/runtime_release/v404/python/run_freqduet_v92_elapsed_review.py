#!/usr/bin/env python3
"""Read existing V91 timing primitives; no simulator rollout or policy update."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v91_timing_components as prior


pilot, source = prior.pilot, prior.source
COMPONENTS = ("release", "realized_holding", "observed_nonholding_stop", "remaining_elapsed")


def key(row):
    return int(row["trip_id"]), int(row["station_id"]), bool(row["direction"])


def read_records(path):
    with np.load(path) as data:
        fields = data["event_fields"].tolist()
        return {kind: [dict(zip(fields, row.tolist())) for row in data[kind]] for kind in ("arrivals", "departures")}


def split_stop_time(records):
    departures = {key(r): r for r in records["departures"]}
    cumulative, last_departure = defaultdict(float), {}
    split = {}
    for arrival in records["arrivals"]:
        k = key(arrival)
        trip = k[0]
        departure = departures[k]
        if arrival["time_s"] < last_departure.get(trip, -np.inf):
            raise AssertionError("a trip reached its next stop before leaving the previous stop")
        nonholding = departure["time_s"] - arrival["time_s"] - (
            departure["realized_holding"] - arrival["realized_holding"])
        if nonholding < 0:
            raise AssertionError("executed holding exceeds the observed stop interval")
        before = cumulative[trip]
        cumulative[trip] += nonholding
        for kind, row, service in (("arrival", arrival, before), ("departure", departure, cumulative[trip])):
            coordinates = [row["release"], row["realized_holding"], service, row["other_elapsed"] - service]
            np.testing.assert_allclose(sum(coordinates), row["time_s"], atol=1e-9, rtol=0)
            split[k, kind] = coordinates
        last_departure[trip] = departure["time_s"]
    return split


def covariance_moments(headways, components):
    h, c = np.asarray(headways), np.asarray(components)
    np.testing.assert_allclose(c.sum(1), h, atol=1e-9, rtol=0)
    centered = c - c.mean(0)
    covariance = centered.T @ centered / len(c)
    np.testing.assert_allclose(covariance.sum(), h.var(), atol=1e-8, rtol=1e-12)
    return dict(samples=len(h), mean=float(h.mean()), variance=float(h.var()),
        covariance=covariance, cv_squared=float(h.var() / h.mean() ** 2))


def series(records, split, kind):
    allowed = {key(r) for r in records["arrivals"]}
    previous, edges, headways, parts = {}, {}, [], []
    first, last = np.zeros(4), {}
    for row in records[kind + "s"]:
        k = key(row)
        if k not in allowed:
            continue
        group = k[1:]
        coordinates = np.asarray(split[k, kind])
        if group in previous:
            prev_key, prev_time, prev_coordinates = previous[group]
            h = row["time_s"] - prev_time
            headways.append(h)
            parts.append(coordinates - prev_coordinates)
            edges[k] = dict(predecessor=prev_key, headway=h)
        else:
            first += coordinates
        previous[group] = k, row["time_s"], coordinates
        last[group] = coordinates
    result = covariance_moments(headways, parts)
    last = np.sum(list(last.values()), axis=0)
    np.testing.assert_allclose((last - first).sum(), np.sum(headways), atol=1e-8, rtol=0)
    return dict(**result, edges=edges, first=first, last=last)


def edge_contributions(base, candidate):
    b, c = base["edges"], candidate["edges"]
    keys = b.keys() | c.keys()
    changed = {k for k in keys if k not in b or k not in c or b[k]["predecessor"] != c[k]["predecessor"]}
    result = {}
    for name, selected in (("changed_predecessor", changed), ("same_predecessor", keys - changed)):
        def contribution(s, table):
            return sum((table[k]["headway"] - s["mean"]) ** 2 for k in selected if k in table) / s["samples"] / s["mean"] ** 2
        result[name + "_cv_squared"] = contribution(candidate, c) - contribution(base, b)
    np.testing.assert_allclose(sum(result.values()), candidate["cv_squared"] - base["cv_squared"], atol=1e-10, rtol=0)
    return result


def effects(base, candidate):
    delta = (candidate["covariance"] - base["covariance"]) / base["mean"] ** 2
    result = {name + "_allocated": float(delta[i].sum()) for i, name in enumerate(COMPONENTS)}
    result.update({name + "_diagonal": float(delta[i, i]) for i, name in enumerate(COMPONENTS)})
    result.update({COMPONENTS[i] + "__" + COMPONENTS[j] + "_cross": float(2 * delta[i, j])
                   for i in range(4) for j in range(i + 1, 4)})
    result["mean_normalization"] = candidate["variance"] * (1 / candidate["mean"] ** 2 - 1 / base["mean"] ** 2)
    result["cv_squared"] = candidate["cv_squared"] - base["cv_squared"]
    np.testing.assert_allclose(sum(result[name + "_allocated"] for name in COMPONENTS) + result["mean_normalization"],
                               result["cv_squared"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(sum(v for k, v in result.items() if k.endswith(("_diagonal", "_cross"))),
                               delta.sum(), atol=1e-10, rtol=0)
    for endpoint in ("first", "last"):
        sign = -1 if endpoint == "first" else 1
        for i, name in enumerate(COMPONENTS):
            result[endpoint + "_" + name + "_mean_s"] = float(sign * (candidate[endpoint][i] - base[endpoint][i]) / base["samples"])
    result["count_normalization_mean_s"] = candidate["mean"] * (1 - candidate["samples"] / base["samples"])
    np.testing.assert_allclose(sum(v for k, v in result.items() if k.endswith("_mean_s")),
                               candidate["mean"] - base["mean"], atol=1e-9, rtol=0)
    result.update(edge_contributions(base, candidate))
    return result


def analyze_cell(root, index):
    old = json.loads((root / "pairs" / f"p{index}.json").read_text())
    native = {}
    for mode in prior.MODES:
        records = read_records(root / "records" / f"p{index}_{mode}.npz")
        split = split_stop_time(records)
        native[mode] = {kind: series(records, split, kind) for kind in ("arrival", "departure")}
        for kind, result in native[mode].items():
            previous = old["summaries"][mode][kind]
            np.testing.assert_allclose([result["mean"], result["variance"], result["cv_squared"]],
                [previous["mean_s"], previous["variance_s2"], previous["cv_squared"]], atol=1e-8, rtol=1e-12)
            allocated = result["covariance"].sum(1)
            expected = previous["covariance_s2"]
            np.testing.assert_allclose(allocated[:2], [expected[k] for k in prior.COMPONENTS[:2]], atol=1e-8, rtol=1e-12)
            np.testing.assert_allclose(allocated[2:].sum(), expected["other_elapsed"], atol=1e-8, rtol=1e-12)
    return dict(index=index, seed=old["seed"], day=old["day"], rep=old["rep"],
        effects={kind: effects(native["late"][kind], native["no_gain_early"][kind]) for kind in ("arrival", "departure")})


def intervals(rows):
    result = {}
    for kind in ("arrival", "departure"):
        result[kind] = {}
        for metric in rows[0]["effects"][kind]:
            values = [float(np.mean([r["effects"][kind][metric] for r in rows if r["day"] == day])) for day in source.DAYS]
            stats = pilot.event.labels.expanded.paired_interval(values, 92000)
            result[kind][metric] = {k: stats[k] for k in ("n_demand_days", "mean", "ci95")}
    return result


def run(root, out):
    rows = [analyze_cell(root, i) for i in range(len(prior.ITEMS))]
    result = dict(status="elapsed_components_readonly_review_complete", combined=intervals(rows),
        by_seed={str(seed): intervals([r for r in rows if r["seed"] == seed]) for seed in pilot.SEEDS},
        record_paths=2 * len(rows), all_v91_moments_and_parent_allocations_reproduced=True,
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        controller_admitted=False, original_gate_changed=False,
        scope="consumed V91 paths; covariance and endpoint accounting, not causal mediation",
        observed_stop_definition="departure minus arrival elapsed, less executed holding, accumulated over matched observed stops",
        remaining_definition="other elapsed minus observed nonholding stop time; includes movement and unobserved terminal/clock effects")
    pilot.write_json(out / "summary/status.json", result)
    pilot.write_json(out / "day_details.json", rows)
    print("V92_ELAPSED_REVIEW_COMPLETE " + json.dumps(result["combined"]["arrival"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v91-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v92_*.py", "-v"], check=True)
    run(args.v91_root, args.out_root)
