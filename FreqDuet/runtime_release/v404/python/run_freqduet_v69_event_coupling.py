#!/usr/bin/env python3
"""Variance-only pilot: logical-event categorical CRN versus V68 stream CRN."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v68_repeated_continuations as repeated


labels, joint = repeated.labels, repeated.joint
CONTEXTS = [i for i, (reference, window) in enumerate(labels.CONTEXTS)
            if window == labels.REFERENCES[reference][1] % len(labels.TARGETS)]
ITEMS = list(itertools.product(CONTEXTS, repeated.FUTURES))
SMOKE_ITEMS = (0, 16 * len(repeated.FUTURES))
METRICS = ("passenger_person_s", "cv_squared", "readiness_delay_s", "holding_s_per_trip")


def categorical_index(probabilities, uniform):
    cdf = np.cumsum(probabilities, dtype=np.float64)
    cdf /= cdf[-1]
    return int(np.searchsorted(cdf, uniform, side="right"))


class EventUniforms:
    """Independent uniforms indexed by trip/stop/direction/local occurrence."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.counts = defaultdict(int)
        self.records = {}

    def draw(self, trip, stop, direction):
        base = (int(trip), int(stop), int(direction))
        occurrence = self.counts[base]
        self.counts[base] += 1
        key = (*base, occurrence)
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, *key]))
        return key, float(generator.random())

    def install(self, runner, policy):
        def get_action(state, deterministic=False):
            tensor = torch.as_tensor(state, dtype=torch.float32)[None, :]
            with torch.no_grad():
                probabilities = policy.dist_info(tensor)[0][0].numpy()
            if deterministic:
                index = int(np.argmax(probabilities))
            else:
                bus = runner._bus_for_agent(runner._label_agent)
                key, uniform = self.draw(bus.trip_id, bus.last_station.station_id, bus.direction)
                index = categorical_index(probabilities, uniform)
                self.records[key] = dict(uniform=uniform, order=len(self.records),
                                         time_s=float(runner.env.current_time),
                                         action_s=float(policy.action_bins[index].item()))
            return np.asarray(policy.action_bins[index].item(), dtype=np.float32)

        policy.get_action = get_action


def alignment(native, alternative):
    common = native.keys() & alternative.keys()
    mismatches = sum(native[k]["uniform"] != alternative[k]["uniform"] for k in common)
    if mismatches:
        raise AssertionError("a common logical policy event received different uniforms")
    return dict(common_events=len(common), uniform_mismatches=mismatches,
                order_shifted=sum(native[k]["order"] != alternative[k]["order"] for k in common),
                time_shifted=sum(native[k]["time_s"] != alternative[k]["time_s"] for k in common),
                action_differences=sum(native[k]["action_s"] != alternative[k]["action_s"] for k in common),
                native_only=len(native.keys() - alternative.keys()),
                alternative_only=len(alternative.keys() - native.keys()))


def old_group_path(root, context, future):
    return repeated.group_path(root, repeated.ITEMS.index((context, future)))


def group_path(root, index):
    context, future = ITEMS[index]
    return root / "groups" / f"c{context}_f{future}.json"


def run_group(root, v59_root, v67_root, v68_root, index, check_noop=False):
    path = group_path(root, index)
    if path.exists():
        return path.stem
    context, future = ITEMS[index]
    spec = labels.context_spec(context)
    seed = repeated.future_seed(context, future)
    original = torch.load(labels.reference_path(v67_root, spec["reference_index"]),
                          map_location="cpu", weights_only=False)["captures"][spec["target_time_s"]]
    old = json.loads(old_group_path(v68_root, context, future).read_text())
    joint.assert_same_state(old["native_action_s"], original["actor_action_s"], "native_current_action")
    joint.assert_same_state(old["future_seed"], seed, "reused_exogenous_future")

    def replay(action, suffix):
        uniforms = EventUniforms(seed + 10000000)

        def continuation(runner, policy):
            startup = repeated.fork_future(runner, policy, seed)
            uniforms.install(runner, policy)
            return dict(startup, event_uniform_seed=uniforms.seed)

        result = labels.run_branch(root, v59_root, spec["reference_index"], (spec["target_time_s"],),
                                   f"c{context}_f{future}_{suffix}", action, continuation=continuation)
        labels.same_prefix(original, result["captures"][spec["target_time_s"]])
        if result["summary"]["scenario_seed"] != seed:
            raise AssertionError("event coupling changed the registered exogenous tape")
        print("V69_ARM_DONE " + json.dumps(dict(context_index=context, future_index=future, arm=suffix)), flush=True)
        return result, uniforms.records

    native, native_records = replay(None, "native")
    native_action = original["actor_action_s"]
    rows = []
    for action in joint.ACTIONS:
        result, records = (native, native_records) if action == native_action else replay(action, f"a{action}")
        joint.assert_same_state(native["captures"][spec["target_time_s"]]["continuation"],
                                result["captures"][spec["target_time_s"]]["continuation"], "common_future_start")
        rows.append(dict(action_s=action, **{k: result["summary"][k] for k in labels.OUTCOMES},
                         conservation_max_error=result["summary"]["stream_audit"]["conservation_max_error"],
                         alignment=None if action == native_action else alignment(native_records, records),
                         future_policy_decisions=len(records),
                         repeat_trip_stop_calls=sum(key[-1] > 0 for key in records)))
    if check_noop:
        forced, forced_records = replay(native_action, "forced_native")
        for key in ("summary", "holds", "upper_path"):
            joint.assert_same_state(native[key], forced[key], "forced_native_" + key)
        joint.assert_same_state(native_records, forced_records, "forced_native_event_uniforms")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(**spec, future_index=future, future_seed=seed,
                                    native_action_s=native_action, rows=rows, exact_prefix=True,
                                    common_future_start=True, forced_native_verified=check_noop,
                                    policy_law="categorical_inverse_cdf_independent_event_uniforms",
                                    diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")), indent=2) + "\n")
    print("V69_GROUP_DONE " + json.dumps(dict(context_index=context, future_index=future, rollouts=7)), flush=True)
    return path.stem


def compact_interval(contexts, getter):
    result = repeated.day_intervals(contexts, getter, 69000)
    return {k: result[k] for k in ("n_demand_days", "mean", "ci95")}


def variance_comparison(contexts, metric):
    event, stream = [np.asarray([np.mean([c["noise"][mode][metric] for c in contexts if c["scenario_seed"] == day])
                                for day in labels.DEMAND_SEEDS]) for mode in ("event", "stream")]
    indices = np.random.default_rng(69001).integers(len(event), size=(10000, len(event)))
    ratios = event[indices].mean(axis=1) / stream[indices].mean(axis=1)
    return dict(event_noise_variance=float(event.mean()), stream_noise_variance=float(stream.mean()),
                ratio=float(event.mean() / stream.mean()), ratio_ci95=np.quantile(ratios, [.025, .975]).tolist(),
                difference=compact_interval(contexts, lambda c: c["noise"]["event"][metric] - c["noise"]["stream"][metric]))


def aggregate(root, v67_root, v68_root):
    contexts, groups = [], []
    for context in CONTEXTS:
        spec = labels.context_spec(context)
        event = [json.loads(group_path(root, ITEMS.index((context, future))).read_text()) for future in repeated.FUTURES]
        stream = [json.loads(old_group_path(v68_root, context, future).read_text()) for future in repeated.FUTURES]
        groups.extend(event)
        for mode in (event, stream):
            if [g["future_index"] for g in mode] != list(repeated.FUTURES):
                raise ValueError("the coupling pilot lacks its eight matched futures")
            if any([r["action_s"] for r in g["rows"]] != list(joint.ACTIONS) for g in mode):
                raise ValueError("the coupling pilot lacks a full seven-action grid")
        native_index = joint.ACTIONS.index(event[0]["native_action_s"])
        arrays = {mode: {metric: np.asarray([[row[metric] for row in g["rows"]] for g in rows], dtype=float)
                         for metric in repeated.RESPONSE_METRICS} for mode, rows in (("event", event), ("stream", stream))}
        old = [json.loads((v67_root / "labels" / f"c{context}_a{a}.json").read_text()) for a in joint.ACTIONS]
        selected = repeated.original_oracle(old, native_index)
        halves = {mode: [repeated.evaluate_selection(a, native_index, np.arange(4), np.arange(4, 8)),
                         repeated.evaluate_selection(a, native_index, np.arange(4, 8), np.arange(4))]
                  for mode, a in arrays.items()}
        contexts.append(dict(**spec,
            noise={mode: {metric: repeated.signal_noise(a - a[:, :1])["single_future_noise_variance"]
                          for metric, a in values.items()} for mode, values in arrays.items()},
            native_difference={metric: float((arrays["event"][metric][:, native_index]
                                              - arrays["stream"][metric][:, native_index]).mean()) for metric in METRICS},
            v67_replication={mode: {metric: float((a[:, selected] - a[:, native_index]).mean()) for metric, a in values.items()}
                             for mode, values in arrays.items()},
            cross_future={mode: {metric: float(np.mean([half[metric] for half in outcomes])) for metric in METRICS}
                          for mode, outcomes in halves.items()},
            signal_noise={mode: {metric: repeated.signal_noise(a - a[:, :1]) for metric, a in values.items()}
                          for mode, values in arrays.items()}))
    alignment_rows = [row["alignment"] for group in groups for row in group["rows"] if row["alignment"] is not None]
    aligned = {key: sum(row[key] for row in alignment_rows) for key in alignment_rows[0]}
    result = dict(status="event_coupling_pilot_complete", contexts=len(contexts), demand_day_clusters=8,
                  original_training_seeds=2, futures_per_context=8, new_formal_rollouts=len(groups) * 7,
                  reused_stream_rollouts=len(groups) * 7, exact_prefixes=all(g["exact_prefix"] for g in groups),
                  common_future_start=all(g["common_future_start"] for g in groups),
                  conservation_max_error=max(r["conservation_max_error"] for g in groups for r in g["rows"]),
                  forced_native_smoke_groups=sum(g["forced_native_verified"] for g in groups), alignment=aligned,
                  max_repeat_trip_stop_calls=max(r["repeat_trip_stop_calls"] for g in groups for r in g["rows"]),
                  variance={m: variance_comparison(contexts, m) for m in METRICS},
                  native_mean_checks={m: compact_interval(contexts, lambda c, m=m: c["native_difference"][m]) for m in METRICS},
                  v67_replication={mode: {m: compact_interval(contexts, lambda c, n=mode, m=m: c["v67_replication"][n][m])
                                         for m in METRICS} for mode in ("event", "stream")},
                  cross_future={mode: {m: compact_interval(contexts, lambda c, n=mode, m=m: c["cross_future"][n][m])
                                       for m in METRICS} for mode in ("event", "stream")},
                  signal_noise={mode: {m: {s: compact_interval(contexts, lambda c, n=mode, m=m, s=s: c["signal_noise"][n][m][s])
                                            for s in ("debiased_signal_squared", "half_sign_agreement")}
                                       for m in ("passenger_person_s", "cv_squared")} for mode in ("event", "stream")},
                  actor_updates=0, dual_updates=0, value_updates=0, main_policy_changed=False,
                  development_prefixes_only=True, policy_promotion=False)
    output = root / "summary"
    output.mkdir(exist_ok=True)
    (root / "groups" / "context_audit.json").write_text(json.dumps(contexts, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V69_AGGREGATE " + json.dumps(result), flush=True)


def smoke(root, v59_root, v67_root, v68_root, workers):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                    str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v6[0-9]_*.py", "-v"], check=True)
    context, future = ITEMS[0]
    spec = labels.context_spec(context)
    old = json.loads(old_group_path(v68_root, context, future).read_text())
    unchanged = labels.run_branch(root, v59_root, spec["reference_index"], (spec["target_time_s"],), "stream_identity",
                                  continuation=lambda r, p: repeated.fork_future(r, p, repeated.future_seed(context, future)))
    native = next(row for row in old["rows"] if row["action_s"] == old["native_action_s"])
    for key in labels.OUTCOMES:
        joint.assert_same_state(native[key], unchanged["summary"][key], "V68_stream_identity_" + key)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(run_group, itertools.repeat(root), itertools.repeat(v59_root), itertools.repeat(v67_root),
                      itertools.repeat(v68_root), SMOKE_ITEMS, (True, True)))
    if not all(json.loads(group_path(root, i).read_text())["forced_native_verified"] for i in SMOKE_ITEMS):
        raise AssertionError("smoke lacks both forced-native event-coupling checks")
    output = root / "smoke_export"
    output.mkdir(exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(status="event_law_prefix_and_native_identity_pass",
                                                     rollouts=17, cached_formal_rollouts=14)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path)
    parser.add_argument("--v67-root", type=Path, required=True)
    parser.add_argument("--v68-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v59_root, args.v67_root, args.v68_root, args.workers)
    elif args.command == "aggregate":
        aggregate(args.out_root, args.v67_root, args.v68_root)
    else:
        if not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("batch lies outside the fixed event-coupling pilot")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(run_group, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                                 itertools.repeat(args.v67_root), itertools.repeat(args.v68_root), range(args.start, args.end)))
        output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed_groups=len(done), formal_rollouts=len(done) * 7)) + "\n")


if __name__ == "__main__":
    main()
