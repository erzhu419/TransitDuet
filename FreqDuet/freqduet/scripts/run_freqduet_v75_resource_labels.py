#!/usr/bin/env python3
"""Expanded independent-day seven-action labels with frozen V59 controllers."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import torch

import freqduet_resource_observation as resources
import run_freqduet_v73_fleet_state as previous


old, source = previous.old, previous.source
labels, joint, event = source.labels, source.joint, source.event
TRAIN_DAYS = tuple(range(1750001, 1750033))
VALID_DAYS = tuple(range(1751001, 1751013))
DAYS = (*TRAIN_DAYS, *VALID_DAYS)
REFERENCES = list(itertools.product(range(len(labels.deployment.CASES)), range(len(DAYS))))
CONTEXTS = list(itertools.product(range(len(REFERENCES)), range(len(source.TARGETS))))
ITEMS = list(itertools.product(range(len(CONTEXTS)), source.FUTURES))


def reference_spec(index):
    case, day = REFERENCES[index]
    seed, actor_scenario = labels.deployment.CASES[case]
    return dict(reference_index=index, case_index=case, train_seed=seed, actor_scenario=actor_scenario,
        scenario_seed=DAYS[day], sampler_seed=17500000 + case * 1000 + day,
        role="training" if DAYS[day] in TRAIN_DAYS else "validation",
        label=f"{labels.deployment.case_name(seed, actor_scenario)}_demand{DAYS[day]}")


def context_spec(index):
    reference, window = CONTEXTS[index]
    return dict(reference_spec(reference), context_index=index, target_time_s=source.TARGETS[window])


def future_seed(context, future):
    return 175000000 + context * 100 + future


def reference_path(root, spec):
    return root / "references" / (spec["label"] + ".pt")


def collect_reference(root, v59_root, spec):
    path = reference_path(root, spec)
    if path.exists():
        return spec["label"]
    observer = previous.FleetObserver()
    result = labels.run_spec(root, v59_root, spec, source.TARGETS, spec["label"] + "_native", observer=observer)
    for target in source.TARGETS:
        resources.encode_resources(result["captures"][target], observer.snapshots[target])
        previous.telemetry.encode_fleet(observer.snapshots[target])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(spec=spec, fleet=observer.snapshots, **result), path)
    print("V75_REFERENCE_DONE " + json.dumps(dict(label=spec["label"], role=spec["role"])), flush=True)
    return spec["label"]


def run_actions(root, v59_root, spec, original, seed, label, check_noop=False):
    target = spec["target_time_s"]

    def replay(action, suffix):
        uniforms = event.EventUniforms(seed + 10000000)

        def continuation(runner, policy):
            startup = source.repeated.fork_future(runner, policy, seed)
            uniforms.install(runner, policy)
            return dict(startup, event_uniform_seed=uniforms.seed)

        result = labels.run_spec(root, v59_root, spec, (target,), label + "_" + suffix, action, continuation)
        labels.same_prefix(original, result["captures"][target])
        if result["summary"]["scenario_seed"] != seed:
            raise AssertionError("V75 continuation changed its registered future tape")
        print("V75_ARM_DONE " + json.dumps(dict(label=label, arm=suffix)), flush=True)
        return result, uniforms.records

    native, native_records = replay(None, "native")
    native_action = original["actor_action_s"]
    rows = []
    for action in joint.ACTIONS:
        result, records = (native, native_records) if action == native_action else replay(action, f"a{action}")
        joint.assert_same_state(native["captures"][target]["continuation"], result["captures"][target]["continuation"], "common_future_start")
        rows.append(dict(action_s=action, **{k: result["summary"][k] for k in labels.OUTCOMES},
            conservation_max_error=result["summary"]["stream_audit"]["conservation_max_error"],
            alignment=None if action == native_action else event.alignment(native_records, records)))
    if check_noop:
        forced, records = replay(native_action, "forced_native")
        for key in ("summary", "holds", "upper_path"):
            joint.assert_same_state(native[key], forced[key], "forced_native_" + key)
        joint.assert_same_state(native_records, records, "forced_native_event_uniforms")
    return dict(native_action_s=native_action, rows=rows, exact_prefix=True, common_future_start=True,
                forced_native_verified=check_noop, future_seed=seed)


def group_path(root, context, future):
    return root / "groups" / f"c{context}_f{future}.json"


def run_group(root, v59_root, index):
    context, future = ITEMS[index]
    path = group_path(root, context, future)
    if path.exists():
        return index
    spec = context_spec(context)
    base = torch.load(reference_path(root, spec), map_location="cpu", weights_only=False)
    result = run_actions(root, v59_root, spec, base["captures"][spec["target_time_s"]],
                         future_seed(context, future), f"c{context}_f{future}")
    previous.write_json(path, dict(**spec, future_index=future, **result,
                                   diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")))
    print("V75_GROUP_DONE " + json.dumps(dict(context_index=context, future_index=future)), flush=True)
    return index


def smoke_group(root, v59_root, spec, window):
    base = torch.load(reference_path(root, spec), map_location="cpu", weights_only=False)
    target = source.TARGETS[window]
    result = run_actions(root, v59_root, dict(spec, target_time_s=target), base["captures"][target],
                         175900000 + spec["case_index"] * 1000 + window, f"smoke_c{spec['case_index']}_w{window}", True)
    previous.write_json(root / "smoke_details" / f"c{spec['case_index']}_w{window}.json", result)
    return window


def smoke(root, v59_root, v70_root, v72_root, v73_root, workers):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v75_*.py", "-v"], check=True)
    base = torch.load(source.reference_path(v70_root, 0), map_location="cpu", weights_only=False)
    legacy = run_actions(root, v59_root, dict(source.reference_spec(0), target_time_s=source.TARGETS[0]),
                         base["captures"][source.TARGETS[0]], source.future_seed(0, 0), "legacy_v70", True)
    cached = json.loads(source.group_path(v70_root, 0).read_text())
    joint.assert_same_state(cached["rows"], legacy["rows"], "V70_seven_action_and_event_alignment_identity")
    specs = []
    for case, day in ((0, 1759901), (3, 1759902)):
        seed, actor_scenario = labels.deployment.CASES[case]
        specs.append(dict(case_index=case, train_seed=seed, actor_scenario=actor_scenario, scenario_seed=day,
            sampler_seed=17599900 + case, role="smoke", label=f"smoke_case{case}_demand{day}"))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(collect_reference, itertools.repeat(root), itertools.repeat(v59_root), specs))
        list(pool.map(smoke_group, itertools.repeat(root), itertools.repeat(v59_root),
                      (specs[0], specs[1], specs[1]), (0, 1, 2)))
    subprocess.run([sys.executable, str(Path(__file__).with_name("run_freqduet_v75_resource_value.py")),
        "smoke", "--out-root", str(root), "--v70-root", str(v70_root),
        "--v72-root", str(v72_root), "--v73-root", str(v73_root)], check=True)
    previous.write_json(root / "smoke_export" / "done.json", dict(status="legacy_identity_and_fresh_smoke_passed",
        legacy_actions=7, forced_native_groups=4, smoke_reference_days=2, smoke_rollouts=34,
        structured_features=len(resources.FEATURES), formal_holdout_consumed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "references", "batch"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path, required=True)
    parser.add_argument("--v70-root", type=Path)
    parser.add_argument("--v72-root", type=Path)
    parser.add_argument("--v73-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v59_root, args.v70_root, args.v72_root, args.v73_root, args.workers)
    elif args.command == "references":
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            list(pool.map(collect_reference, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                          (reference_spec(i) for i in range(len(REFERENCES)))))
        previous.write_json(args.out_root / "reference_exports" / "done.json", dict(status="references_complete", references=len(REFERENCES)))
    else:
        if args.end is None or args.shard is None or not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("batch needs a registered shard and a valid contiguous group range")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            list(pool.map(run_group, itertools.repeat(args.out_root), itertools.repeat(args.v59_root), range(args.start, args.end)))
        previous.write_json(args.out_root / "shard_exports" / str(args.shard) / "done.json",
                            dict(status="action_labels_complete", start=args.start, end=args.end))


if __name__ == "__main__":
    main()
