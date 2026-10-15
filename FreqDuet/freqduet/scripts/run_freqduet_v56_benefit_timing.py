#!/usr/bin/env python3
"""Locate the timing of frozen V38 four-arm effects in the original 12 contexts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
import pandas as pd

from audit_freqduet_v38_dominant_case import select_branches
from audit_freqduet_v38_mechanism import ARMS, cumulative_at
from complete_freqduet_v34_optimized_evaluation import OPTIMIZED_ROOT
import run_freqduet_v37_joint_lower as joint
from run_freqduet_v38_observed_replay import replay


EXPOSURES = ("waiting_person_seconds", "onboard_person_seconds",
             "dispatch_backlog_trip_seconds")
TRACE_FILES = ("snapshot.json", "exposure.npz", "holds.csv", "terminals.csv", "launches.csv")
DOMINANT = (933013, 45007, 7200)


def context_name(seed, scenario, target):
    return f"seed{seed}_eval{scenario}_t{target}"


def unique_branches(branches):
    unique, aliases = {}, {}
    for arm in ARMS:
        row = branches[arm]
        pair = (int(row.first_action_s), int(row.second_action_s))
        if pair not in unique:
            unique[pair] = arm
        aliases[arm] = unique[pair]
    return aliases


def validate_snapshot(snapshot, item):
    if not snapshot["matches_v37"]:
        raise ValueError("trace did not reproduce V37")
    for field in ("seed", "scenario", "target_time_s", "first_action_s", "second_action_s"):
        if snapshot[field] != item[field]:
            raise ValueError(f"wrong trace {field}: {snapshot[field]} != {item[field]}")
    joint.assert_same_state({m: snapshot["episode"][m] for m in joint.OUTCOMES},
                            item["expected"], "V56_matches_V37")
    for decision in ("first", "second"):
        joint.assert_same_state(snapshot["decisions"][decision]["identity"],
                                item["identities"][decision], decision)


def prepare(root, v37, v38):
    contexts, items = [], []
    for seed, scenario, target in itertools.product(joint.SEEDS, joint.SCENARIOS, joint.TARGET_TIMES):
        name = context_name(seed, scenario, target)
        bank = pd.concat([pd.read_csv(v37 / "records" / f"{joint.shard_name(seed, scenario, target, a)}.csv")
                          for a in joint.ACTIONS], ignore_index=True)
        pairs = list(zip(bank.first_action_s, bank.second_action_s))
        if len(pairs) != 49 or set(pairs) != set(itertools.product(joint.ACTIONS, repeat=2)):
            raise ValueError(f"{name}: original seven-by-seven action bank is incomplete")
        branches = select_branches(bank)
        aliases = unique_branches(branches)
        context = dict(name=name, seed=seed, scenario=scenario, target_time_s=target, arms={})
        for arm in ARMS:
            label = f"{name}_{aliases[arm]}"
            context["arms"][arm] = label
            if aliases[arm] != arm:
                continue
            row = branches[arm]
            saved = json.loads(row.to_json())
            cached = v38 / "observed" / arm
            item = dict(label=label, seed=seed, scenario=scenario, target_time_s=target,
                        first_action_s=int(row.first_action_s), second_action_s=int(row.second_action_s),
                        expected={m: float(row[m]) for m in joint.OUTCOMES},
                        identities={d: {f: saved[f"{d}_{f}"]
                                        for f in ("time_s", "bus_id", "trip_id", "station_id", "direction")}
                                    for d in ("first", "second")}, reused_trace=None)
            if (seed, scenario, target) == DOMINANT and all((cached / f).is_file() for f in TRACE_FILES):
                validate_snapshot(json.loads((cached / "snapshot.json").read_text()), item)
                item["reused_trace"] = str(cached)
            items.append(item)
        contexts.append(context)
    plan = dict(contexts=contexts, items=items, selection="unchanged V38 select_branches on V37 bank")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("frozen V56 action plan changed")
    path.write_text(json.dumps(plan, indent=2) + "\n")
    n_reuse = sum(i["reused_trace"] is not None for i in items)
    status = dict(contexts=len(contexts), unique_arms=len(items), reused_traces=n_reuse,
                  missing_replays=len(items) - n_reuse, max_full_day_replays=48,
                  training_runs=0, main_policy_changed=False,
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    export = root / "prepare_export"
    export.mkdir(exist_ok=True)
    (export / "done.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


def load_plan(root):
    return json.loads((root / "plan.json").read_text())


def trace_path(root, item):
    return Path(item["reused_trace"]) if item["reused_trace"] else root / "observed" / item["label"]


def load_trace(root, item):
    cell = trace_path(root, item)
    snapshot = json.loads((cell / "snapshot.json").read_text())
    validate_snapshot(snapshot, item)
    with np.load(cell / "exposure.npz") as archive:
        timeline = archive["timeline"]
    holds = pd.read_csv(cell / "holds.csv")
    if float(holds.action_s.sum()) != snapshot["episode"]["commanded_holding_vehicle_seconds"]:
        raise ValueError("holding event sum differs from episode commands")
    return dict(snapshot=snapshot, timeline=timeline, holds=holds,
                terminals=pd.read_csv(cell / "terminals.csv"),
                launches=pd.read_csv(cell / "launches.csv"))


def run_item(root, v37, item):
    marker = root / "ready" / f"{item['label']}.json"
    if marker.exists():
        return item["label"]
    try:
        if not item["reused_trace"]:
            replay(root, v37, item["seed"], item["scenario"], item["target_time_s"],
                   item["label"], item["first_action_s"], item["second_action_s"], True)
        load_trace(root, item)
    except Exception as error:
        failures = root / "failures"
        failures.mkdir(exist_ok=True)
        with (failures / f"{item['label']}.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(error=repr(error), source_commit=os.environ.get(
                "FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
        raise
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps(dict(label=item["label"], reused=bool(item["reused_trace"]),
                                      matches_v37=True)) + "\n")
    print(f"V56_ARM_DONE {item['label']} reused={bool(item['reused_trace'])}", flush=True)
    return item["label"]


def batch(root, v37, start, end, workers):
    pending = [i for i in load_plan(root)["items"] if not i["reused_trace"]]
    if not 0 <= start < end <= len(pending) or workers < 1:
        raise ValueError("batch range is outside frozen missing-arm plan")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        done = list(pool.map(run_item, itertools.repeat(root), itertools.repeat(v37), pending[start:end]))
    export = root / "exports" / f"batch_{start}_{end}"
    export.mkdir(parents=True, exist_ok=True)
    (export / "done.json").write_text(json.dumps(dict(start=start, end=end, completed=done)) + "\n")


def evaluation_horizon(root):
    _, _, load_config, _ = joint.optimized_runtime(root)
    from env.sim import env_bus

    config = load_config(str(joint.MODEL_ROOT / "configs_freqduet" / f"{joint.CURRENT}.yaml"))
    settings = config["env"]
    env = env_bus(str(OPTIMIZED_ROOT / settings["path"]),
                  env_config=settings, route_sigma=settings["route_sigma"])
    env.reset()
    return float(env.protocol.evaluation_end_time_s)


def exposure_at(data, time):
    timeline = data["timeline"]
    end = float(timeline[-1, 0])
    if time <= end:
        return cumulative_at(timeline, time)
    episode = data["snapshot"]["episode"]
    if (episode["done_reason"] != "service_cleared" or episode["trips_unlaunched"] != 0
            or episode["trips_incomplete"] != 0):
        raise ValueError("trace is truncated rather than service-cleared")
    # Demand has ended and no vehicle or unlaunched trip remains. Any unserved
    # passengers keep waiting to the fixed censor clock; all other rates are zero.
    return timeline[-1, 1:] + (time - end) * np.array([episode["passengers_unserved"], 0, 0])


def common_cuts(reference, horizon):
    decisions = reference["snapshot"]["decisions"]
    first = float(decisions["first"]["identity"]["time_s"])
    cuts = [("before_first", first), ("second_decision", float(decisions["second"]["identity"]["time_s"])),
            ("15min", first + 900), ("30min", first + 1800)]
    returns = []
    for decision in ("first", "second"):
        trip = decisions[decision]["identity"]["trip_id"]
        rows = reference["terminals"].loc[reference["terminals"].trip_id == trip]
        if len(rows) != 1:
            raise ValueError(f"reference intervention trip {trip} lacks a unique terminal return")
        time = float(rows.iloc[0].time_s)
        returns.append(time)
        cuts.append((f"{decision}_reference_return", time))
    cuts.extend([("both_reference_returns", max(returns)), ("day_end", horizon)])
    if any(time > horizon for _, time in cuts):
        raise ValueError("requested timing cut exceeds the common evaluation horizon")
    return cuts


def timing_rows(context, traces, horizon):
    reference = traces["reference"]
    cuts = common_cuts(reference, horizon)
    before = exposure_at(reference, cuts[0][1])
    records = []
    for arm, data in traces.items():
        if data["timeline"][-1, 0] > horizon:
            raise ValueError("trace extends beyond fixed evaluation horizon")
        joint.assert_same_state(data["snapshot"]["decisions"]["first"]["identity"],
                                reference["snapshot"]["decisions"]["first"]["identity"])
        if not np.array_equal(exposure_at(data, cuts[0][1]), before):
            raise ValueError("exposure changed before the first intervention")
        for cut, time in cuts:
            exposure = exposure_at(data, time)
            ref_exposure = exposure_at(reference, time)
            row = dict(context=context["name"], train_seed=context["seed"], eval_seed=context["scenario"],
                       target_time_s=context["target_time_s"], arm=arm, cut=cut, time_s=time,
                       elapsed_s=time - cuts[0][1])
            for i, metric in enumerate(EXPOSURES):
                row[metric] = float(exposure[i] - before[i])
                row[f"delta_{metric}"] = float(exposure[i] - ref_exposure[i])
            row["delta_passenger_exposure_s"] = row["delta_waiting_person_seconds"] + row["delta_onboard_person_seconds"]
            row["delta_commanded_holding_s"] = float(
                data["holds"].loc[data["holds"].time_s < time].action_s.sum()
                - reference["holds"].loc[reference["holds"].time_s < time].action_s.sum())
            row["delta_readiness_delay_s_sum"] = float(
                data["launches"].loc[data["launches"].time_s <= time].readiness_s.sum()
                - reference["launches"].loc[reference["launches"].time_s <= time].readiness_s.sum())
            row["delta_launched_trips"] = int((data["launches"].time_s <= time).sum()
                                              - (reference["launches"].time_s <= time).sum())
            records.append(row)
    return records


def reversal_rows(timing, endpoints):
    rows = []
    for context, group in timing.groupby("context", sort=True):
        for arm in ARMS[1:]:
            panel = group.loc[group.arm == arm].set_index("cut")
            endpoint = endpoints.loc[(endpoints.context == context) & (endpoints.arm == arm)].iloc[0]
            full = float(panel.loc["day_end", "delta_passenger_exposure_s"])
            journey = float(endpoint.delta_restricted_total_journey_horizon_min)
            for cut in ("15min", "30min", "both_reference_returns"):
                short = float(panel.loc[cut, "delta_passenger_exposure_s"])
                rows.append(dict(context=context, arm=arm, cut=cut, short_delta_person_s=short,
                                 full_day_delta_person_s=full, tail_delta_person_s=full - short,
                                 full_day_journey_delta_min=journey,
                                 exposure_sign_reversal=bool(short * full < 0),
                                 short_exposure_vs_journey_sign_reversal=bool(short * journey < 0),
                                 exposure_vs_journey_sign_disagreement=bool(full * journey < 0)))
    return pd.DataFrame(rows)


def aggregate(root):
    plan = load_plan(root)
    horizon = evaluation_horizon(root)
    traces = {i["label"]: load_trace(root, i) for i in plan["items"]}
    timing, endpoints = [], []
    for context in plan["contexts"]:
        arms = {arm: traces[label] for arm, label in context["arms"].items()}
        timing.extend(timing_rows(context, arms, horizon))
        ref = arms["reference"]["snapshot"]["episode"]
        for arm, data in arms.items():
            snapshot = data["snapshot"]
            row = dict(context=context["name"], arm=arm,
                       first_action_s=snapshot["first_action_s"], second_action_s=snapshot["second_action_s"])
            for decision in ("first", "second"):
                row.update({f"{decision}_{f}": v for f, v in snapshot["decisions"][decision]["identity"].items()})
                trip = arms["reference"]["snapshot"]["decisions"][decision]["identity"]["trip_id"]
                returns = data["terminals"].set_index("trip_id", verify_integrity=True)
                row[f"{decision}_reference_trip_return_s"] = float(returns.loc[trip, "time_s"])
            for m in joint.OUTCOMES:
                row[m] = data["snapshot"]["episode"][m]
                row[f"delta_{m}"] = row[m] - ref[m]
            endpoints.append(row)
    timing, endpoints = pd.DataFrame(timing), pd.DataFrame(endpoints)
    reversals = reversal_rows(timing, endpoints)
    delta_metrics = [m for m in timing if m.startswith("delta_")]
    interaction = timing.pivot(index=["context", "cut", "time_s", "elapsed_s"], columns="arm", values=delta_metrics)
    interaction = (interaction.xs("joint", axis=1, level=1) - interaction.xs("first_only", axis=1, level=1)
                   - interaction.xs("second_only", axis=1, level=1)).reset_index()
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    for name, frame in dict(timing=timing, endpoints=endpoints, reversals=reversals, interaction=interaction).items():
        frame.to_csv(summary / f"{name}.csv", index=False)
    reversals.groupby(["arm", "cut"])[["exposure_sign_reversal", "short_exposure_vs_journey_sign_reversal",
                                        "exposure_vs_journey_sign_disagreement"]].sum().to_csv(summary / "sign_counts.csv")
    failures = sum(sum(1 for _ in p.open()) for p in (root / "failures").glob("*.jsonl"))
    status = dict(status="complete_development_timing_diagnosis", contexts=len(plan["contexts"]),
                  unique_arms=len(plan["items"]), reused_traces=sum(bool(i["reused_trace"]) for i in plan["items"]),
                  failed_arm_attempts=failures, all_endpoints_match_v37=True,
                  exposure_units="tick-integrated person/trip seconds, not restricted journey estimator",
                  evaluation_horizon_s=horizon,
                  selection=plan["selection"], training_runs=0, main_policy_changed=False,
                  independent_confirmation=False, diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(reversals.groupby(["arm", "cut"]).exposure_sign_reversal.sum().to_string(), flush=True)


def smoke(root, v37):
    plan = load_plan(root)
    for item in plan["items"]:
        if item["reused_trace"]:
            run_item(root, v37, item)
    dominant = next(c for c in plan["contexts"] if (c["seed"], c["scenario"], c["target_time_s"]) == DOMINANT)
    arms = {a: load_trace(root, next(i for i in plan["items"] if i["label"] == label))
            for a, label in dominant["arms"].items()}
    horizon = evaluation_horizon(root)
    dominant_timing = timing_rows(dominant, arms, horizon)
    item = next(i for i in plan["items"] if not i["reused_trace"] and not i["label"].endswith("_reference"))
    run_item(root, v37, item)
    export = root / "smoke_export"
    export.mkdir(exist_ok=True)
    pd.DataFrame(dominant_timing).to_csv(export / "dominant_timing.csv", index=False)
    (export / "done.json").write_text(json.dumps(dict(replayed=item["label"], matches_v37=True,
                                                      reused_dominant_accounting_pass=True,
                                                      evaluation_horizon_s=horizon)) + "\n")
    print("V56_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v37-root", type=Path)
    parser.add_argument("--v38-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out_root, args.v37_root, args.v38_root)
    elif args.command == "smoke":
        smoke(args.out_root, args.v37_root)
    elif args.command == "batch":
        batch(args.out_root, args.v37_root, args.start, args.end, args.workers)
    else:
        aggregate(args.out_root)
