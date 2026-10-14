#!/usr/bin/env python3
"""Matched two-decision lower-action pilot under the unchanged V9 deployment."""

import argparse
import copy
from dataclasses import asdict
import gc
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from audit_freqduet_v36_actor_alignment import summarize
from audit_freqduet_v36_joint_space import PRIMARY, SYSTEM_OUTCOMES, joint_space
from run_freqduet_v36_lower_credit import (
    BASE_EP, CURRENT, MODEL_ROOT, OUTCOMES, TARGET_TIMES, TRAIN_ROOT,
    assert_same_state, case_name, check_unchanged, optimized_runtime,
    rng_state, runtime_config, same_prefix, seed_runtime,
)


SEEDS = (12011, 933013)
SCENARIOS = (45007, 934011)
ACTIONS = (0, 5, 10, 15, 20, 30, 45)
CONTEXT_KEYS = ["train_seed", "eval_seed", "target_time_s"]


def shard_name(seed, scenario, target, first_action):
    return f"seed{seed}_eval{scenario}_t{target}_a{first_action}"


def second_matches(first, event):
    return (event["station_id"] == first["station_id"]
            and event["direction"] == first["direction"]
            and event["bus_id"] != first["bus_id"])


def run_branch(root, seed, scenario, target, label, first_action=None, second_action=None):
    _, Runner, load_config, torch = optimized_runtime(root)
    cell = root / "cells" / f"seed{seed}_eval{scenario}_t{target}" / label
    seed_runtime(torch, seed)
    config = load_config(str(MODEL_ROOT / "configs_freqduet" / f"{CURRENT}.yaml"))
    runner = Runner(runtime_config(config, seed, cell / "logs", 40), device="cpu")
    checkpoints = TRAIN_ROOT / "logs" / f"{CURRENT}_seed{seed}/checkpoints"
    runner.load_checkpoint(checkpoints, ep=BASE_EP, require_deployment_state=True)
    captures, lower_prefix, upper_prefix = {}, [], []
    original_lower = runner._lower_action_for_agent
    original_upper = runner._upper_callback_v2

    def upper_trace(state, trip):
        result = original_upper(state, trip)
        if len(captures) < 2:
            upper_prefix.append(dict(
                time_s=float(runner.env.current_time), trip_id=int(trip.launch_turn),
                state=np.asarray(state).copy(), target_headway_s=float(result),
                decision=copy.deepcopy(runner._ep_trip_records[-1])))
        return result

    def lower_trace(obs, key, last_action=0.0, deterministic=False):
        action = original_lower(obs, key, last_action=last_action, deterministic=deterministic)
        if len(captures) == 2:
            return action
        bus = runner._bus_for_agent(key)
        raw = np.asarray(obs, dtype=np.float32).copy()
        identity = dict(time_s=float(runner.env.current_time), bus_id=int(key),
                        trip_id=int(bus.trip_id), station_id=int(raw[1]), direction=bool(bus.direction))
        event = dict(identity, raw_state=raw, previous_action_s=float(last_action),
                     actor_action_s=float(np.asarray(action).item()))
        lower_prefix.append(event)
        departure = runner._pending_lower_action_context[int(key)]
        eligible = (not runner._lower_terminal_action_masked(bus)
                    and not runner._fixed_expert_active and departure.evidence_valid)
        decision = None
        if eligible and not captures and identity["time_s"] >= target:
            decision, forced = "first", first_action
        elif eligible and "first" in captures and second_matches(captures["first"]["identity"], identity):
            decision, forced = "second", second_action
        if decision is None:
            return action
        captures[decision] = dict(
            identity=identity, raw_state=raw,
            encoded_state=runner._augment_lower_state(obs, last_action).copy(),
            previous_action_s=float(last_action), actor_action_s=event["actor_action_s"],
            departure_context=asdict(departure), lower_prefix=copy.deepcopy(lower_prefix),
            upper_prefix=copy.deepcopy(upper_prefix), rng=rng_state(runner, torch))
        # A later decision's history includes the actual first intervention.
        if forced is not None:
            event["executed_action_s"] = float(forced)
            return np.asarray([forced], dtype=np.float32)
        event["executed_action_s"] = event["actor_action_s"]
        return action

    runner._lower_action_for_agent = lower_trace
    runner._upper_callback_v2 = upper_trace
    runner.evaluate([scenario], output_dir=cell / "evaluation", policy_ep=BASE_EP)
    if set(captures) != {"first", "second"}:
        raise ValueError("two eligible same-stop/direction distinct-vehicle decisions were not observed")
    if any(not value["upper_prefix"] for value in captures.values()):
        raise ValueError("upper dispatch prefix was not observed")
    episode = pd.read_csv(cell / "evaluation/evaluation.csv").iloc[0].to_dict()
    del runner
    gc.collect()
    print(f"JOINT_BRANCH_DONE seed{seed} eval{scenario} t{target} {label}", flush=True)
    return dict(episode=episode, captures=captures)


def run_shard(root, v36_root, seed, scenario, target, first_action, smoke=False):
    name = shard_name(seed, scenario, target, first_action)
    ready = root / "ready" / name / "ready.json"
    if ready.is_file():
        return
    common = dict(root=root, seed=seed, scenario=scenario, target=target)
    reference = run_branch(**common, label=f"a{first_action}_reference")
    archive = TRAIN_ROOT.parent / "protocol_v6_v35_actor_cross_b200/cells" / (
        f"{CURRENT}_seed{seed}_U200_L200/evaluation/evaluation.csv")
    old = pd.read_csv(archive).set_index("eval_seed").loc[scenario].to_dict()
    old["eval_seed"] = scenario
    check_unchanged(reference["episode"], old)
    first_only = run_branch(**common, label=f"a{first_action}_first_only", first_action=first_action)
    same_prefix(reference["captures"]["first"], first_only["captures"]["first"])
    original = pd.read_csv(v36_root / "records" / case_name(CURRENT, seed, scenario, BASE_EP) / "rollouts.csv")
    original = original.loc[(original.target_time_s == target) & (original.action_s == first_action)]
    if len(original) != 1:
        raise ValueError("V36 first-only branch missing or duplicated")
    assert_same_state({metric: first_only["episode"][metric] for metric in OUTCOMES},
                      original.iloc[0][list(OUTCOMES)].to_dict(), "V36_first_only")
    first_capture = first_only["captures"]["first"]
    for field in ("time_s", "bus_id", "trip_id", "station_id", "direction"):
        assert_same_state(first_capture["identity"][field], original.iloc[0][field], field)
    if first_action == reference["captures"]["first"]["actor_action_s"]:
        check_unchanged(reference["episode"], first_only["episode"])
    second_actor = first_only["captures"]["second"]["actor_action_s"]
    actions = sorted({0, 45, int(second_actor)}) if smoke else ACTIONS
    rows = []
    for second_action in actions:
        branch = run_branch(**common, label=f"a{first_action}_b{second_action}",
                            first_action=first_action, second_action=second_action)
        same_prefix(reference["captures"]["first"], branch["captures"]["first"])
        same_prefix(first_only["captures"]["second"], branch["captures"]["second"])
        if second_action == second_actor:
            check_unchanged(first_only["episode"], branch["episode"])
        row = dict(train_seed=seed, eval_seed=scenario, target_time_s=target,
                   first_action_s=first_action, second_action_s=second_action,
                   first_actor_s=reference["captures"]["first"]["actor_action_s"],
                   second_actor_s=second_actor)
        for metric in OUTCOMES:
            row[metric] = branch["episode"][metric]
            row[f"first_only_{metric}"] = first_only["episode"][metric]
            row[f"reference_{metric}"] = reference["episode"][metric]
        for decision in ("first", "second"):
            row.update({f"{decision}_{field}": value
                        for field, value in branch["captures"][decision]["identity"].items()})
        rows.append(row)
    record = root / "records" / f"{name}.csv"
    record.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(record, index=False)
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text(json.dumps(dict(train_seed=seed, eval_seed=scenario, target_time_s=target,
                                    first_action_s=first_action, second_actions_s=list(actions),
                                    smoke=smoke, rollouts=2 + len(actions), exact_prefixes=True,
                                    first_only_matches_v36=True, noop_matches=True,
                                    diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))) + "\n")
    print(f"JOINT_SHARD_DONE {name} rollouts={2 + len(actions)}", flush=True)


def context_spaces(group):
    for metric in OUTCOMES:
        if group[f"reference_{metric}"].nunique() != 1:
            raise ValueError("reference differs across first-action shards")
    if group.first_actor_s.nunique() != 1:
        raise ValueError("first reference actor differs across shards")
    baseline = pd.Series({metric: group[f"reference_{metric}"].iloc[0] for metric in SYSTEM_OUTCOMES})
    first = group.drop_duplicates("first_action_s")[[f"first_only_{m}" for m in SYSTEM_OUTCOMES]].copy()
    first.columns = list(SYSTEM_OUTCOMES)
    second = group.loc[group.first_action_s == group.first_actor_s.iloc[0], list(SYSTEM_OUTCOMES)]
    singles = pd.concat([first, second], ignore_index=True)
    single_result, best_single = joint_space(singles, baseline)
    first_result, _ = joint_space(first, baseline)
    joint_result, best_joint = joint_space(group, baseline)
    incremental_result, _ = joint_space(group, best_single)
    results = dict(first_only_vs_actor=first_result, single_union_vs_actor=single_result,
                   joint_vs_actor=joint_result, joint_vs_best_single=incremental_result)
    frontier_delta = float(best_joint[PRIMARY[0]] - best_single[PRIMARY[0]])
    return [dict(contrast=contrast, joint_minus_single_frontier_journey_min=frontier_delta, **result)
            for contrast, result in results.items()]


def aggregate(root):
    expected = {shard_name(*item) for item in itertools.product(SEEDS, SCENARIOS, TARGET_TIMES, ACTIONS)}
    ready = list((root / "ready").glob("*/ready.json"))
    if {path.parent.name for path in ready} != expected:
        raise ValueError("joint-action pilot is incomplete or outside its frozen cohort")
    markers = [json.loads(path.read_text()) for path in ready]
    if any(marker["smoke"] for marker in markers):
        raise ValueError("smoke cannot replace a formal shard")
    pairs = pd.concat([pd.read_csv(root / "records" / f"{name}.csv")
                       for name in sorted(expected)], ignore_index=True)
    if pairs.duplicated([*CONTEXT_KEYS, "first_action_s", "second_action_s"]).any():
        raise ValueError("duplicate pair record")
    rows = []
    for key, group in pairs.groupby(CONTEXT_KEYS):
        if set(zip(group.first_action_s, group.second_action_s)) != set(itertools.product(ACTIONS, repeat=2)):
            raise ValueError("context lacks the full 49-action grid")
        for result in context_spaces(group.sort_values(["first_action_s", "second_action_s"])):
            rows.append(dict(zip(CONTEXT_KEYS, key), **result))
    contexts = pd.DataFrame(rows)
    summary = root / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(summary / "context_spaces.csv", index=False)
    sys.path.insert(0, str(MODEL_ROOT))
    from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval
    metrics = [name for name in contexts if name not in [*CONTEXT_KEYS, "contrast"]]
    summarize(contexts, ["contrast"], metrics, hierarchical_interval).to_csv(summary / "joint_space.csv", index=False)
    status = dict(status="complete_exploratory_two_decision_space", contexts=len(rows) // 4,
                  pairs=len(pairs), shards=len(expected), rollouts=sum(m["rollouts"] for m in markers),
                  training_runs=0, independent_confirmation=False, deployable_policy=False,
                  source_commits=sorted({m["diagnostic_commit"] for m in markers}),
                  exact_prefixes=True, first_only_matches_v36=True, noop_matches=True,
                  comparison="joint vs actor; joint vs best single at either decision; hindsight only")
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(contexts.groupby("contrast")[metrics].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "smoke", "aggregate"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v36-root", type=Path)
    parser.add_argument("--train-seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--eval-seed", type=int, choices=SCENARIOS, default=SCENARIOS[0])
    parser.add_argument("--target-time", type=int, choices=TARGET_TIMES, default=TARGET_TIMES[0])
    parser.add_argument("--first-action", type=int, choices=ACTIONS, default=0)
    args = parser.parse_args()
    if args.command == "aggregate":
        aggregate(args.out_root)
    else:
        if args.v36_root is None:
            parser.error("--v36-root is required for rollout controls")
        run_shard(args.out_root, args.v36_root, args.train_seed, args.eval_seed,
                  args.target_time, args.first_action, smoke=args.command == "smoke")
