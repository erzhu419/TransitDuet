#!/usr/bin/env python3
"""Repeat matched full-day holding grids with new post-prefix random streams."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import run_freqduet_v67_action_labels as labels


joint = labels.joint
FUTURES = tuple(range(8))
ITEMS = list(itertools.product(range(len(labels.CONTEXTS)), FUTURES))
RESPONSE_METRICS = ("passenger_person_s", "cv_squared", "readiness_delay_s",
                    "trips_unlaunched", "trips_incomplete", "holding_s_per_trip")
CONSTRAINT_METRICS = ("cv_squared", "readiness_delay_s", "trips_unlaunched", "trips_incomplete")


def future_seed(context, future):
    return 68000000 + context * 100 + future


def physical_prefix(env):
    return copy.deepcopy(dict(
        time_s=env.current_time,
        pending=[[(p.appear_time, p.destination_station.station_id)
                  for p in station._pending_passengers] for station in env.stations],
        route_speed=[route.speed_limit for route in env.routes],
        demand=env._demand_multipliers, od=env._od_multipliers, peak=env._peak_shift,
    ))


def fork_future(runner, policy, seed):
    env = runner.env
    before = physical_prefix(env)
    # Preserve sampled current-window arrivals and day-level latent profiles.
    # Only future calls to the tape receive new innovations.
    env.scenario_tape = type(env.scenario_tape)(seed=seed, version=env.scenario_tape.version)
    policy.set_sampling_state(torch.Generator().manual_seed(seed + 10000000).get_state())
    upper = runner.upper_trainer.policy_net
    if upper.sampling_state() is not None:
        upper.set_sampling_state(torch.Generator().manual_seed(seed + 20000000).get_state())
    random.seed(seed + 30000000)
    np.random.seed(seed + 40000000)
    torch.manual_seed(seed + 50000000)
    for index, rng in enumerate(runner._runtime_numpy_streams().values()):
        rng.seed(seed + 60000000 + index * 1000000)
    joint.assert_same_state(before, physical_prefix(env), "unchanged_physical_prefix")
    return dict(physical=before, rng=joint.rng_state(runner, torch),
                policy_sampling=policy.sampling_state(), upper_sampling=upper.sampling_state(),
                scenario_seed=env.scenario_tape.seed, scenario_version=env.scenario_tape.version,
                scenario_streams=copy.deepcopy(env.scenario_tape._streams))


def group_path(root, index):
    context, future = ITEMS[index]
    return root / "groups" / f"c{context}_f{future}.json"


def run_group(root, v59_root, v67_root, index, check_noop=False):
    path = group_path(root, index)
    if path.exists():
        return path.stem
    context, future = ITEMS[index]
    spec = labels.context_spec(context)
    seed = future_seed(context, future)
    original = torch.load(labels.reference_path(v67_root, spec["reference_index"]),
                          map_location="cpu", weights_only=False)["captures"][spec["target_time_s"]]

    def replay(action, suffix):
        result = labels.run_branch(root, v59_root, spec["reference_index"], (spec["target_time_s"],),
                                   f"c{context}_f{future}_{suffix}", action,
                                   continuation=lambda runner, policy: fork_future(runner, policy, seed))
        capture = result["captures"][spec["target_time_s"]]
        labels.same_prefix(original, capture)
        if result["summary"]["scenario_seed"] != seed:
            raise AssertionError("the continuation did not use the registered future seed")
        return result

    native = replay(None, "native")
    native_action = original["actor_action_s"]
    rows = []
    for action in joint.ACTIONS:
        result = native if action == native_action else replay(action, f"a{action}")
        joint.assert_same_state(native["captures"][spec["target_time_s"]]["continuation"],
                                result["captures"][spec["target_time_s"]]["continuation"], "common_future_start")
        rows.append(dict(action_s=action, **{k: result["summary"][k] for k in labels.OUTCOMES},
                         conservation_max_error=result["summary"]["stream_audit"]["conservation_max_error"]))
    if check_noop:
        forced = replay(native_action, "forced_native")
        for key in ("summary", "holds", "upper_path"):
            joint.assert_same_state(native[key], forced[key], "forced_native_" + key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(**spec, future_index=future, future_seed=seed,
                                    native_action_s=native_action, rows=rows, exact_prefix=True,
                                    common_future_start=True, forced_native_verified=check_noop,
                                    diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT")), indent=2) + "\n")
    print("V68_GROUP_DONE " + json.dumps(dict(context_index=context, future_index=future, rollouts=7)), flush=True)
    return path.stem


def constrained_action(means, native_index):
    feasible = np.ones(len(joint.ACTIONS), dtype=bool)
    for name in CONSTRAINT_METRICS:
        feasible &= means[name] <= means[name][native_index]
    costs = np.where(feasible, means["passenger_person_s"], np.inf)
    best = int(np.argmin(costs))
    # Equal passenger outcomes do not justify changing the native action.
    return native_index if costs[best] == costs[native_index] else best


def original_oracle(rows, native_index):
    eligible = [i for i, row in enumerate(rows)
                if all(row[k] <= rows[native_index][k] for k in CONSTRAINT_METRICS)]
    return min(eligible, key=lambda i: rows[i]["passenger_person_s"])


def evaluate_selection(arrays, native_index, training_futures, evaluation_futures):
    means = {name: values[training_futures].mean(axis=0) for name, values in arrays.items()}
    selected = constrained_action(means, native_index)
    return dict(action_s=joint.ACTIONS[selected], changed=selected != native_index,
                **{name: float((values[evaluation_futures, selected]
                                - values[evaluation_futures, native_index]).mean()) for name, values in arrays.items()})


def signal_noise(differences):
    variance = differences[:, 1:].var(axis=0, ddof=1)
    mean_squared = np.square(differences[:, 1:].mean(axis=0))
    left, right = differences[:4, 1:].mean(axis=0), differences[4:, 1:].mean(axis=0)
    informative = (left != 0) | (right != 0)
    return dict(single_future_noise_variance=float(variance.mean()),
                mean_effect_squared=float(mean_squared.mean()),
                debiased_signal_squared=float((mean_squared - variance / len(differences)).mean()),
                half_sign_agreement=float(((np.sign(left) == np.sign(right)) & informative).sum()
                                          / max(int(informative.sum()), 1)),
                half_sign_pairs=int(informative.sum()))


def frozen_models(v59_root, v66_root, case):
    calibration = json.loads(labels.shrink.calibration_path(v66_root, case).read_text())
    payload = torch.load(calibration["base_value_source"], map_location="cpu", weights_only=False)
    models = {}
    for name in labels.residual.CANDIDATES:
        weights = payload["models"][name]
        model = labels.residual.ResidualValue(name, weights["center"], weights["scale"], payload["hidden"])
        model.load_state_dict(weights)
        models[name] = model.eval()
    source = torch.load(v59_root / "cells" / labels.deployment.case_name(*labels.deployment.CASES[case])
                        / "learning_smoke.pt", map_location="cpu", weights_only=False)
    models["frozen"] = labels.warmup.load_cost_model(source["trainer"]["cost_q_net"]).eval()
    return models, payload, {k: v["alpha"] for k, v in calibration["coefficients"].items()}


def day_intervals(contexts, getter, seed):
    means = [float(np.mean([getter(c) for c in contexts if c["scenario_seed"] == day]))
             for day in labels.DEMAND_SEEDS]
    return labels.expanded.paired_interval(means, seed)


def aggregate(root, v59_root, v66_root, v67_root):
    joint.optimized_runtime(root)
    groups = [json.loads(group_path(root, index).read_text()) for index in range(len(ITEMS))]
    models = {case: frozen_models(v59_root, v66_root, case) for case in range(len(labels.deployment.CASES))}
    contexts, flat = [], []
    for context in range(len(labels.CONTEXTS)):
        spec = labels.context_spec(context)
        selected = [g for g in groups if g["context_index"] == context]
        if [g["future_index"] for g in selected] != list(FUTURES):
            raise ValueError("context is missing the predeclared eight independent futures")
        if any([r["action_s"] for r in g["rows"]] != list(joint.ACTIONS) for g in selected):
            raise ValueError("future is missing the complete seven-action grid")
        original = torch.load(labels.reference_path(v67_root, spec["reference_index"]),
                              map_location="cpu", weights_only=False)["captures"][spec["target_time_s"]]
        native_index = joint.ACTIONS.index(original["actor_action_s"])
        arrays = {name: np.asarray([[row[name] for row in g["rows"]] for g in selected], dtype=float)
                  for name in RESPONSE_METRICS}
        old = [json.loads((v67_root / "labels" / f"c{context}_a{a}.json").read_text()) for a in joint.ACTIONS]
        old_action = original_oracle(old, native_index)
        replication = dict(action_s=joint.ACTIONS[old_action], changed=old_action != native_index,
                           **{name: float((a[:, old_action] - a[:, native_index]).mean()) for name, a in arrays.items()})
        halves = [evaluate_selection(arrays, native_index, np.arange(4), np.arange(4, 8)),
                  evaluate_selection(arrays, native_index, np.arange(4, 8), np.arange(4))]
        cross_future = {name: float(np.mean([h[name] for h in halves])) for name in RESPONSE_METRICS}
        oracle = evaluate_selection(arrays, native_index, np.arange(8), np.arange(8))
        truth = arrays["cv_squared"].mean(axis=0) - arrays["cv_squared"][:, 0].mean()
        predictions = labels.action_predictions(original, *models[spec["case_index"]])
        metrics = {name: labels.predictor_metrics(value, truth) for name, value in predictions.items()}
        metrics["zero_delta"] = labels.predictor_metrics(np.zeros_like(truth), truth)
        context_row = dict(**spec, native_action_s=original["actor_action_s"], v67_replication=replication,
                           cross_future=cross_future, cross_future_halves=halves, mean_future_oracle=oracle,
                           predictor_metrics=metrics, signal_noise={name: signal_noise(a - a[:, :1])
                                                                  for name, a in arrays.items()})
        contexts.append(context_row)
        for g in selected:
            native = g["rows"][native_index]
            for row in g["rows"]:
                flat.append(dict(context_index=context, case_index=spec["case_index"], scenario_seed=spec["scenario_seed"],
                                 target_time_s=spec["target_time_s"], future_index=g["future_index"], future_seed=g["future_seed"],
                                 **row, **{f"delta_native_{name}": row[name] - native[name] for name in RESPONSE_METRICS}))
    summary = root / "summary"
    summary.mkdir(exist_ok=True)
    (root / "groups" / "context_audit.json").write_text(json.dumps(contexts, indent=2) + "\n")
    pd.DataFrame(flat).to_csv(root / "groups" / "repeated_action_effects.csv", index=False)
    effects = {mode: {metric: day_intervals(contexts, lambda c, m=mode, k=metric: c[m][k], 68000)
                     for metric in RESPONSE_METRICS} for mode in ("v67_replication", "cross_future", "mean_future_oracle")}
    contrasts = {name: {metric: day_intervals(contexts, lambda c, n=name, k=metric:
                                           c["predictor_metrics"][n][k] - c["predictor_metrics"]["zero_delta"][k], 68100)
                        for metric in ("delta_mae", "delta_rmse")} for name in labels.PREDICTORS}
    signal = {metric: {stat: day_intervals(contexts, lambda c, m=metric, s=stat: c["signal_noise"][m][s], 68200)
                       for stat in contexts[0]["signal_noise"][metric]} for metric in ("passenger_person_s", "cv_squared")}
    status = dict(status="repeated_matched_continuations_complete", contexts=len(contexts), independent_future_seeds=len(groups),
                  futures_per_context=len(FUTURES), formal_rollouts=len(groups) * len(joint.ACTIONS),
                  demand_day_clusters=len(labels.DEMAND_SEEDS), original_training_seeds=2,
                  exact_prefixes=all(g["exact_prefix"] for g in groups), common_future_start=all(g["common_future_start"] for g in groups),
                  conservation_max_error=max(r["conservation_max_error"] for g in groups for r in g["rows"]),
                  forced_native_smoke_groups=sum(g["forced_native_verified"] for g in groups),
                  v67_selected_changes=sum(c["v67_replication"]["changed"] for c in contexts),
                  mean_future_oracle_improving_contexts=sum(c["mean_future_oracle"]["passenger_person_s"] < 0 for c in contexts),
                  effects=effects, signal_noise=signal, paired_predictor_vs_zero=contrasts,
                  predictor_metrics={name: {metric: float(np.mean([c["predictor_metrics"][name][metric] for c in contexts]))
                                            for metric in ("delta_mae", "delta_rmse")}
                                     for name in (*labels.PREDICTORS, "zero_delta")},
                  actor_updates=0, dual_updates=0, value_updates=0, main_policy_changed=False,
                  conditional_on_fixed_day_profile=True, fresh_prefix_confirmation=False, control_admitted=False)
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print("V68_AGGREGATE " + json.dumps(status), flush=True)


def smoke(root, v59_root, v67_root, workers):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                    str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v6[0-8]_*.py", "-v"], check=True)
    spec = labels.context_spec(0)
    original = torch.load(labels.reference_path(v67_root, spec["reference_index"]), map_location="cpu", weights_only=False)
    unchanged = labels.run_branch(root, v59_root, spec["reference_index"], (spec["target_time_s"],), "no_fork_smoke")
    for key in ("summary", "holds", "upper_path"):
        joint.assert_same_state(original[key], unchanged[key], "unchanged_V67_" + key)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        list(pool.map(run_group, itertools.repeat(root), itertools.repeat(v59_root), itertools.repeat(v67_root), (0, 1), (True, True)))
    if not all(json.loads(group_path(root, i).read_text())["forced_native_verified"] for i in (0, 1)):
        raise AssertionError("smoke lacks its forced-native identity checks")
    output = root / "smoke_export"
    output.mkdir(exist_ok=True)
    (output / "done.json").write_text(json.dumps(dict(status="prefix_fork_and_native_identity_pass", rollouts=17, cached_formal_rollouts=14)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "batch", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v59-root", type=Path, required=True)
    parser.add_argument("--v67-root", type=Path, required=True)
    parser.add_argument("--v66-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root, args.v59_root, args.v67_root, args.workers)
    elif args.command == "aggregate":
        aggregate(args.out_root, args.v59_root, args.v66_root, args.v67_root)
    else:
        if not 0 <= args.start < args.end <= len(ITEMS):
            parser.error("batch lies outside the fixed context/future design")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            done = list(pool.map(run_group, itertools.repeat(args.out_root), itertools.repeat(args.v59_root),
                                 itertools.repeat(args.v67_root), range(args.start, args.end)))
        output = args.out_root / "exports" / f"batch_{args.start}_{args.end}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "done.json").write_text(json.dumps(dict(completed_groups=len(done), formal_rollouts=len(done) * 7)) + "\n")


if __name__ == "__main__":
    main()
