#!/usr/bin/env python3
"""Frozen three-policy test of an observable regularity-moment projection."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from freqduet_regularity_projection import RegularityProjection
import run_freqduet_v78_system_policy_step as pilot


DAYS = tuple(range(1810001, 1810033))
MODES = (("stochastic", 0), ("stochastic", 1), ("stochastic", 2), ("deterministic", 0))
ITEMS = list(itertools.product(pilot.SEEDS, DAYS, MODES))
CONTRASTS = dict(projected_vs_base=("projected", "base"), raw_vs_base=("raw", "base"),
                 projected_vs_raw=("projected", "raw"))


def run_item(root, v78_root, index):
    path = root / "pairs" / f"p{index}.json"
    if path.exists():
        return index
    seed, day, (mode, rep) = ITEMS[index]
    theta = json.loads((v78_root / "adapters" / f"seed{seed}.json").read_text())["theta"]
    sampler = 181000000 + seed + 100 * (day - DAYS[0]) + rep
    results, events = {}, {}
    for label, weights in (("base", [0.] * 4), ("raw", theta)):
        row, records, _, _ = pilot.rollout(root, seed, day, mode, sampler, weights, f"p{index}_{label}")
        results[label], events[label] = row, records
    rule = RegularityProjection()
    row, records, _, _ = pilot.rollout(root, seed, day, mode, sampler, theta, f"p{index}_projected",
                                      probability_rule=rule)
    results["projected"], events["projected"] = row, records
    if rule.max_projected_deficit > 1e-10:
        raise AssertionError("observable regularity moment constraint failed")
    pilot.write_json(path, dict(index=index, train_seed=seed, scenario_seed=day, mode=mode, rep=rep,
        results=results, projection=rule.summary(),
        alignment={name: pilot.event.alignment(events[left], events[right]) for name, (left, right) in
                   CONTRASTS.items()} if mode == "stochastic" else None,
        effects={name: {m: results[left][m] - results[right][m] for m in pilot.METRICS}
                 for name, (left, right) in CONTRASTS.items()}))
    print("V81_PROJECTION_CELL_DONE " + str(index), flush=True)
    return index


def intervals(rows):
    result = {}
    for contrast in CONTRASTS:
        result[contrast] = {}
        for metric in pilot.METRICS:
            values = [float(np.mean([r["effects"][contrast][metric] for r in rows
                                     if r["scenario_seed"] == day])) for day in DAYS]
            summary = pilot.event.labels.expanded.paired_interval(values, 81000)
            result[contrast][metric] = {k: summary[k] for k in ("n_demand_days", "mean", "ci95")}
    return result


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    effects = {mode: intervals([r for r in rows if r["mode"] == mode])
               for mode in ("stochastic", "deterministic")}
    gates = {}
    for mode, contrasts in effects.items():
        primary = contrasts["projected_vs_base"]
        gates[mode] = primary["passenger_person_s"]["ci95"][1] < 0 and all(
            primary[m]["ci95"][1] <= 0 for m in
            ("cv_squared", "readiness_delay_s", "trips_unlaunched", "trips_incomplete"))
    pilot.write_json(root / "summary" / "status.json", dict(status="observable_regularity_projection_pilot_complete",
        effects=effects, joint_gates=gates, primary_mode="stochastic", primary_contrast="projected_vs_base",
        by_seed={str(s): {mode: intervals([r for r in rows if r["train_seed"] == s and r["mode"] == mode])
            for mode in effects} for s in pilot.SEEDS},
        projection={mode: dict(decisions=sum(r["projection"]["decisions"] for r in rows if r["mode"] == mode),
            active=sum(r["projection"]["active"] for r in rows if r["mode"] == mode),
            max_projected_gain_deficit=max(r["projection"]["max_projected_gain_deficit"]
                                          for r in rows if r["mode"] == mode),
            mean_kl_to_raw=float(np.mean([r["projection"]["mean_kl_to_raw"] for r in rows if r["mode"] == mode])),
            mean_modal_gain_change=float(np.mean([r["projection"]["mean_modal_gain_change"] for r in rows if r["mode"] == mode])))
            for mode in effects},
        alignment={c: {k: sum(r["alignment"][c][k] for r in rows if r["alignment"] is not None)
            for k in next(r["alignment"][c] for r in rows if r["alignment"] is not None)} for c in CONTRASTS},
        demand_days=len(DAYS), original_training_seeds=len(pilot.SEEDS), sampling_repeats=3,
        cells=len(rows), rollouts=3 * len(rows), actor_updates=0, critic_updates=0,
        production_dual_updates=0, main_policy_changed=False, controller_admitted=False,
        local_moment_constraint_is_not_a_full_day_cv_guarantee=True))
    print("V81_PROJECTION_PILOT_COMPLETE " + json.dumps(gates), flush=True)


def smoke(root, v78_root, v56_root):
    for pattern in ("test_v78_*.py", "test_v81_*.py"):
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
            str(Path(__file__).resolve().parents[1] / "tests"), "-p", pattern, "-v"], check=True)
    rule = RegularityProjection()
    zero = pilot.rollout(root, pilot.SEEDS[0], pilot.joint.SCENARIOS[0], "deterministic", 181990000,
        [0.] * 4, "smoke_zero", probability_rule=rule)
    native = pilot.reference_trace(v56_root, pilot.SEEDS[0], pilot.joint.SCENARIOS[0])
    pilot.joint.check_unchanged(pilot.csv_episode(zero[2]), native["snapshot"]["episode"])
    pd.testing.assert_frame_equal(pd.DataFrame(zero[3]), native["holds"], check_exact=True)
    if rule.active:
        raise AssertionError("zero tilt must not activate a projection")
    run_item(root, v78_root, 0)
    pilot.write_json(root / "smoke" / "done.json", dict(status="tests_native_identity_and_three_policy_cell_passed",
                                                       exact_zero_tilt_path=True, reusable_cell=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v78-root", type=Path)
    parser.add_argument("--v56-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "smoke":
        smoke(args.out_root, args.v78_root, args.v56_root)
    elif args.command == "run":
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(run_item, args.out_root, args.v78_root, i) for i in range(args.start, args.end)]
            for task in pending:
                task.result()
        pilot.write_json(args.out_root / "shards" / f"{args.start}-{args.end}.json", dict(status="complete"))
    else:
        aggregate(args.out_root)


if __name__ == "__main__":
    main()
