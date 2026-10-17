#!/usr/bin/env python3
"""Confirm the unchanged V93 rule across archived actors and fresh days."""

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

import run_freqduet_v93_release_holding as frozen
from run_freqduet_protocol_v6_checkpoint_trajectory import TRAIN_SEEDS
from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval


pilot = frozen.pilot
DEVELOPMENT_SEEDS = tuple(pilot.SEEDS)
VALIDATION_SEEDS = tuple(s for s in TRAIN_SEEDS if s not in DEVELOPMENT_SEEDS)
DAYS = tuple(range(1941001, 1941017))
ITEMS = list(itertools.product(TRAIN_SEEDS, DAYS, range(3)))
MODES = ("late", "preserve", "no_gain_early")
CONTRASTS = dict(preserve_vs_late=("preserve", "late"), all_vs_late=("no_gain_early", "late"),
    all_vs_preserve=("no_gain_early", "preserve"))


def timing_effect(base, candidate):
    b, c = base.coordinates, candidate.coordinates
    delta = (np.asarray(c["covariance"]) - b["covariance"]) / b["mean"] ** 2
    parts = dict(release_diagonal=float(delta[0, 0]), holding_diagonal=float(delta[1, 1]),
        other_diagonal=float(delta[2, 2]), release_holding_cross=float(2 * delta[0, 1]),
        release_other_cross=float(2 * delta[0, 2]), holding_other_cross=float(2 * delta[1, 2]))
    measured = frozen.timing.component_effect(base.summary["arrival"], candidate.summary["arrival"])
    np.testing.assert_allclose(sum(parts.values()) + measured["mean_normalization"],
                               c["cv_squared"] - b["cv_squared"], atol=1e-10, rtol=0)
    return dict(**parts, **measured)


def collect(root, seed, day, rep, index, *, artifact_root):
    actual = {}
    for mode in MODES:
        actual[mode] = frozen.replay(root, seed, day, rep, mode)
        actual[mode][3].save(artifact_root / "records" / f"p{index}_{mode}.npz")
    rows = {mode: r[0] for mode, r in actual.items()}
    result = dict(index=index, seed=seed, day=day, rep=rep, results=rows,
        contexts={mode: r[3].group_summary for mode, r in actual.items()},
        effects={name: {m: rows[a][m] - rows[b][m] for m in frozen.METRICS}
                 for name, (a, b) in CONTRASTS.items()},
        timing_effects={name: timing_effect(actual[b][3], actual[a][3]) for name, (a, b) in CONTRASTS.items()},
        alignment={mode: pilot.event.alignment(actual["late"][1], actual[mode][1]) for mode in MODES[1:]})
    pilot.write_json(artifact_root / "pairs" / f"p{index}.json", result)
    print("V94_CONFIRMATION_CELL_DONE " + str(index), flush=True)
    return result, actual["late"][1], actual["late"][2]


def run_cell(root, index):
    if not (root / "pairs" / f"p{index}.json").exists():
        collect(root, *ITEMS[index], index, artifact_root=root)


def describe(rows, field):
    result = {}
    for contrast, metrics in rows[0][field].items():
        frame = pd.DataFrame([dict(train_seed=r["seed"], eval_seed=r["day"], **r[field][contrast]) for r in rows])
        frame = frame.groupby(["train_seed", "eval_seed"], as_index=False)[list(metrics)].mean()
        result[contrast] = {m: dict(mean=float(frame[m].mean()), ci95=list(hierarchical_interval(frame, m)),
            n_training_seeds=int(frame.train_seed.nunique()), n_demand_days=int(frame.eval_seed.nunique())) for m in metrics}
    return result


def day_stats(rows):
    result = {}
    for contrast, metrics in rows[0]["effects"].items():
        days = sorted({r["day"] for r in rows})
        result[contrast] = {}
        for metric in metrics:
            values = [float(np.mean([r["effects"][contrast][metric] for r in rows if r["day"] == d])) for d in days]
            stats = pilot.event.labels.expanded.paired_interval(values, 94000)
            result[contrast][metric] = {k: stats[k] for k in ("n_demand_days", "mean", "ci95")}
    return result


def joint_gate(stats):
    primary = stats["preserve_vs_late"]
    return primary["passenger_person_s"]["ci95"][1] < 0 and all(primary[m]["ci95"][1] <= 0
        for m in ("cv_squared", "readiness_delay_s", "trips_unlaunched", "trips_incomplete"))


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    if [(r["seed"], r["day"], r["rep"]) for r in rows] != ITEMS:
        raise AssertionError("confirmation rows differ from the registered actor/day/repeat grid")
    cohorts = {name: [r for r in rows if r["seed"] in seeds] for name, seeds in (
        ("validation_actors", VALIDATION_SEEDS), ("development_actor_replication", DEVELOPMENT_SEEDS), ("all_actors", TRAIN_SEEDS))}
    summaries = {name: dict(effects=describe(chosen, "effects"), timing_effects=describe(chosen, "timing_effects"))
                 for name, chosen in cohorts.items()}
    result = dict(status="frozen_compensation_confirmation_complete", cohorts=summaries,
        primary_cohort="validation_actors", primary_contrast="preserve_vs_late", mechanism_contrast="all_vs_preserve",
        primary_joint_diagnostic_gate_passed=joint_gate(summaries["validation_actors"]["effects"]),
        by_seed={str(seed): day_stats([r for r in rows if r["seed"] == seed]) for seed in TRAIN_SEEDS},
        absolute={mode: {m: float(np.mean([r["results"][mode][m] for r in rows])) for m in frozen.METRICS} for mode in MODES},
        context_totals={mode: {g: {k: float(sum(r["contexts"][mode][g][k] for r in rows))
            for k in (*frozen.GROUP_FIELDS, "selected")} for g in frozen.GROUPS} for mode in MODES},
        evaluation_rollouts=len(ITEMS) * len(MODES), archived_training_seeds=list(TRAIN_SEEDS),
        validation_training_seeds=list(VALIDATION_SEEDS), development_training_seeds=list(DEVELOPMENT_SEEDS),
        demand_days=list(DAYS), sampler_repeats=3, bootstrap="existing crossed actor/day bootstrap; 5000 draws after repeat averaging",
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        original_gate_changed=False, frozen_v93_source="0561d3776e", alternative_arms_are_not_a_winner_search=True,
        scope="archived actor holdout for the V93 rule and fresh demand days, not new training or fixed-baseline superiority")
    pilot.write_json(root / "summary/status.json", result)
    print("V94_COMPENSATION_CONFIRMATION_COMPLETE " + json.dumps(dict(
        primary=result["cohorts"]["validation_actors"]["effects"]["preserve_vs_late"],
        joint_gate=result["primary_joint_diagnostic_gate_passed"])), flush=True)


def smoke_case(root, seed, index):
    day, rep = frozen.DAYS[0], 0
    current, events, holds = collect(root, seed, day, rep, index, artifact_root=root / "smoke_runs")
    original = pilot.rollout(root, seed, day, "stochastic", 187000000 + seed + 100 * day,
        [0.] * 4, f"v94_native_seed{seed}")
    for metric in (*pilot.METRICS, "action_counts"):
        pilot.joint.assert_same_state(original[0][metric], current["results"]["late"][metric], "v94_native_endpoints")
    pilot.joint.assert_same_state(original[1], events, "v94_native_sampling_path")
    pilot.joint.assert_same_state(original[3], holds, "v94_native_executed_holding")
    for mode in MODES[1:]:
        if sum(g["selected"] for g in current["contexts"][mode].values()) == 0:
            raise AssertionError("confirmation smoke must execute both frozen interventions")


def smoke(root):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v94_*.py", "-v"], check=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(smoke_case, root, seed, i) for i, seed in enumerate((12037, 933031))]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="additional_actor_paths_frozen_regimes_and_crossed_statistics_passed",
        prospective_days_read=False, actors=(12037, 933031)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.out_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for task in [pool.submit(run_cell, args.out_root, i) for i in range(args.start, args.end)]:
                task.result()
        pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))
