#!/usr/bin/env python3
"""Fresh paired policy-direction response; diagnostic, not candidate selection."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import run_freqduet_v79_policy_step_diagnosis as audit


pilot, step = audit.pilot, audit.step
DAYS = tuple(range(1800001, 1800033))
REPEATS = (0, 1)
ITEMS = list(itertools.product(pilot.SEEDS, DAYS, REPEATS))
EPSILON = .1
SCALES = dict(reference=0., minus=-EPSILON, plus=EPSILON, full=1.)
COSTS = ("passenger_person_s", "cv_squared", "readiness_delay_s")
UNITS = np.array([step.PASSENGER_SCALE, 1., 1.])


def responses(values):
    base = values["reference"]
    centered = (values["plus"] - values["minus"]) / (2. * EPSILON)
    full = values["full"] - base
    return dict(centered=centered, positive=(values["plus"] - base) / EPSILON,
                full=full, full_minus_centered=full - centered,
                small_positive_effect=values["plus"] - base)


def run_item(root, v78_root, index):
    path = root / "items" / f"i{index}.json"
    if path.exists():
        return index
    seed, day, rep = ITEMS[index]
    theta = np.asarray(json.loads((v78_root / "adapters" / f"seed{seed}.json").read_text())["theta"])
    sampler = 180000000 + seed + 100 * (day - DAYS[0]) + rep
    cell = root / "cells" / f"i{index}"
    results, records, alignments = {}, None, {}
    for label, scale in SCALES.items():
        row, events, _, _ = pilot.rollout(cell, seed, day, "stochastic", sampler,
            theta * scale, label, save_training=label == "reference")
        results[label] = row
        if label == "reference":
            records = events
        else:
            alignments[label] = pilot.event.alignment(records, events)
    pilot.write_json(path, dict(index=index, train_seed=seed, scenario_seed=day, rep=rep,
        theta=theta.tolist(), results=results, alignments=alignments))
    print("V80_DIRECTION_RESPONSE_DONE " + str(index), flush=True)
    return index


def score_direction(root, seed, theta):
    means, pseudo_values = [], []
    for rep in REPEATS:
        days = []
        for day in DAYS:
            index = ITEMS.index((seed, day, rep))
            with np.load(root / "cells" / f"i{index}" / "training" / f"seed{seed}_day{day}.npz") as saved:
                days.append(dict(saved))
        mean, pseudo = audit.jackknife_gradients(days)
        means.append((mean @ theta) * UNITS)
        pseudo_values.append((pseudo @ theta) * UNITS)
    return np.mean(means, axis=0), np.mean(pseudo_values, axis=0)


def paired_responses(rows, seeds):
    daily = []
    for day in DAYS:
        selected = [r for r in rows if r["scenario_seed"] == day and r["train_seed"] in seeds]
        daily.append({k: np.mean([responses({label: np.array([r["results"][label][m] for m in COSTS])
            for label in SCALES})[k] for r in selected], axis=0) for k in
            ("centered", "positive", "full", "full_minus_centered", "small_positive_effect")})
    return {k: np.asarray([r[k] for r in daily]) for k in daily[0]}


def contrast_summary(score_mean, score_pseudo, paired):
    result = {label: audit.uncertainty(values.mean(0), values) for label, values in paired.items()}
    result["score"] = audit.uncertainty(score_mean, score_pseudo)
    result["score_minus_centered"] = audit.uncertainty(
        score_mean - paired["centered"].mean(0), score_pseudo - paired["centered"])
    return result


def aggregate(root, v78_root):
    rows = [json.loads((root / "items" / f"i{i}.json").read_text()) for i in range(len(ITEMS))]
    by_seed, scores = {}, {}
    for seed in pilot.SEEDS:
        theta = np.asarray(json.loads((v78_root / "adapters" / f"seed{seed}.json").read_text())["theta"])
        for row in rows:
            if row["train_seed"] == seed:
                np.testing.assert_array_equal(row["theta"], theta)
        mean, pseudo = score_direction(root, seed, theta)
        scores[seed] = (mean, pseudo)
        by_seed[str(seed)] = contrast_summary(mean, pseudo, paired_responses(rows, (seed,)))
    mean = np.mean([v[0] for v in scores.values()], axis=0)
    pseudo = np.mean([v[1] for v in scores.values()], axis=0)
    pilot.write_json(root / "summary" / "status.json", dict(status="fixed_direction_response_diagnosis_complete",
        metrics=COSTS, epsilon=EPSILON, scales=SCALES, days=len(DAYS), original_training_seeds=len(pilot.SEEDS),
        repeats_per_day=len(REPEATS), rollouts=4 * len(rows),
        combined=contrast_summary(mean, pseudo, paired_responses(rows, pilot.SEEDS)), by_seed=by_seed,
        alignment={label: {k: sum(r["alignments"][label][k] for r in rows)
            for k in rows[0]["alignments"][label]} for label in SCALES if label != "reference"},
        inference_unit="demand day; score jackknife recomputes other-day baselines",
        small_perturbation_is_finite_not_exact_derivative=True, fitting_updates=0,
        actor_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        step_selection_performed=False, v78_gates_changed=False))
    print("V80_DIRECTION_RESPONSE_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "run", "aggregate"))
    parser.add_argument("--v78-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "smoke":
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
            str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v80_*.py", "-v"], check=True)
        run_item(args.out_root, args.v78_root, 0)
        pilot.write_json(args.out_root / "smoke" / "done.json", dict(status="tests_and_full_paired_cell_passed"))
    elif args.command == "run":
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(run_item, args.out_root, args.v78_root, i) for i in range(args.start, args.end)]
            for task in pending:
                task.result()
        pilot.write_json(args.out_root / "shards" / f"{args.start}-{args.end}.json", dict(status="complete"))
    else:
        aggregate(args.out_root, args.v78_root)


if __name__ == "__main__":
    main()
