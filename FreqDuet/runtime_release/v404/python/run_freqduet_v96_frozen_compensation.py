#!/usr/bin/env python3
"""Fresh-day confirmation of the unchanged V93/V95 compensation regimes."""

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

import run_freqduet_v95_compensation_specificity as prior


confirmation, frozen, pilot = prior.confirmation, prior.frozen, prior.pilot
DAYS = tuple(range(1961001, 1961033))
ITEMS = list(itertools.product(confirmation.VALIDATION_SEEDS, DAYS, range(3)))
MODES = ("late", "preserve", "uniform")
CONTRASTS = prior.CONTRASTS
KEY_METRICS = ("journey_min", "cv_squared", "readiness_delay_s", "holding_s_per_trip")
DOSE_METRICS = ("expected_reduction_sum_s", "expected_reduction_minus_shadow_preserve_s",
                "compensation_loss_expected_reduction_sum_s")


def axis_summary(rows, field, metrics):
    result = {}
    for contrast in rows[0][field]:
        frame = pd.DataFrame([dict(seed=r["seed"], day=r["day"], **r[field][contrast]) for r in rows])
        frame = frame.groupby(["seed", "day"], as_index=False)[list(metrics)].mean()
        result[contrast] = {}
        for axis in ("seed", "day"):
            means = frame.groupby(axis)[list(metrics)].mean()
            result[contrast][axis] = dict(means={str(k): v.to_dict() for k, v in means.iterrows()},
                negative_counts={m: int((means[m] < 0).sum()) for m in metrics},
                positive_counts={m: int((means[m] > 0).sum()) for m in metrics},
                jointly_negative=int((means < 0).all(axis=1).sum()))
    return result


def prepare(root, v95_root):
    previous = json.loads((v95_root / "prepare/status.json").read_text())
    rows = [json.loads((v95_root / "pairs" / f"p{i}.json").read_text()) for i in range(len(prior.ITEMS))]
    pilot.write_json(root / "prepare/status.json", dict(status="v95_axes_reviewed_and_original_probability_retained",
        probability=previous["probability"], probability_source=previous["probability_source"],
        v95_effect_axes=axis_summary(rows, "effects", KEY_METRICS),
        v95_dose_axes=axis_summary(rows, "dose_effects", DOSE_METRICS),
        new_rollouts=0, future_days_read=False, main_policy_changed=False))
    print("V96_PREPARE_COMPLETE", flush=True)


def collect(root, seed, day, rep, index, probability, *, artifact_root):
    actual = {}
    for mode in MODES:
        actual[mode] = (prior.replay(root, seed, day, rep, probability) if mode == "uniform"
                        else frozen.replay(root, seed, day, rep, mode))
        actual[mode][3].save(artifact_root / "records" / f"p{index}_{mode}.npz")
    rows = {mode: r[0] for mode, r in actual.items()}
    contexts = {mode: r[3].group_summary for mode, r in actual.items()}
    doses = {mode: prior.dose(contexts[mode], mode, probability) for mode in ("preserve", "uniform")}
    result = dict(index=index, seed=seed, day=day, rep=rep, results=rows, contexts=contexts,
        effects={name: {m: rows[a][m] - rows[b][m] for m in frozen.METRICS} for name, (a, b) in CONTRASTS.items()},
        timing_effects={name: confirmation.timing_effect(actual[b][3], actual[a][3]) for name, (a, b) in CONTRASTS.items()},
        doses=doses, dose_effects=dict(uniform_vs_preserve={m: doses["uniform"][m] - doses["preserve"][m] for m in doses["uniform"]}),
        probability=probability)
    pilot.write_json(artifact_root / "pairs" / f"p{index}.json", result)
    print("V96_FROZEN_CELL_DONE " + str(index), flush=True)
    return result


def run_cell(root, index, probability):
    if not (root / "pairs" / f"p{index}.json").exists():
        collect(root, *ITEMS[index], index, probability, artifact_root=root)


def aggregate(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(ITEMS))]
    if [(r["seed"], r["day"], r["rep"]) for r in rows] != ITEMS:
        raise AssertionError("V96 rows differ from the fixed fresh-day grid")
    effects = confirmation.describe(rows, "effects")
    result = dict(status="frozen_compensation_fresh_day_confirmation_complete", effects=effects,
        timing_effects=confirmation.describe(rows, "timing_effects"), dose_effects=confirmation.describe(rows, "dose_effects"),
        by_seed={str(s): confirmation.day_stats([r for r in rows if r["seed"] == s]) for s in confirmation.VALIDATION_SEEDS},
        effect_axes=axis_summary(rows, "effects", KEY_METRICS), dose_axes=axis_summary(rows, "dose_effects", DOSE_METRICS),
        absolute={mode: {m: float(np.mean([r["results"][mode][m] for r in rows])) for m in frozen.METRICS} for mode in MODES},
        absolute_expected_doses={mode: {m: float(np.mean([r["doses"][mode][m] for r in rows])) for m in rows[0]["doses"][mode]}
                                 for mode in ("preserve", "uniform")},
        preserve_joint_gate_passed=confirmation.joint_gate(effects),
        uniform_joint_gate_passed=confirmation.joint_gate(dict(preserve_vs_late=effects["uniform_vs_late"])),
        direct_joint_gate_passed=confirmation.joint_gate(dict(preserve_vs_late=effects["preserve_vs_uniform"])),
        direct_regularity_improved=effects["preserve_vs_uniform"]["cv_squared"]["ci95"][1] < 0,
        primary_contrast="preserve_vs_late", mechanism_contrast="preserve_vs_uniform",
        probability=json.loads((root / "prepare/status.json").read_text())["probability"],
        evaluation_rollouts=len(ITEMS) * len(MODES), archived_training_seeds=list(confirmation.VALIDATION_SEEDS),
        demand_days=list(DAYS), sampler_repeats=3, bootstrap="5000 crossed actor/day draws after repeat averaging",
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False, controller_admitted=False,
        original_gate_changed=False, probability_refitted=False,
        scope="one fixed fresh-day confirmation; dynamic regime effects, not exact dose mediation, training repair or fixed-baseline superiority")
    pilot.write_json(root / "summary/status.json", result)
    print("V96_FROZEN_CONFIRMATION_COMPLETE " + json.dumps(effects["preserve_vs_uniform"]), flush=True)


def smoke_case(root, v95_root, seed, probability):
    index = next(i for i, r in enumerate(prior.ITEMS) if r[1] == seed)
    _, seed, day, rep = prior.ITEMS[index]
    current = collect(root, seed, day, rep, index, probability, artifact_root=root / "smoke_runs")
    old = json.loads((v95_root / "pairs" / f"p{index}.json").read_text())
    for field in ("effects", "timing_effects", "dose_effects"):
        pilot.joint.assert_same_state(current[field], old[field], "v96_consumed_v95_" + field)


def smoke(root, v95_root, probability):
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
        str(Path(__file__).resolve().parents[1] / "tests"), "-p", "test_v96_*.py", "-v"], check=True)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        for task in [pool.submit(smoke_case, root, v95_root, seed, probability) for seed in (12037, 933031)]:
            task.result()
    pilot.write_json(root / "smoke/done.json", dict(status="collector_tests_and_two_consumed_v95_cells_reproduced",
        future_days_read=False, probability=probability))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "run", "aggregate"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v95-root", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=len(ITEMS))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out_root, args.v95_root)
    elif args.command == "aggregate":
        aggregate(args.out_root)
    else:
        probability = json.loads((args.out_root / "prepare/status.json").read_text())["probability"]
        if args.command == "smoke":
            smoke(args.out_root, args.v95_root, probability)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
                for task in [pool.submit(run_cell, args.out_root, i, probability) for i in range(args.start, args.end)]:
                    task.result()
            pilot.write_json(args.out_root / f"shards/{args.start}-{args.end}.json", dict(status="complete"))
