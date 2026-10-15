#!/usr/bin/env python3
"""Prefix-only two-parameter travel mean reversion, screened before rollouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from freqduet_conditioned_travel import road_key
from freqduet_horizon_travel import fit_reversion, reverted_mean
from run_freqduet_v49_residual_dependence import cluster_summary
from run_freqduet_v52_clock_propagation import items
import run_freqduet_v46_travel_blocks as previous


def prepare_day(arguments):
    root, base, seed, scenario, cohort = arguments
    label = f"seed{seed}_eval{scenario}"
    legs = pd.read_csv(base / "protocol_v6_v47_boundary_apc" / "observed" / label /
        "completed_legs.csv", dtype={"direction": str})
    current = json.loads((base / "protocol_v6_v50_conditioned_travel" / "prepared" / label / "models.json").read_text())
    historical = {clock: json.loads((base / "protocol_v6_v48_passenger_uncertainty" / "origins" / label /
        f"clock_{clock}" / "forecasts.json").read_text())["travel_model"] for clock in previous.CLOCKS}
    records = []
    for clock in previous.CLOCKS:
        model, old = current[str(clock)], historical[clock]
        if model["latest_completion_s"] >= clock or old["latest_completion_s"] >= clock:
            raise ValueError("non-prefix source model in horizon fit")
        for row in legs.loc[(legs.start_s > clock) & (legs.start_s <= clock + 1800)].itertuples():
            key = road_key(row.direction, row.station_id)
            records.append(dict(origin=clock, start_s=row.start_s, end_s=row.end_s, actual=row.duration_s,
                current=model["means"][key], historical=old["means"][key]))
    records = pd.DataFrame(records)
    models = {clock: dict(current[str(clock)], historical_means=historical[clock]["means"],
        reversion=fit_reversion(records, clock)) for clock in previous.CLOCKS}
    scored_clocks = {item[-1] for item in items()}
    scores = []
    for row in records.loc[records.origin.isin(scored_clocks)].itertuples():
        candidate = float(reverted_mean(row.current, row.historical, row.start_s - row.origin,
            models[row.origin]["reversion"]["beta"]))
        for mode, predicted in (("v48", row.historical), ("v50", row.current), ("horizon", candidate)):
            scores.append(dict(mode=mode, lead="0-900" if row.start_s <= row.origin + 900 else "900-1800",
                error=predicted - row.actual, mae=abs(predicted - row.actual)))
    out = root / "prepared" / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "models.json").write_text(json.dumps(models) + "\n")
    pd.DataFrame(scores).groupby(["mode", "lead"])[["mae", "error"]].mean().to_csv(out / "scores.csv")
    pd.DataFrame([dict(clock=c, **m["reversion"]) for c, m in models.items()]).to_csv(out / "support.csv", index=False)
    print(f"HORIZON_PREPARED {label}", flush=True)


def prepare_summary(root):
    frame = pd.concat([pd.read_csv(root / "prepared" / f"seed{s}_eval{d}" / "scores.csv").assign(
        train_seed=s, eval_seed=d, cohort=c) for s, d, c in previous.previous.items()], ignore_index=True)
    reference = frame.loc[((frame.lead == "0-900") & (frame['mode'] == 'v50')) |
        ((frame.lead == "900-1800") & (frame['mode'] == 'v48')),
        ["train_seed", "eval_seed", "lead", "mae", "error"]]
    paired = frame.loc[frame['mode'] == 'horizon'].merge(reference,
        on=["train_seed", "eval_seed", "lead"], suffixes=("", "_reference"), validate="one_to_one")
    paired['mae_delta'] = paired.mae - paired.mae_reference
    mean = paired.groupby(["cohort", "lead"])[["mae", "mae_reference"]].mean()
    passed = bool((mean.mae <= mean.mae_reference).all())
    out = root / "prepare_summary"
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "day_paired.csv", index=False)
    result = cluster_summary(paired, ["lead"], ["mae_delta"])
    result.to_csv(out / "paired_clusters.csv", index=False)
    (out / "status.json").write_text(json.dumps(dict(prequential_pass=passed, policy_changed=False,
        demand_clusters=6, short_reference="v50", long_reference="v48",
        all_demand_seeds_previously_examined=True)) + "\n")
    print(mean.to_string(), flush=True)
    print(result.to_string(index=False), flush=True)
    print(f"HORIZON_PREQUENTIAL_PASS={passed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "prepare", "prepare-summary"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--data-base", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare-summary":
        prepare_summary(args.out_root)
    elif args.command == "smoke":
        suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1] /
            "tests"), pattern=pattern) for pattern in ("test_v50*.py", "test_v53*.py"))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        prepare_day((args.out_root, args.data_base, *previous.previous.items()[0]))
        out = args.out_root / "smoke_export"
        out.mkdir(parents=True, exist_ok=True)
        (out / "done.json").write_text('{"horizon_smoke": true}\n')
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            list(pool.map(prepare_day, [(args.out_root, args.data_base, *item)
                for item in previous.previous.items()[args.start:args.end]]))
        out = args.out_root / "prepare_exports" / f"batch_{args.start}_{args.end}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "done.json").write_text('{"horizon_preparation": true}\n')
