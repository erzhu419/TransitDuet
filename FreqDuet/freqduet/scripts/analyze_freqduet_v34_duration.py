#!/usr/bin/env python3
"""Separate absolute learning progress from changes in relative V34 effects."""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from run_freqduet_protocol_v6_checkpoint_trajectory import (
    CONFIGS, CONTEXT, CURRENT, METRICS, MODEL_ROOT, REFERENCE,
    V9_TRAIN_SEEDS, V33_TRAIN_SEEDS, V9_EVAL_SEEDS, V33_EVAL_SEEDS,
    checkpoint_change, paired_trajectory,
)


TEMPORAL_METRICS = (
    "headway_cv", "restricted_total_journey_horizon_min",
    "lower_action_std", "holding_vehicle_seconds_per_launched_trip",
    "upper_delta_std",
)


def duration_analysis(frame, interval):
    means = frame.groupby(["config", "checkpoint_ep"])[list(METRICS)].mean().reset_index()
    cohorts = {
        "all_train_all_eval": (None, None),
        "v9_train_v9_eval": (V9_TRAIN_SEEDS, V9_EVAL_SEEDS),
        "v33_train_v9_eval": (V33_TRAIN_SEEDS, V9_EVAL_SEEDS),
        "v9_train_v33_eval": (V9_TRAIN_SEEDS, V33_EVAL_SEEDS),
        "v33_train_v33_eval": (V33_TRAIN_SEEDS, V33_EVAL_SEEDS),
    }
    temporal, seed_changes = [], []
    for config in CONFIGS:
        rows = frame.loc[frame.config == config]
        change = checkpoint_change(rows, 39, 199)
        per_seed = change.groupby("train_seed")[list(TEMPORAL_METRICS)].mean().reset_index()
        per_seed.insert(0, "config", config)
        seed_changes.append(per_seed)
        for cohort, (train_seeds, eval_seeds) in cohorts.items():
            subset = change if train_seeds is None else change.loc[
                change.train_seed.isin(train_seeds) & change.eval_seed.isin(eval_seeds)]
            for metric in TEMPORAL_METRICS:
                low, high = interval(subset, metric)
                temporal.append(dict(config=config, cohort=cohort, metric=metric,
                                     delta_ep199_minus_ep39=float(subset[metric].mean()),
                                     ci95_low=low, ci95_high=high,
                                     positive_train_seed_fraction=float(
                                         subset.groupby("train_seed")[metric].mean().gt(0).mean())))
    relative = []
    for reference in (REFERENCE, CONTEXT):
        paired = paired_trajectory(frame, CURRENT, reference)
        rows = checkpoint_change(paired, 39, 199)
        rows = rows.groupby("train_seed")[list(TEMPORAL_METRICS)].mean().reset_index()
        rows.insert(0, "reference", reference)
        relative.append(rows)
    return means, pd.DataFrame(temporal), pd.concat(seed_changes), pd.concat(relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.out_root / "summary/diagnostic_status.json").read_text())
    if status["status"] != "complete_exploratory_diagnostic" or not status["matches_archived_v9"]:
        raise ValueError("V34 completion and historical reproduction must precede duration analysis")
    sys.path.insert(0, str(MODEL_ROOT))
    from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval

    frame = pd.concat([pd.read_csv(path) for path in
                       sorted((args.out_root / "records").glob("*.csv"))], ignore_index=True)
    tables = duration_analysis(frame, hierarchical_interval)
    names = ("absolute_checkpoint_means.csv", "absolute_late_minus_early.csv",
             "training_seed_absolute_changes.csv", "training_seed_relative_changes.csv")
    for name, table in zip(names, tables):
        table.to_csv(args.out_root / "summary" / name, index=False)
    print(tables[0].to_string(index=False), flush=True)
    primary = tables[1].loc[tables[1].cohort == "all_train_all_eval"]
    print(primary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
