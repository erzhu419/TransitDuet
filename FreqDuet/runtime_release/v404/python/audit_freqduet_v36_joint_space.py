#!/usr/bin/env python3
"""Hindsight joint-endpoint space in the existing seven-action V36 bank."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from audit_freqduet_v36_actor_alignment import KEYS, OUTCOMES, load_records, summarize


PRIMARY = ("restricted_total_journey_horizon_min", "headway_cv", "fleet_readiness_delay_mean_s")
SYSTEM_OUTCOMES = OUTCOMES[:-2]


def joint_space(candidates, baseline):
    """Select a hindsight journey minimum subject to no harm on all three endpoints."""
    values = candidates[list(PRIMARY)].to_numpy(dtype=float)
    base = baseline[list(PRIMARY)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(base).all():
        raise ValueError("non-finite joint endpoint")
    delta = values - base
    feasible = (delta <= 0).all(axis=1)
    if not feasible.any():
        raise ValueError("candidate bank lacks a jointly feasible reference")
    alternatives = feasible & (delta < 0).any(axis=1)
    chosen = candidates.loc[feasible].sort_values(list(PRIMARY), kind="stable").iloc[0]
    result = dict(
        has_pareto_improvement=float(alternatives.any()),
        has_all_three_strict_improvement=float((delta < 0).all(axis=1).any()),
        has_journey_improvement=float((feasible & (delta[:, 0] < 0)).any()),
        feasible_actions=int(feasible.sum()), pareto_improving_actions=int(alternatives.sum()),
    )
    result.update({f"selected_delta_{metric}": float(chosen[metric] - baseline[metric])
                   for metric in SYSTEM_OUTCOMES})
    return result, chosen


def aggregate(root, destination, statistics_root):
    _, rollouts, source_status = load_records(root)
    rows = []
    for key, group in rollouts.groupby(KEYS, sort=True):
        if group.actor_action_s.nunique() != 1:
            raise ValueError("reference actor action differs within one context")
        actor = float(group.actor_action_s.iloc[0])
        baseline = group.set_index("action_s", verify_integrity=True).loc[actor]
        result, chosen = joint_space(group.sort_values("action_s"), baseline)
        rows.append(dict(zip(KEYS, key), actor_action_s=actor,
                         selected_action_s=float(chosen.action_s), **result))
    contexts = pd.DataFrame(rows)
    destination.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(destination / "per_context.csv", index=False)
    summary = destination / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(statistics_root))
    from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval
    metrics = [name for name in contexts if name.startswith(("has_", "selected_delta_"))]
    metrics += ["feasible_actions", "pareto_improving_actions"]
    groups = ["config", "anchor_lower_ep"]
    effects = summarize(contexts, groups, metrics, hierarchical_interval)
    effects.to_csv(summary / "joint_space.csv", index=False)
    contexts.groupby([*groups, "train_seed"])[metrics].mean().to_csv(
        summary / "training_seed_means.csv")
    descriptive = contexts.groupby(groups).agg(
        contexts=("actor_action_s", "size"),
        pareto_contexts=("has_pareto_improvement", "sum"),
        journey_contexts=("has_journey_improvement", "sum"),
        strict_three_contexts=("has_all_three_strict_improvement", "sum"),
        median_selected_journey_delta=("selected_delta_restricted_total_journey_horizon_min", "median"),
        min_selected_journey_delta=("selected_delta_restricted_total_journey_horizon_min", "min"))
    descriptive.to_csv(summary / "descriptive.csv")
    status = dict(status="complete_exploratory_single_action_joint_space", contexts=len(contexts),
                  source_status=source_status, new_rollouts=0, training_runs=0,
                  analysis_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"),
                  selection="hindsight journey minimum; journey/CV/readiness each <= reference",
                  precision="strict comparison of recorded CSV outcomes; no service-effect threshold",
                  independent_confirmation=False, deployable_policy=False)
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(descriptive.to_string(), flush=True)
    print(effects.loc[effects.config.str.contains("confirmed_main")].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--statistics-root", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.v36_root, args.out_root, args.statistics_root)
