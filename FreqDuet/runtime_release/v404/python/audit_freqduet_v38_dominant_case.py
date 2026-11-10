#!/usr/bin/env python3
"""Resolve V37's dominant development context into matched branch controls."""

import argparse
import json
from pathlib import Path

import pandas as pd

from audit_freqduet_v36_joint_space import SYSTEM_OUTCOMES, joint_space
from run_freqduet_v37_joint_lower import ACTIONS, OUTCOMES


def select_branches(group):
    group = group.sort_values(["first_action_s", "second_action_s"])
    reference = group.loc[(group.first_action_s == group.first_actor_s)
                          & (group.second_action_s == group.second_actor_s)]
    if len(reference) != 1:
        raise ValueError("actor reference is not unique")
    reference = reference.iloc[0]
    singles = group.loc[(group.second_action_s == group.second_actor_s)
                        | (group.first_action_s == group.first_actor_s)]
    _, single = joint_space(singles, reference)
    _, joint = joint_space(group, single)
    first_only = group.loc[(group.first_action_s == joint.first_action_s)
                          & (group.second_action_s == group.second_actor_s)].iloc[0]
    second_only = group.loc[(group.first_action_s == reference.first_action_s)
                           & (group.second_action_s == joint.second_action_s)].iloc[0]
    return dict(reference=reference, best_single=single, joint=joint,
                first_only=first_only, second_only=second_only)


def inspect(root, destination):
    group = pd.concat([pd.read_csv(root / "records" / f"seed933013_eval45007_t7200_a{a}.csv")
                       for a in ACTIONS], ignore_index=True)
    branches = select_branches(group)
    identities = [f"{decision}_{field}" for decision in ("first", "second")
                  for field in ("time_s", "bus_id", "trip_id", "station_id", "direction")]
    columns = ["first_action_s", "second_action_s", "first_actor_s", "second_actor_s",
               *identities, *OUTCOMES]
    plan = {name: json.loads(row[columns].to_json()) for name, row in branches.items()}
    result = dict(train_seed=933013, eval_seed=45007, target_time_s=7200,
                  scope="posthoc development mechanism diagnosis", branches=plan)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "action_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v37-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.v37_root, args.out_root)
