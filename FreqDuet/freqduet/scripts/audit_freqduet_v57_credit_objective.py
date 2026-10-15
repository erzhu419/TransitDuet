#!/usr/bin/env python3
"""Compare recorded lower scores, realized trip credit and full-day action effects."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from audit_freqduet_v36_actor_alignment import KEYS, load_records, summarize
from run_freqduet_v36_lower_credit import MODEL_ROOT, OUTCOMES


JOURNEY = "restricted_total_journey_horizon_min"
PRIMARY = (JOURNEY, "headway_cv", "fleet_readiness_delay_mean_s")


def choose(frame, column, maximize, actor):
    optimum = frame[column].max() if maximize else frame[column].min()
    tied = frame.loc[frame[column] == optimum].sort_values("action_s")
    incumbent = tied.loc[tied.action_s == actor]
    return (incumbent if len(incumbent) else tied).iloc[0]


def audit_context(values, rollouts, anchor):
    scores = values.loc[values.model_ep == anchor].sort_values("action_s")
    actions = rollouts.sort_values("action_s").copy()
    if (not np.array_equal(scores.action_s.to_numpy(), actions.action_s.to_numpy())
            or scores.lambda_value.nunique() != 1 or actions.actor_action_s.nunique() != 1):
        raise ValueError("recorded score and rollout contracts differ")
    if not actions.trip_done.all():
        raise ValueError("remaining-trip return is incomplete")
    actions["trip_objective"] = actions.trip_reward_return - float(scores.lambda_value.iloc[0]) * actions.trip_cost_return
    if not np.isfinite(actions[[*PRIMARY, "trip_objective"]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite objective or endpoint")
    actor = float(actions.actor_action_s.iloc[0])
    base = actions.loc[actions.action_s == actor].iloc[0]
    preferences = {
        "actor": base,
        "recorded_energy": actions.loc[actions.action_s == choose(scores, "energy", False, actor).action_s].iloc[0],
        "realized_trip_objective": choose(actions, "trip_objective", True, actor),
        "full_day_journey": choose(actions, JOURNEY, False, actor),
    }
    selected = []
    for name, row in preferences.items():
        result = dict(selector=name, actor_action_s=actor, selected_action_s=float(row.action_s),
                      action_delta_s=float(row.action_s - actor),
                      trip_objective_delta=float(row.trip_objective - base.trip_objective))
        result.update({f"delta_{m}": float(row[m] - base[m]) for m in OUTCOMES})
        result["all_three_no_harm"] = bool((row[list(PRIMARY)] <= base[list(PRIMARY)]).all())
        selected.append(result)
    trip_gain = actions.trip_objective - base.trip_objective
    journey_delta = actions[JOURNEY] - base[JOURNEY]
    alternatives = actions.action_s != actor
    local_best, global_best = preferences["realized_trip_objective"], preferences["full_day_journey"]
    incidence = dict(nonactor_actions=int(alternatives.sum()),
                     trip_gain_day_harm_actions=int(((trip_gain > 0) & (journey_delta > 0) & alternatives).sum()),
                     day_gain_trip_harm_actions=int(((trip_gain < 0) & (journey_delta < 0) & alternatives).sum()),
                     trip_best_day_harm=bool(local_best[JOURNEY] > base[JOURNEY]),
                     trip_best_day_gain=bool(local_best[JOURNEY] < base[JOURNEY]),
                     day_best_trip_harm=bool(global_best.trip_objective < base.trip_objective),
                     local_global_same_action=bool(local_best.action_s == global_best.action_s))
    return selected, incidence


def audit(root, destination):
    values, rollouts, source = load_records(root)
    score_groups = {key: group for key, group in values.groupby(KEYS)}
    selected, incidence = [], []
    for key, group in rollouts.groupby(KEYS, sort=True):
        records, counts = audit_context(score_groups[key], group, key[KEYS.index("anchor_lower_ep")])
        identity = dict(zip(KEYS, key))
        selected.extend([dict(identity, **row) for row in records])
        incidence.append(dict(identity, **counts))
    selected, incidence = pd.DataFrame(selected), pd.DataFrame(incidence)
    destination.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination / "selected_contexts.csv", index=False)
    incidence.to_csv(destination / "incidence_contexts.csv", index=False)
    summary = destination / "summary"
    summary.mkdir(exist_ok=True)
    groups = ["config", "anchor_lower_ep"]
    metrics = ["action_delta_s", "trip_objective_delta", "all_three_no_harm", *[f"delta_{m}" for m in OUTCOMES]]
    sys.path.insert(0, str(MODEL_ROOT))
    from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval
    summarize(selected, [*groups, "selector"], metrics, hierarchical_interval).to_csv(summary / "effects.csv", index=False)
    counts = incidence.groupby(groups)[[m for m in incidence if m not in KEYS]].sum()
    counts["contexts"] = incidence.groupby(groups).size()
    counts.to_csv(summary / "incidence.csv")
    selected.groupby([*groups, "selector"])[metrics].mean().to_csv(summary / "means.csv")
    status = dict(status="complete_development_credit_objective_audit", contexts=len(incidence),
                  actions=len(rollouts), source_contexts=source["contexts"], new_rollouts=0, training_runs=0,
                  main_policy_changed=False, independent_confirmation=False,
                  return_scope="recorded deterministic discounted remaining-trip reward minus recorded lambda times cost; no entropy return",
                  comparison="same state/action grid, continuation and deployment; selectors are hindsight diagnostics, not policies",
                  diagnostic_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"))
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    print(counts.to_string(), flush=True)
    main = selected.loc[selected.config.str.contains("confirmed_main")]
    print(main.groupby(["anchor_lower_ep", "selector"])[["action_delta_s", "trip_objective_delta", *[f"delta_{m}" for m in PRIMARY], "all_three_no_harm"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.v36_root, args.out_root)
