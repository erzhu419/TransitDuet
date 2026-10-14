#!/usr/bin/env python3
"""Audit fixed-state actor optimization using existing V36 counterfactuals."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


KEYS = ["config", "train_seed", "eval_seed", "anchor_lower_ep", "target_time_s"]
GROUPS = ["config", "anchor_lower_ep", "model_ep"]
OUTCOMES = (
    "headway_cv", "restricted_total_journey_horizon_min",
    "restricted_wait_horizon_min", "restricted_in_vehicle_horizon_min",
    "holding_vehicle_seconds_per_launched_trip", "holding_passenger_seconds",
    "fleet_readiness_delay_mean_s", "upper_interval_wait_cost_sum",
    "upper_interval_onboard_cost_sum", "upper_interval_dispatch_backlog_cost_sum",
    "trip_reward_return", "trip_cost_return",
)
ALIGNMENT = (
    "action_mismatch", "positive_energy_regret", "energy_regret",
    "energy_regret_over_alpha", "actor_minus_energy_action_s",
    "kl_actor_to_boltzmann", "total_variation", "soft_objective_gap",
    "actor_action_s", "energy_action_s", "actor_expected_action_s",
    "boltzmann_expected_action_s", "energy_margin", "alpha", "lambda_value",
    "reward_q_regret_component", "pessimism_regret_component",
    "cost_regret_component",
)
EFFECTS = tuple(f"energy_minus_actor_{metric}" for metric in OUTCOMES)


def audit_state(values, rollouts):
    values = values.sort_values("action_s").reset_index(drop=True)
    actions = values.action_s.to_numpy(dtype=float)
    if len(set(actions)) != len(actions) or len(actions) < 2:
        raise ValueError("actions must be distinct")
    probability = values.probability.to_numpy(dtype=float)
    energy = values.energy.to_numpy(dtype=float)
    alpha = float(values.alpha.iloc[0])
    if (not np.isfinite(probability).all() or (probability < 0).any()
            or not np.isclose(probability.sum(), 1.0, atol=1e-6, rtol=0.0)):
        raise ValueError("invalid recorded categorical probabilities")
    if (not np.isfinite(energy).all() or not np.isfinite(alpha) or alpha <= 0
            or not values.alpha.eq(alpha).all()):
        raise ValueError("invalid recorded energy or temperature")
    probability = probability / probability.sum()
    actor = int(probability.argmax())
    best = int(energy.argmin())
    shifted_energy = energy - energy[best]
    log_weights = -shifted_energy / alpha
    log_partition = float(np.log(np.exp(log_weights).sum()))
    log_boltzmann = log_weights - log_partition
    boltzmann = np.exp(log_boltzmann)
    log_probability = np.zeros_like(probability)
    np.log(probability, out=log_probability, where=probability > 0)
    kl = float(np.sum(probability * (log_probability - log_boltzmann)))
    objective_gap = float(np.sum(probability * (
        shifted_energy + alpha * log_probability)) + alpha * log_partition)
    row = dict(
        action_mismatch=float(actor != best),
        positive_energy_regret=float(shifted_energy[actor] > 1e-6),
        energy_regret=float(shifted_energy[actor]),
        energy_regret_over_alpha=float(shifted_energy[actor] / alpha),
        actor_minus_energy_action_s=float(actions[actor] - actions[best]),
        kl_actor_to_boltzmann=kl,
        total_variation=float(np.abs(probability - boltzmann).sum() / 2),
        soft_objective_gap=objective_gap,
        actor_action_s=float(actions[actor]), energy_action_s=float(actions[best]),
        actor_expected_action_s=float(probability @ actions),
        boltzmann_expected_action_s=float(boltzmann @ actions),
        energy_margin=float(np.sort(energy)[1] - energy[best]),
        alpha=alpha, lambda_value=float(values.lambda_value.iloc[0]),
        reward_q_regret_component=float(
            values.q_mean.iloc[best] - values.q_mean.iloc[actor]),
        pessimism_regret_component=float(
            values.pessimism.iloc[actor] - values.pessimism.iloc[best]),
        cost_regret_component=float(
            values.cost_penalty.iloc[actor] - values.cost_penalty.iloc[best]),
    )
    if not np.isclose(objective_gap, alpha * kl, atol=1e-8):
        raise ValueError("soft objective gap does not equal alpha times KL")
    components = sum(row[name] for name in (
        "reward_q_regret_component", "pessimism_regret_component", "cost_regret_component"))
    if not np.isclose(components, row["energy_regret"], atol=1e-5):
        raise ValueError("recorded energy does not match its objective components")
    outcomes = rollouts.set_index("action_s", verify_integrity=True)
    if set(outcomes.index) != set(actions):
        raise ValueError("counterfactual actions do not match value actions")
    for metric in OUTCOMES:
        row[f"energy_minus_actor_{metric}"] = float(
            outcomes.loc[actions[best], metric] - outcomes.loc[actions[actor], metric])
    return row


def load_records(root):
    status = json.loads((root / "summary/status.json").read_text())
    ready = sorted((root / "ready").glob("*/ready.json"))
    if (status["status"] != "complete_exploratory_lower_credit"
            or len(ready) != status["cases"]):
        raise ValueError("V36 cohort has not completed aggregation")
    values, rollouts = [], []
    for path in ready:
        record = root / "records" / path.parent.name
        values.append(pd.read_csv(record / "action_values.csv"))
        rollouts.append(pd.read_csv(record / "rollouts.csv"))
    values, rollouts = pd.concat(values, ignore_index=True), pd.concat(rollouts, ignore_index=True)
    for frame, keys, expected_size in (
            (values, [*KEYS, "model_ep"], 14), (rollouts, KEYS, 7)):
        if frame.duplicated([*keys, "action_s"]).any():
            raise ValueError("duplicate action record")
        # Both historical models must cover the seven existing actions in every context.
        if not frame.groupby(KEYS).size().eq(expected_size).all():
            raise ValueError("incomplete action/model context")
        if not set(frame.action_s) == {0, 5, 10, 15, 20, 30, 45}:
            raise ValueError("unexpected action alphabet")
    if (set(values.model_ep) != {39, 199}
            or not values.groupby([*KEYS, "model_ep"]).size().eq(7).all()):
        raise ValueError("historical model coverage differs from V36")
    contexts = set(map(tuple, values[KEYS].drop_duplicates().to_numpy()))
    if (len(contexts) != status["contexts"]
            or contexts != set(map(tuple, rollouts[KEYS].drop_duplicates().to_numpy()))):
        raise ValueError("value and rollout contexts do not match the cohort")
    return values, rollouts, status


def summarize(frame, groups, metrics, interval):
    rows = []
    for key, group in frame.groupby(groups, sort=True):
        paired = group.groupby(["train_seed", "eval_seed"])[list(metrics)].mean().reset_index()
        for metric in metrics:
            low, high = interval(paired, metric)
            rows.append(dict(zip(groups, key), metric=metric, mean=float(paired[metric].mean()),
                             ci95_low=low, ci95_high=high))
    return pd.DataFrame(rows)


def aggregate(root, destination, statistics_root):
    values, rollouts, source_status = load_records(root)
    rollout_groups = {key: group for key, group in rollouts.groupby(KEYS)}
    rows = []
    for key, group in values.groupby([*KEYS, "model_ep"], sort=True):
        rows.append(dict(zip([*KEYS, "model_ep"], key),
                         **audit_state(group, rollout_groups[key[:-1]])))
    states = pd.DataFrame(rows)
    destination.mkdir(parents=True, exist_ok=True)
    # State-level diagnostics stay server-side; only the summary directory is exported.
    states.to_csv(destination / "per_state.csv", index=False)
    summary = destination / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(statistics_root))
    from scripts.run_freqduet_protocol_v2_matrix import hierarchical_interval
    metrics = (*ALIGNMENT, *EFFECTS)
    effects = summarize(states, GROUPS, metrics, hierarchical_interval)
    effects.to_csv(summary / "state_effects.csv", index=False)
    states.groupby([*GROUPS, "train_seed"])[list(metrics)].mean().to_csv(
        summary / "training_seed_means.csv")
    early = states.loc[states.model_ep == 39].set_index(KEYS)
    late = states.loc[states.model_ep == 199].set_index(KEYS)
    change = (late[list(metrics)] - early[list(metrics)]).reset_index()
    summarize(change, ["config", "anchor_lower_ep"], metrics, hierarchical_interval).to_csv(
        summary / "late_minus_early.csv", index=False)
    descriptive = []
    for key, group in states.groupby(GROUPS):
        mismatch = group.loc[group.action_mismatch == 1]
        row = dict(zip(GROUPS, key), states=len(group), mismatches=len(mismatch),
                   kl_median=float(group.kl_actor_to_boltzmann.median()),
                   kl_p90=float(group.kl_actor_to_boltzmann.quantile(0.9)),
                   energy_regret_p90=float(group.energy_regret.quantile(0.9)))
        row.update({f"mismatch_mean_{metric}": float(mismatch[metric].mean())
                    if len(mismatch) else None for metric in EFFECTS})
        descriptive.append(row)
    pd.DataFrame(descriptive).to_csv(summary / "mismatch_descriptive.csv", index=False)
    status = dict(status="complete_exploratory_actor_alignment", contexts=source_status["contexts"],
                  state_models=len(states), new_rollouts=0, training_runs=0,
                  independent_confirmation=False, source_status=source_status,
                  analysis_commit=os.environ.get("FREQDUET_DIAGNOSTIC_COMMIT"),
                  contrast="energy-minimizing action minus actor action, one intervention only",
                  scope="fixed recorded Q/cost/alpha; deterministic branch outcomes, not soft-Q truth")
    (summary / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    main = effects.loc[effects.config.str.contains("confirmed_main") & effects.metric.isin(
        ["action_mismatch", "energy_regret", "kl_actor_to_boltzmann",
         "actor_minus_energy_action_s", "energy_minus_actor_restricted_total_journey_horizon_min",
         "energy_minus_actor_holding_vehicle_seconds_per_launched_trip"])]
    print(main.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v36-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--statistics-root", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.v36_root, args.out_root, args.statistics_root)


if __name__ == "__main__":
    main()
