#!/usr/bin/env python3
"""Posthoc V68 strict-winner, fixed-action and horizon diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_freqduet_v68_repeated_continuations as audit


METRICS = ("passenger_person_s", "cv_squared", "readiness_delay_s")


def interval(rows, getter):
    result = audit.day_intervals(rows, getter, 68300)
    return {key: result[key] for key in ("n_demand_days", "mean", "ci95")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--v67-root", type=Path, required=True)
    args = parser.parse_args()
    contexts = json.loads((args.out_root / "groups/context_audit.json").read_text())
    old = {c["context_index"]: c for c in json.loads(
        (args.v67_root / "labels/predictor_audit.json").read_text())}
    strict = {i for i, c in old.items() if c["oracle_person_s"] < 0}
    ties = {c["context_index"] for c in contexts
            if c["v67_replication"]["changed"] and c["context_index"] not in strict}
    result = dict(strict_v67_winners=len(strict), zero_gain_changed_ties=len(ties))
    for name, indices in (("strict_winners_contribution", strict), ("zero_gain_ties_contribution", ties)):
        # Retain all 96 prefixes in the denominator to decompose the primary mean.
        result[name] = {metric: interval(contexts, lambda c, m=metric:
                                         c["v67_replication"][m] if c["context_index"] in indices else 0.)
                        for metric in METRICS}
    winners = [c for c in contexts if c["context_index"] in strict]
    result["strict_winner_replication_counts"] = dict(
        passenger_improved=sum(c["v67_replication"]["passenger_person_s"] < 0 for c in winners),
        passenger_worsened=sum(c["v67_replication"]["passenger_person_s"] > 0 for c in winners),
        passenger_tied=sum(c["v67_replication"]["passenger_person_s"] == 0 for c in winners),
        joint_mean_nonworse=sum(c["v67_replication"]["passenger_person_s"] < 0
                               and all(c["v67_replication"][m] <= 0 for m in audit.CONSTRAINT_METRICS)
                               for c in winners))
    halves = [h for c in contexts for h in c["cross_future_halves"]]
    result["cross_future_changes"] = sum(h["changed"] for h in halves)
    result["cross_future_changed_halves"] = dict(
        improved=sum(h["changed"] and h["passenger_person_s"] < 0 for h in halves),
        worsened=sum(h["changed"] and h["passenger_person_s"] > 0 for h in halves))
    action_contexts = []
    for context in contexts:
        groups = [json.loads(audit.group_path(args.out_root, context["context_index"] * len(audit.FUTURES) + f).read_text())
                  for f in audit.FUTURES]
        native = audit.joint.ACTIONS.index(context["native_action_s"])
        arrays = {m: np.asarray([[r[m] for r in g["rows"]] for g in groups]) for m in METRICS}
        action_contexts.append(dict(
            scenario_seed=context["scenario_seed"],
            effects={m: (a - a[:, native:native + 1]).mean(axis=0) for m, a in arrays.items()}))
    result["fixed_action_probe"] = {
        str(action): {metric: interval(action_contexts, lambda c, k=metric, i=i: c["effects"][k][i])
                      for metric in METRICS} for i, action in enumerate(audit.joint.ACTIONS)}
    result["by_window_signal"] = {
        str(target): {metric: {stat: interval(
            [c for c in contexts if c["target_time_s"] == target],
            lambda c, m=metric, s=stat: c["signal_noise"][m][s])
            for stat in ("single_future_noise_variance", "debiased_signal_squared", "half_sign_agreement")}
            for metric in ("passenger_person_s", "cv_squared")} for target in audit.labels.TARGETS}
    result.update(stage="posthoc_development_diagnostic", rollouts_added=0, main_policy_changed=False)
    output = args.out_root / "focused_summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V68_FOCUSED_AUDIT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
