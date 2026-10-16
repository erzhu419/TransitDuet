#!/usr/bin/env python3
"""Posthoc V69 signal/noise and held-future constraint diagnostics; no rollouts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v69_event_coupling as pilot


def signal_ratio(contexts, metric):
    days = pilot.labels.DEMAND_SEEDS
    signal, noise = [np.asarray([
        np.mean([c["signal_noise"]["event"][metric][stat]
                 for c in contexts if c["scenario_seed"] == day]) for day in days])
        for stat in ("debiased_signal_squared", "single_future_noise_variance")]
    indices = np.random.default_rng(69070).integers(len(days), size=(10000, len(days)))
    boot_ratio = signal[indices].mean(axis=1) / noise[indices].mean(axis=1)
    ratio = float(signal.mean() / noise.mean())
    ci = np.quantile(boot_ratio, [.025, .975])
    positive = np.maximum(signal, 0.)
    omitted_means = (signal.sum() - signal) / (len(days) - 1)
    return dict(single_future_signal_to_noise=ratio, single_future_ci95=ci.tolist(),
                eight_future_mean_signal_to_noise=8 * ratio, eight_future_mean_ci95=(8 * ci).tolist(),
                positive_signal_days=int((signal > 0).sum()),
                max_day_share_of_positive_signal=float(positive.max() / positive.sum()),
                leave_one_day_out_signal_range=[float(omitted_means.min()), float(omitted_means.max())],
                by_day=[dict(scenario_seed=day, corrected_signal_squared=float(s),
                             single_future_noise_variance=float(n)) for day, s, n in zip(days, signal, noise)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                    str(Path(__file__).resolve().parents[1] / "tests"),
                    "-p", "test_v69_signal_audit.py", "-v"], check=True)
    contexts = json.loads((args.out_root / "groups/context_audit.json").read_text())
    halves = []
    for context in contexts:
        groups = [json.loads(pilot.group_path(args.out_root, pilot.ITEMS.index((context["context_index"], f))).read_text())
                  for f in pilot.repeated.FUTURES]
        native = pilot.joint.ACTIONS.index(groups[0]["native_action_s"])
        arrays = {metric: np.asarray([[r[metric] for r in group["rows"]] for group in groups], dtype=float)
                  for metric in pilot.repeated.RESPONSE_METRICS}
        for train, evaluate in ((np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))):
            halves.append(pilot.repeated.evaluate_selection(arrays, native, train, evaluate))
    changed = [h for h in halves if h["changed"]]
    daily_effects = [dict(scenario_seed=day,
                         passenger_person_s=float(np.mean([c["cross_future"]["event"]["passenger_person_s"]
                                                           for c in contexts if c["scenario_seed"] == day])),
                         readiness_delay_s=float(np.mean([c["cross_future"]["event"]["readiness_delay_s"]
                                                          for c in contexts if c["scenario_seed"] == day])))
                    for day in pilot.labels.DEMAND_SEEDS]
    effects = np.asarray([d["passenger_person_s"] for d in daily_effects])
    improvement = np.maximum(-effects, 0.)
    omitted_effects = (effects.sum() - effects) / (len(effects) - 1)
    result = dict(stage="posthoc_development_signal_audit", contexts=len(contexts), demand_day_clusters=8,
                  signal_noise={m: signal_ratio(contexts, m) for m in ("passenger_person_s", "cv_squared")},
                  held_future_selection=dict(
                      half_decisions=len(halves), changed=len(changed),
                      changed_passenger_improved=sum(h["passenger_person_s"] < 0 for h in changed),
                      changed_passenger_worsened=sum(h["passenger_person_s"] > 0 for h in changed),
                      changed_passenger_tied=sum(h["passenger_person_s"] == 0 for h in changed),
                      changed_joint_nonworse=sum(h["passenger_person_s"] < 0 and
                                                all(h[m] <= 0 for m in pilot.repeated.CONSTRAINT_METRICS) for h in changed),
                      constraint_violations={m: sum(h[m] > 0 for h in changed) for m in pilot.repeated.CONSTRAINT_METRICS},
                      improved_days=int((effects < 0).sum()),
                      max_day_share_of_gross_improvement=float(improvement.max() / improvement.sum()),
                      leave_one_day_out_passenger_range=[float(omitted_effects.min()), float(omitted_effects.max())],
                      by_day=daily_effects),
                  rollouts_added=0, model_updates=0, main_policy_changed=False)
    output = args.out_root / "focused_summary"
    output.mkdir(exist_ok=True)
    (output / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print("V69_SIGNAL_AUDIT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
