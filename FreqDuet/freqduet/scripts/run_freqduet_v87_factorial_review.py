#!/usr/bin/env python3
"""Summarize V87 cells with an additive, order-symmetric actor decomposition."""

import argparse
import json
from pathlib import Path

import numpy as np

import run_freqduet_v87_logit_factorial as experiment


def attribution(values):
    e = experiment.factorial(values)
    response = .5 * (e["late_feedback_at_early_mean"] + e["late_feedback_at_late_mean"])
    mean = -.5 * (e["early_mean_at_early_feedback"] + e["early_mean_at_late_feedback"])
    total = -e["early_vs_late"]
    np.testing.assert_allclose(mean + response, total, atol=1e-7, rtol=0)
    return dict(late_minus_early=total, mean_preference=mean, state_response=response,
                forward_interaction=-e["interaction"])


def interval(values):
    result = experiment.pilot.event.labels.expanded.paired_interval(values, 87001)
    return {k: result[k] for k in ("n_demand_days", "mean", "ci95")}


def review(root):
    rows = [json.loads((root / "pairs" / f"p{i}.json").read_text()) for i in range(len(experiment.ITEMS))]
    original = json.loads((root / "summary/status.json").read_text())
    absolute = {label: {m: float(np.mean([r["results"][label][m] for r in rows])) for m in experiment.METRICS}
                for label in experiment.POLICIES}
    effects, by_seed = {}, {}
    for seed in (None, *experiment.pilot.SEEDS):
        group = rows if seed is None else [r for r in rows if r["seed"] == seed]
        result = {}
        for metric in experiment.METRICS:
            cells = [(r["day"], attribution({label: r["results"][label][metric] for label in experiment.POLICIES}))
                     for r in group]
            result[metric] = {name: interval([float(np.mean([v[name] for day, v in cells if day == d]))
                for d in experiment.EVAL_DAYS]) for name in cells[0][1]}
            old = original["effects"] if seed is None else original["by_seed"][str(seed)]
            np.testing.assert_allclose(result[metric]["late_minus_early"]["mean"],
                -old["early_vs_late"][metric]["mean"], atol=1e-7, rtol=0)
        if seed is None:
            effects = result
        else:
            by_seed[str(seed)] = result
    result = dict(status="v87_symmetric_review_complete", absolute=absolute, attribution=effects, by_seed=by_seed,
        interval_unit="16 demand-day means, averaging paired sampler repeats and fixed actors",
        interaction="forward interaction reported separately; already shared by the two symmetric contributions",
        actor_updates=0, main_policy_changed=False)
    experiment.pilot.write_json(root / "review/status.json", result)
    print("V87_ABSOLUTE " + json.dumps(absolute), flush=True)
    print("V87_SYMMETRIC " + json.dumps(effects), flush=True)
    experiment.pilot.write_json(root / "review_ready/done.json", dict(status=result["status"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    review(parser.parse_args().out_root)
