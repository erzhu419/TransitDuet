"""Preregistered paired-prefix bootstrap for frozen conditioning predictors."""

import numpy as np

from .compute_budget import count_work


COMPARISONS = (("tracking_distance", "wide"), ("tracking_distance", "state_rms"),
    ("tracking_distance", "base"), ("tracking_distance", "zero"), ("state_rms", "wide"),
    ("state_rms", "base"), ("state_rms", "zero"), ("wide", "base"), ("wide", "zero"), ("base", "zero"))


def bootstrap_root(readout, cell, plan):
    names = [a + "_minus_" + b for a, b in COMPARISONS]
    for name in names:
        if [p["seed"] for p in readout["comparisons"][name]["prefix_reports"]] != cell["readout_prefix_seeds"]:
            raise ValueError("bootstrap requires the complete declared 16-prefix roster")
    if len(cell["readout_prefix_seeds"]) != 16:
        raise ValueError("bootstrap is not a partial-shard readout")
    values = np.asarray([[p["MSE_delta"] for p in readout["comparisons"][name]["prefix_reports"]] for name in names]).T
    rng = np.random.default_rng(cell["bootstrap_seed"])
    indices = rng.integers(0, 16, size=(plan["bootstrap_replicates"], 16))
    draws = values[indices].mean(axis=1)
    count_work("bootstrap_prefix_resamples", len(indices))
    bounds = np.quantile(draws, [.025, .95, .975], axis=0, method="linear")
    for j, name in enumerate(names):
        contrast = readout["comparisons"][name]
        contrast["prefix_bootstrap"] = {"nominal_two_sided_95": [float(bounds[0, j]), float(bounds[2, j])],
            "nominal_one_sided_95_upper": float(bounds[1, j])}
        contrast["prefix_halves"] = [{"MSE_delta": float(v.mean())} for v in (values[:8, j], values[8:, j])]
    passed = True
    for reference in plan["primary_comparators"]:
        contrast = readout["comparisons"]["tracking_distance_minus_" + reference]
        directional = all(p["MSE_delta"] < 0 for half in ("suffix_halves", "window_halves", "prefix_halves") for p in contrast[half])
        if contrast["prefix_bootstrap"]["nominal_one_sided_95_upper"] >= 0 or not directional:
            passed = False
    readout.update(prediction_confirmation_rule_passed=passed,
        bootstrap={"replicates": plan["bootstrap_replicates"], "seed": cell["bootstrap_seed"],
            "unit": "whole_prefix_mean_paired_loss", "paired_resamples_shared_across_comparisons": True,
            "scope": "nominal_each_comparison_given_frozen_models_not_simultaneous_or_training_seed_coverage"})
    return readout
