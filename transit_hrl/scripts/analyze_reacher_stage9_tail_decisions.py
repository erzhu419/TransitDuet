"""Compare W and W+tail action placement on crossed cached development windows."""

import json
from pathlib import Path

import numpy as np

from freq_hrl.experiments.plan_validity_reliability import paired_mean_interval
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts.run_plan_validity_boundary_tail import load_cache


def top_at_equal_counts(scores, factual, clocks):
    actions = np.zeros(len(scores), dtype=bool)
    for clock in np.unique(clocks):
        group = np.flatnonzero(clocks == clock)
        count = int(factual[group].sum())
        actions[group[np.argsort(-scores[group], kind="stable")[:count]]] = True
    return actions


def analyze(rows, rule, horizon):
    rows = sorted(rows, key=lambda r: (r["seed"], r["opportunity_step"]))
    prefixes = np.array([r["seed"] for r in rows])
    steps = np.array([r["opportunity_step"] for r in rows])
    clocks = steps % 50
    factual = np.array([rule.score(tuple(r["feature_names"]), np.asarray(r["causal_features"]),
        clock / 25, (horizon - step) / horizon) >= rule.threshold for r, clock, step in zip(rows, clocks, steps)])
    W, G = (np.asarray([[[s[key] for s in w["suffix_draws"]] for w in r["windows"]] for r in rows])
            for key in ("W", "G"))
    if not np.isfinite(G).all() or not np.isfinite(W).all() or not np.array_equal(W, np.broadcast_to(W[:, :, :1], W.shape)):
        raise ValueError("cached W must stay fixed within each realized window")
    directions = []
    for fit, test in ((slice(0, 2), slice(2, 4)), (slice(2, 4), slice(0, 2))):
        ws, gs, held = [values[:, part].mean(axis=(1, 2)) for values, part in ((W, fit), (G, fit), (G, test))]
        w = top_at_equal_counts(ws, factual, clocks)
        g = top_at_equal_counts(gs, factual, clocks)
        comparisons = {}
        for key, a, b in (("G_rank_minus_W_rank", g, w), ("G_rank_minus_C0", g, factual),
                          ("W_rank_minus_C0", w, factual), ("G_sign_minus_W_sign", gs > 0, ws > 0)):
            delta = (a.astype(int) - b) * held
            by_prefix = [float(delta[prefixes == p].mean()) for p in np.unique(prefixes)]
            comparisons[key] = {**paired_mean_interval(by_prefix), "changed_checks": int((a != b).sum()),
                                "prefix_deltas": by_prefix}
        directions.append({"selection_windows": list(range(4))[fit], "evaluation_windows": list(range(4))[test],
                           "mean_absolute_held_G": float(np.abs(held).mean()), "comparisons": comparisons})
    return {"prefixes": np.unique(prefixes).tolist(), "checks": len(rows), "C0_now": int(factual.sum()),
            "clock_counts": {int(c): [int((clocks == c).sum()), int(factual[clocks == c].sum())] for c in np.unique(clocks)},
            "directions": directions}


def main():
    path = Path("protocols")
    source = json.loads((path / "plan_validity_boundary_tail_2026-11-11.json").read_text())
    variance = json.loads((path / "plan_validity_boundary_variance_2026-11-11.json").read_text())
    rules = json.loads(Path(variance["source_manifest"]).read_text())["cells"]
    cells = []
    for cell in source["cells"]:
        if cell["task"] != "reacher_tracking":
            continue
        root = cell["optimizer_seed"]
        ref, = [r for r in rules if (r["task"], r["optimizer_seed"]) == ("reacher_tracking", root)]
        frozen = json.loads((Path(ref["phase_cache_directory"]) / "frozen_rules.json").read_text())
        summary = analyze(load_cache(cell, source), TimingRule(**frozen["checks_clock"]), source["horizon"])
        cells.append({"optimizer_seed": root, **summary})
    print(json.dumps({"cells": cells, "new_fits": 0, "new_environment_steps": 0,
        "scope": "descriptive_same_development_prefixes_crossed_independent_windows_not_policy_or_oracle"}, separators=(",", ":")))


if __name__ == "__main__":
    main()
