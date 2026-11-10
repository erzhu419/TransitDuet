"""A-only cut selection on new prefixes for unchanged full-model scores."""

import numpy as np

from .plan_validity_decision_calibration import _B_gain, calibrate_cut
from .plan_validity_phase_support import predict
from .plan_validity_window_coverage import values


def validate_rows(rows, *, seeds, probes, a, b):
    roster = [(r["seed"], r["opportunity_step"]) for r in rows]
    if (len(rows) != len(seeds) * probes or len(set(roster)) != len(rows)
        or set(r["seed"] for r in rows) != set(seeds)
        or any(sum(r["seed"] == s for r in rows) != probes for s in seeds)):
        raise ValueError("independent-prefix roster differs")
    for row in rows:
        ids = [r["future_seed"] for block in ("A", "B") for r in row[block + "_replicas"]]
        if len(row["A_replicas"]) != a or len(row["B_replicas"]) != b or len(set(ids)) != a + b:
            raise ValueError("independent-future counts or separation differ")


def select_cuts(rows, rules):
    w, g = (values(rows, k, "A").mean(axis=1) for k in ("W", "G"))
    return {head: calibrate_cut(predict(rule, rows), w, g, rule.threshold) for head, rule in rules.items()}


def cut_readout(rows, rules, selected, previous, *, fresh_prefix):
    groups = np.asarray([r["seed"] for r in rows])
    factual = np.asarray([r["factual_stop"] for r in rows], dtype=int)
    effects = {k: values(rows, k, "B") for k in ("W", "G")}
    if fresh_prefix:
        effects = {k: np.column_stack((values(rows, k, "A"), v)) for k, v in effects.items()}
    half = effects["W"].shape[1] // 2
    result = {}
    for head, rule in rules.items():
        scores = predict(rule, rows)
        actions = {name: (scores >= cut).astype(int) for name, cut in (
            ("quantile", rule.threshold), ("previous_full_calibration", previous[head]["cut"]),
            ("independent_calibration", selected[head]["cut"]))}
        changes = {"independent_minus_" + name: actions["independent_calibration"] - action
            for name, action in {**{k: v for k, v in actions.items() if k != "independent_calibration"}, "C0": factual}.items()}
        panels = {}
        for block, subset in (("first_half", slice(0, half)), ("second_half", slice(half, 2 * half)), ("all", slice(None))):
            panels[block] = {}
            for name, change in changes.items():
                panels[block][name] = {}
                for target, samples in effects.items():
                    gain = _B_gain(change, samples[:, subset], groups)
                    gain["prefix_interval"] = gain.pop("training_prefix_interval")
                    panels[block][name][target] = gain
        result[head] = {"choose_now": {**{k: int(v.sum()) for k, v in actions.items()}, "C0": int(factual.sum())},
            "changed_actions": {k: int(np.count_nonzero(v)) for k, v in changes.items()},
            "future_count": effects["W"].shape[1], "panels": panels}
    return result
