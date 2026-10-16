"""Measure local and propagated effects of removing one fitting prefix."""

import numpy as np

from freq_hrl.domains.transit import native_old_key_response as response

FIELDS = response.FIELDS
REGIONS = ("excluded", "retained_same_root", "retained_other_root", "retained",
    "held_same_root", "held_other_root", "held")


def regions(states, scope, removed):
    states = np.asarray(states)
    same = states[:, 0] == removed[0]
    if scope == "train60":
        excluded = same & (states[:, 1] == removed[1])
        return {"excluded": excluded, "retained_same_root": same & ~excluded,
            "retained_other_root": ~same, "retained": ~excluded}
    return {"held_same_root": same, "held_other_root": ~same, "held": np.ones(len(states), bool)}


def function_gap(data, inner, full, mask):
    feasible = (data["raw_logits"] <= 0) & (np.arange(6)[None, :] != data["originals"][:, None])
    feasible &= mask[:, None]
    difference = inner["components"]-full["components"]
    values = {"cost": inner["cost"]-full["cost"], "service": difference[..., 0], "risk": difference[..., 1]}
    return {"feasible_action_cases": int(feasible.sum()),
        "squared_sums": {f: float(np.sum(v[feasible]**2)) for f, v in values.items()}}


def outcomes(values, inner, full, raw):
    mean = np.asarray(values).mean(axis=1)
    ix = np.arange(len(raw))
    def stats(ch):
        delta = mean[ix, ch]-mean[ix, raw]
        return np.column_stack((delta[:, (0, 3, 1, 2)],
            (mean[ix, ch, 3] > 0).astype(int)-(mean[ix, raw, 3] > 0).astype(int), np.asarray(ch) != raw))
    a, b = stats(inner), stats(full)
    n = len(raw)
    delta = a-b
    return {"cases": n, "disagreements": int(np.count_nonzero(np.asarray(inner) != full)),
        "inner": dict(zip(FIELDS, map(float, a.mean(axis=0)))),
        "full": dict(zip(FIELDS, map(float, b.mean(axis=0)))),
        "effect": dict(zip(FIELDS, map(float, delta.mean(axis=0)))),
        "absolute_state_effect": dict(zip(FIELDS[:4], map(float, np.abs(delta[:, :4]).mean(axis=0))))}


def summarize(pieces):
    if not pieces: return {"bank_cell_exposures": 0, "regions": {}}
    result = {}
    for region in REGIONS:
        xs = [p["regions"][region] for p in pieces if region in p["regions"]]
        if not xs: continue
        n = sum(x["whole"]["cases"] for x in xs)
        rms_n = sum(x["function_gap"]["feasible_action_cases"] for x in xs)
        def profile(xs, label):
            count = sum(x[label]["cases"] for x in xs)
            return {p: {f: sum(x[label]["cases"]*x[label][p][f] for x in xs)/count for f in FIELDS[:4]}
                for p in ("inner", "full", "effect")}
        result[region] = {"state_model_exposures": n, "prefix_model_exposures": len(xs),
            "action_disagreements": sum(x["whole"]["disagreements"] for x in xs),
            "action_disagreement_rate": sum(x["whole"]["disagreements"] for x in xs)/n,
            "prefixes_with_action_change": sum(x["whole"]["disagreements"] > 0 for x in xs),
            **profile(xs, "whole"),
            "absolute_state_effect": {f: sum(x["whole"]["cases"]*x["whole"]["absolute_state_effect"][f] for x in xs)/n for f in FIELDS[:4]},
            "absolute_prefix_mean_effect": {f: sum(x["whole"]["cases"]*abs(x["whole"]["effect"][f]) for x in xs)/n for f in FIELDS[:4]},
            "opposite_halves": [profile(xs, h) for h in ("first8", "last8")],
            "feasible_prediction_rms": {f: float(np.sqrt(sum(x["function_gap"]["squared_sums"][f] for x in xs)/rms_n)) if rms_n else None
                for f in ("cost", "service", "risk")}}
    cells = {(tuple(p["cell"]), p["bank"]) for p in pieces}
    return {"bank_cell_exposures": len(cells), "regions": result}
