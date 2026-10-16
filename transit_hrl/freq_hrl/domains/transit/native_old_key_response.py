"""Frozen-key response paths under validation, common futures and held prefixes."""

import numpy as np

from freq_hrl.domains.transit.native_selection_attribution import FIELDS

PATHS = ("validation", "inner_opposite", "full_seen", "full_held")
STEPS = ("validation_to_inner", "inner_to_full_seen", "seen_to_held")
PHYSICAL = FIELDS[:4]


def action_stats(values, chosen, raw, roots):
    mean = np.asarray(values).mean(axis=1)
    chosen, raw, roots = (np.asarray(x) for x in (chosen, raw, roots))
    ix = np.arange(len(raw))
    delta = mean[ix, chosen]-mean[ix, raw]
    values = np.column_stack((delta[:, (0, 3, 1, 2)],
        (mean[ix, chosen, 3] > 0).astype(int)-(mean[ix, raw, 3] > 0).astype(int), chosen != raw))
    def totals(mask):
        return {"cases": int(mask.sum()), "sums": dict(zip(FIELDS, map(float, values[mask].sum(axis=0))))}
    return {"all": totals(np.ones(len(raw), bool)),
        "by_root": {str(r): totals(roots == r) for r in sorted(set(roots))}}


def combine(pieces):
    def mean(xs):
        n = sum(x["cases"] for x in xs)
        return {"cases": n, **{f: sum(x["sums"][f] for x in xs)/n for f in FIELDS}}
    roots = sorted({r for p in pieces for r in p["by_root"]})
    return {"all": mean([p["all"] for p in pieces]),
        "by_root": {r: mean([p["by_root"][r] for p in pieces if r in p["by_root"]]) for r in roots}}


def violations(stats):
    return {f"{r}:{f}" for r, x in stats["by_root"].items() for f in ("delta_cost", "delta_wait_min", "delta_unsafe")
        if x[f] > (0. if f == "delta_unsafe" else 1e-12)}


def diagnose(paths, halves):
    a, b, c, d = (paths[p] for p in PATHS)
    steps = {s: {f: after["all"][f]-before["all"][f] for f in FIELDS}
        for s, before, after in zip(STEPS, (a, b, c), (b, c, d))}
    residual = max(abs(sum(steps[s][f] for s in STEPS)-(d["all"][f]-a["all"][f])) for f in FIELDS)
    if residual > 1e-12: raise ValueError("Old-key response path does not telescope")
    av = violations(a)
    h0, h1 = (violations(h["inner_opposite"]) for h in halves)
    flags = {"pooled_validation_point_harm": bool(av),
        "same_validation_violation_in_both_opposite_halves": bool(av & h0 & h1),
        "validation_violations_absent_in_both_opposite_halves": bool(av) and not (av & (h0 | h1)),
        "inner_wait_benefit_full_seen_wait_harm": b["all"]["delta_wait_min"] < -1e-12 and c["all"]["delta_wait_min"] > 1e-12,
        "inner_wait_benefit_full_seen_wait_harm_both_halves": all(h["inner_opposite"]["all"]["delta_wait_min"] < -1e-12
            and h["full_seen"]["all"]["delta_wait_min"] > 1e-12 for h in halves),
        "seen_wait_harm_held_wait_benefit": c["all"]["delta_wait_min"] > 1e-12 and d["all"]["delta_wait_min"] < -1e-12,
        "seen_wait_harm_held_wait_benefit_both_halves": all(h["full_seen"]["all"]["delta_wait_min"] > 1e-12
            and h["full_held"]["all"]["delta_wait_min"] < -1e-12 for h in halves),
        "inner_no_action_full_seen_changes": a["all"]["changed_rate"] == 0 and c["all"]["changed_rate"] > 0}
    return {"steps": steps, "flags": flags, "decomposition_max_abs_difference": residual}


def summarize(pieces, denominator):
    if not pieces: return {"bank_cell_exposures": 0, "conditional": None,
        "full_denominator_contributions": {p: dict.fromkeys(PHYSICAL, 0.) for p in ("full_seen", "full_held")}}
    n = len(pieces)
    mean = lambda xs: {f: sum(x[f] for x in xs)/len(xs) for f in xs[0]}
    paths = {p: mean([{f: x["paths"][p]["all"][f] for f in FIELDS} for x in pieces]) for p in PATHS}
    return {"bank_cell_exposures": n, "conditional": {"path_means": paths,
        "opposite_halves": [{p: mean([{f: x["halves"][h][p]["all"][f] for f in PHYSICAL} for x in pieces])
            for p in PATHS[1:]} for h in (0, 1)],
        "step_means": {s: mean([{f: x["diagnosis"]["steps"][s][f] for f in PHYSICAL} for x in pieces]) for s in STEPS},
        "diagnostic_counts": {f: sum(x["diagnosis"]["flags"][f] for x in pieces) for f in pieces[0]["diagnosis"]["flags"]},
        "inner_full_action_disagreement": sum(x["action_disagreement"] for x in pieces)/n},
        "full_denominator_contributions": {p: {f: paths[p][f]*n/denominator for f in PHYSICAL} for p in ("full_seen", "full_held")}}
