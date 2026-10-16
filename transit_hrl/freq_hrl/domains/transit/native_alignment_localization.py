"""Additive localization of frozen-head alignment and cross-future concentration."""

import numpy as np

from freq_hrl.domains.transit import native_risk_alignment as source

FUTURES = ("same", "opposite")
MODES = {"natural": ("excluded", "matched_held"), "paired": ("paired_excluded", "paired_held")}
TERMS = ("correction_squared", "alignment", "delta_MSE")
DIMENSIONS = {"prefix": ("root", "scene_seed"), "condition": ("root", "scenario"),
    "probe": ("probe_index",), "condition_probe": ("root", "scenario", "probe_index")}


def probe_records(ex_rows, he_rows, pred_ex, pred_he, raw_ex, raw_he, ex_eligible, he_eligible, ex_targets, he_targets):
    pe, ph, paired_weights, _ = source.common_pairs(ex_rows, he_rows, ex_eligible, he_eligible)
    ex_weights, he_weights = map(source.state_weights, (ex_eligible, he_eligible))
    ex_probe = np.asarray([r["probe_index"] for r in ex_rows]);he_probe = np.asarray([r["probe_index"] for r in he_rows])
    records, error = [], 0.
    for probe in sorted(set(ex_probe.tolist())):
        ei, hi, pi = np.flatnonzero(ex_probe == probe), np.flatnonzero(he_probe == probe), np.flatnonzero(ex_probe[pe] == probe)
        metrics = {}
        for future in FUTURES:
            ey, hy = ex_targets[future], he_targets[future]
            populations = ((pred_ex[ei], raw_ex[ei], ey[ei], ex_weights[ei], int(ex_eligible[ei].any(axis=1).sum())),
                (pred_he[hi], raw_he[hi], hy[hi], he_weights[hi], int(he_eligible[hi].any(axis=1).sum())),
                (pred_ex[pe[pi]], raw_ex[pe[pi]], ey[pe[pi]], paired_weights[pi], int(paired_weights[pi].sum() > 0)),
                (pred_he[ph[pi]], raw_he[ph[pi]], hy[ph[pi]], paired_weights[pi], int(paired_weights[pi].sum() > 0)))
            metrics[future] = {}
            for pop, args in zip(("excluded", "matched_held", "paired_excluded", "paired_held"), populations):
                metrics[future][pop], e = source.components(*args);error = max(error, e)
        records.append({"probe_index": probe, "metrics": metrics})
    return records, error


def aggregate(rows):
    output = {}
    for future in FUTURES:
        s = {pop: source.summarize([r["metrics"][future][pop] for r in rows]) for pair in MODES.values() for pop in pair}
        s.update({mode+"_gap": source.contrast(s[ex], s[he]) for mode, (ex, he) in MODES.items()})
        output[future] = s
    return output


def signal(row, totals, future, mode, term):
    ex, he = MODES[mode]
    return sum(sign*row["metrics"][future][pop][term]/totals[future][pop]["units"]
        for pop, sign in ((ex, 1), (he, -1)) if totals[future][pop]["units"])


def contributions(rows, fields, totals):
    bins = {}
    for row in rows: bins.setdefault(tuple(row[k] for k in fields), []).append(row)
    return [{"key": list(key), "contributions": {f: {mode: {term: sum(signal(r, totals, f, mode, term) for r in rs)
        for term in TERMS} for mode in MODES} for f in FUTURES}} for key, rs in sorted(bins.items())]


def stability(rows, totals, mode):
    buckets = {k: {"rows": 0, "opposite_alignment_contribution": 0.} for k in ("positive_both", "negative_both", "sign_flip", "zero_either")}
    ex, he = MODES[mode]
    for r in rows:
        if not (r["metrics"]["same"][ex]["units"] or r["metrics"]["same"][he]["units"]): continue
        a, b = (signal(r, totals, f, mode, "alignment") for f in FUTURES)
        kind = "positive_both" if a > 0 and b > 0 else "negative_both" if a < 0 and b < 0 else "sign_flip" if a*b < 0 else "zero_either"
        buckets[kind]["rows"] += 1;buckets[kind]["opposite_alignment_contribution"] += b
    n = sum(b["rows"] for k, b in buckets.items() if k != "zero_either")
    return {"buckets": buckets, "nonzero_both_rows": n,
        "same_sign_fraction": (buckets["positive_both"]["rows"]+buckets["negative_both"]["rows"])/n if n else None}


def concentration(rows, prefix_bins, totals, mode, ranking_future):
    evaluation_future = "opposite" if ranking_future == "same" else "same"
    ranked = sorted(prefix_bins, key=lambda b: (-b["contributions"][ranking_future][mode]["alignment"], b["key"]))
    total = totals[evaluation_future][mode+"_gap"]["alignment"]
    absolute = sum(abs(b["contributions"][evaluation_future][mode]["alignment"]) for b in prefix_bins)
    output = []
    for k in (1, 3, 5):
        selected = ranked[:k];keys = {tuple(b["key"]) for b in selected}
        value = sum(b["contributions"][evaluation_future][mode]["alignment"] for b in selected)
        remaining = aggregate([r for r in rows if (r["root"], r["scene_seed"]) not in keys])[evaluation_future]
        output.append({"top_k": k, "prefixes": [b["key"] for b in selected], "evaluation_future": evaluation_future,
            "selected_alignment_contribution": value, "share_of_net_gap": value/total if total else None,
            "share_of_absolute_mass": sum(abs(b["contributions"][evaluation_future][mode]["alignment"]) for b in selected)/absolute if absolute else None,
            "remaining_alignment_gap": remaining[mode+"_gap"]["alignment"],
            "remaining_units": {p: remaining[p]["units"] for p in MODES[mode]}})
    return output


def summarize(rows):
    totals = aggregate(rows)
    breakdown = {name: contributions(rows, fields, totals) for name, fields in DIMENSIONS.items()}
    error = max(abs(sum(b["contributions"][f][m][t] for b in bins)-totals[f][m+"_gap"][t])
        for bins in breakdown.values() for f in FUTURES for m in MODES for t in TERMS)
    if error > 1e-12: raise ValueError("Alignment localization is not additive")
    return {"totals": totals, "breakdown": breakdown,
        "stability": {m: stability(rows, totals, m) for m in MODES},
        "concentration": {m: {f: concentration(rows, breakdown["prefix"], totals, m, f) for f in FUTURES} for m in MODES},
        "partition_max_abs_error": error}
