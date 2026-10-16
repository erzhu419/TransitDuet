"""Paired held-query attribution for cached equal-size prefix-deletion models."""

import numpy as np

from freq_hrl.domains.transit.native_shared_risk import state_weights


def conditions(train, held, groups):
    lookup = {}
    for row in train:
        key = row["root"], row["scene_seed"]
        condition = row["root"], row["scenario"]
        if key in lookup and lookup[key] != condition: raise ValueError("Inconsistent prefix condition")
        lookup[key] = condition
    selected = [lookup[tuple(g)] for g in groups]
    root = np.array([[r == h["root"] for h in held] for r, _ in selected])
    regime = np.array([[s == h["scenario"] for h in held] for _, s in selected])
    matched = root&regime
    support = np.array([sum(c == (r["root"], r["scenario"]) for c in lookup.values()) for r in held])
    if len(groups) != len(lookup) or set(map(tuple, groups)) != set(lookup) or not np.array_equal(matched.sum(axis=0), support):
        raise ValueError("Incomplete prefix-removal support")
    return {"all": np.ones_like(matched), "matched": matched, "other": ~matched,
        "same_root_other_regime": root&~regime, "other_root_same_regime": ~root&regime,
        "other_root_other_regime": ~root&~regime}, support


def paired_stats(full, deleted, target, eligible, masks):
    """Average single-model losses per query, not loss of averaged predictions."""
    w = state_weights(eligible)
    valid = w.sum(axis=1) > 0
    f = np.sum(w*(full-target)**2, axis=1)
    p = np.sum(w[None]*(deleted-target[None])**2, axis=2)
    results = {}
    for name, mask in masks.items():
        count = mask.sum(axis=0)
        ids = valid&(count > 0)
        if not ids.any(): continue
        mean = (p*mask).sum(axis=0)/np.maximum(count, 1)
        results[name] = {"state_exposures": int(ids.sum()),
            "prefix_state_exposures": int(mask[:, ids].sum()),
            "full_squared_error_sum": float(f[ids].sum()), "prefix54_squared_error_sum": float(mean[ids].sum())}
    matched = (p*masks["matched"]).sum(axis=0)/masks["matched"].sum(axis=0)
    other = (p*masks["other"]).sum(axis=0)/masks["other"].sum(axis=0)
    gap = (p*masks["all"]).sum(axis=0)/masks["all"].sum(axis=0)-f
    share = masks["matched"].sum(axis=0)/masks["all"].sum(axis=0)
    error = float(np.max(np.abs((gap-(share*(matched-f)+(1-share)*(other-f)))[valid]), initial=0.))
    if error > 1e-12: raise ValueError("Held-removal mixture identity failed")
    return results, error


def summarize(rows):
    regions = sorted({k for r in rows for k in r})
    out = {}
    for name in regions:
        rs = [r[name] for r in rows if name in r]
        n = sum(r["state_exposures"] for r in rs)
        full = sum(r["full_squared_error_sum"] for r in rs)/n
        deleted = sum(r["prefix54_squared_error_sum"] for r in rs)/n
        out[name] = {"state_exposures": n, "prefix_state_exposures": sum(r["prefix_state_exposures"] for r in rs),
            "full60_MSE": full, "prefix54_MSE": deleted, "prefix54_minus_full60_MSE": deleted-full}
    out["matched_minus_other_MSE"] = out["matched"]["prefix54_MSE"]-out["other"]["prefix54_MSE"]
    return out
