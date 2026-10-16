"""Moment-pooled scale/cosine attribution of frozen correction alignment."""

from itertools import permutations
from math import prod, sqrt

from freq_hrl.domains.transit import native_alignment_localization as source

DOMINANT_PREFIXES = ((401, 914010401), (397, 913970003), (401, 914010303))
DOMINANT_PROBES = (8, 9)
FACTORS = ("population_mass", "correction_RMS", "raw_residual_RMS", "normalized_alignment")
PARTITION = ("dominant_both", "dominant_prefix_other_probe", "other_prefix_dominant_probe", "neither")


def subsets(rows):
    result = {k: [] for k in ("all", "dominant_prefix", "other_prefix", "dominant_probe", "other_probe", *PARTITION)}
    for r in rows:
        p, q = (r["root"], r["scene_seed"]) in DOMINANT_PREFIXES, r["probe_index"] in DOMINANT_PROBES
        result["all"].append(r);result["dominant_prefix" if p else "other_prefix"].append(r)
        result["dominant_probe" if q else "other_probe"].append(r)
        result[PARTITION[0 if p and q else 1 if p else 2 if q else 3]].append(r)
    return result


def population(summary, total_units):
    n = summary["units"]
    d = sqrt(summary["correction_squared"]) if n else None
    r = sqrt(summary["raw_MSE"]) if n else None
    rho = summary["alignment"]/(2*d*r) if n and d*r > 0 else None
    return {"units": n, "population_mass": n/total_units if total_units else 0.,
        "correction_RMS": d, "raw_residual_RMS": r, "normalized_alignment": rho,
        **{k: summary[k] for k in ("alignment", "correction_squared", "delta_MSE")}}


def product_effects(held, excluded):
    """Average the exact held-to-excluded replacement path over all 24 orders."""
    effects = dict.fromkeys(FACTORS, 0.)
    orders = list(permutations(FACTORS))
    for order in orders:
        current = {k: held[k] for k in FACTORS}
        for k in order:
            before = 2*prod(current.values());current[k] = excluded[k]
            effects[k] += (2*prod(current.values())-before)/len(orders)
    return effects


def compare(ex_summary, he_summary, ex_total, he_total):
    ex, he = population(ex_summary, ex_total), population(he_summary, he_total)
    def contribution(p, term): return p["population_mass"]*p[term] if p["units"] else 0.
    gaps = {term: contribution(ex, term)-contribution(he, term) for term in source.TERMS}
    effects, error = None, abs(gaps["delta_MSE"]-gaps["correction_squared"]-gaps["alignment"])
    if ex["normalized_alignment"] is not None and he["normalized_alignment"] is not None:
        effects = product_effects(he, ex)
        error = max(error, abs(sum(effects.values())-gaps["alignment"]))
        for p in (ex, he): error = max(error, abs(2*prod(p[k] for k in FACTORS)-contribution(p, "alignment")))
    if error > 1e-12: raise ValueError("Scale/alignment product identity failed")
    return {"excluded": ex, "held": he, "global_contribution": gaps, "effects": effects,
        "within_alignment_gap": ex["alignment"]-he["alignment"] if ex["units"] and he["units"] else None,
        "identity_max_abs_error": error}


def summarize(rows):
    groups = {k: source.aggregate(rs) for k, rs in subsets(rows).items()}
    totals = groups["all"];output, partition_error, factor_error = {}, 0., 0.
    for future in source.FUTURES:
        output[future] = {}
        for mode, (ex, he) in source.MODES.items():
            cases = {k: compare(s[future][ex], s[future][he], totals[future][ex]["units"], totals[future][he]["units"]) for k, s in groups.items()}
            output[future][mode] = cases;factor_error = max(factor_error, *(c["identity_max_abs_error"] for c in cases.values()))
            for partition in (PARTITION, ("dominant_prefix", "other_prefix"), ("dominant_probe", "other_probe")):
                for term in source.TERMS: partition_error = max(partition_error, abs(sum(cases[k]["global_contribution"][term] for k in partition)-cases["all"]["global_contribution"][term]))
    if partition_error > 1e-12: raise ValueError("Subgroup global contributions are not additive")
    return {"totals": totals, "comparisons": output, "partition_max_abs_error": partition_error,
        "factorization_max_abs_error": factor_error}
