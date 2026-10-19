"""Global retain/fallback calibration with an explicit service-retention budget."""

from statistics import NormalDist

import numpy as np

from freq_hrl.domains.transit import native_abstention_transfer as base

FIELDS = base.FIELDS
QUANTILES = tuple(np.arange(1, 17)/16.)


def family(fleet, service, eligible):
    """Define both one-dimensional families on fit-role scores, before targets."""
    if not np.any(eligible):
        raise ValueError("No eligible fit proposals for a cutoff family")
    result = [{"family": "base", "threshold": None}]
    for name, score in (("risk", fleet), ("joint", fleet+service)):
        thresholds = np.unique(np.r_[0., np.quantile(score[eligible], QUANTILES)])
        result.extend({"family": name, "threshold": float(t)} for t in thresholds)
    return result


def choices(raw, proposal, eligible, fleet, service, candidates):
    scores = {"risk": fleet, "joint": fleet+service}
    return np.stack([np.where(eligible if c["family"] == "base" else
        eligible & (scores[c["family"]] <= c["threshold"]), proposal, raw)
        for c in candidates], axis=1)


def values(targets, selected, raw):
    # Profiles share futures and are averaged before either uncertainty estimate.
    return np.stack([base.paired_values(targets, selected[:, i], raw).mean(axis=0)
        for i in range(selected.shape[1])], axis=2)


def calibrate(actual, cube, candidates, changed_counts):
    """Development policy selection, never an untouched-control validation."""
    block = actual[cube]
    g, r, j, d, n, _ = block.shape
    means = block.mean(axis=(2, 3))
    point = means.mean(axis=(0, 1))
    rng = np.random.default_rng(297)
    samples = sum(means[k, rng.choice(r, (5000, r), replace=True)].mean(axis=1)/g
        for k in range(g))
    prefix_ci = np.quantile(samples, [.025, .975], axis=0)
    mc_se = np.sqrt(actual.var(axis=1, ddof=1).sum(axis=0)/(d*(g*r*j)**2))
    z = NormalDist().inv_cdf(.975)
    mc_ci = np.stack((point-z*mc_se, point+z*mc_se))
    halves = np.stack([actual[:, a:b].mean(axis=(0, 1)) for a, b in ((0, d//2), (d//2, d))])
    wait, cost, service = (FIELDS.index(f) for f in ("wait_min", "cost", "service"))

    # G_candidate >= .5 G_base iff W_candidate - .5 W_base <= 0.
    difference = means[..., wait]-.5*means[..., [0], wait]
    estimate = point[:, wait]-.5*point[0, wait]
    prefix_se = np.sqrt(difference.var(axis=1, ddof=1).sum(axis=0)/(g*g*r))
    sampled_difference = samples[..., wait]-.5*samples[:, [0], wait]
    standardized = np.divide(sampled_difference-estimate, prefix_se,
        out=np.zeros_like(sampled_difference), where=prefix_se > 0)
    critical = max(0., float(np.quantile(standardized.max(axis=1), .975)))
    prefix_upper = estimate+critical*prefix_se
    paired_wait = actual[..., wait]-.5*actual[:, :, [0], wait]
    paired_mc_se = np.sqrt(paired_wait.var(axis=1, ddof=1).sum(axis=0)/(d*(g*r*j)**2))
    mc_upper = estimate+NormalDist().inv_cdf(1-.025/n)*paired_mc_se
    half_difference = halves[..., wait]-.5*halves[:, [0], wait]
    baseline_supported = bool(prefix_ci[1, 0, wait] < 0 and mc_ci[1, 0, wait] < 0
        and np.all(halves[:, 0, wait] < 0))
    retention_ok = (prefix_upper <= 0) & (mc_upper <= 0) & np.all(half_difference <= 0, axis=0)
    admissible = baseline_supported & retention_ok & (point[:, cost] < 0) & (point[:, service] < 0)
    admissible &= np.all(halves[:, :, cost] < 0, axis=0)
    objective = np.maximum(prefix_ci[1, :, cost], mc_ci[1, :, cost])

    statistics = []
    for i, candidate in enumerate(candidates):
        statistics.append({**candidate, "index": i, "changed_profile_states": int(changed_counts[i]),
            "mean": dict(zip(FIELDS, map(float, point[i]))),
            "prefix_CI95": {f: prefix_ci[:, i, k].tolist() for k, f in enumerate(FIELDS)},
            "conditional_MC_CI95": {f: mc_ci[:, i, k].tolist() for k, f in enumerate(FIELDS)},
            "halves": {h: dict(zip(FIELDS, map(float, halves[k, i]))) for k, h in enumerate(("first8", "last8"))},
            "waiting_gain_retention": float(point[i, wait]/point[0, wait]) if point[0, wait] < 0 else None,
            "retention_difference_prefix_simultaneous_upper": float(prefix_upper[i]),
            "retention_difference_MC_bonferroni_upper": float(mc_upper[i]),
            "retention_constraint_passed": bool(retention_ok[i]),
            "selection_eligible": bool(admissible[i]), "cost_upper_objective": float(objective[i])})
    selected = {}
    for name in ("risk", "joint"):
        indices = [i for i, c in enumerate(candidates) if c["family"] in ("base", name) and admissible[i]]
        selected[name] = statistics[min(indices, key=lambda i: (objective[i], point[i, cost], i))] if indices else None
    joint = selected["joint"]
    admitted = bool(joint is not None and 0 < joint["changed_profile_states"] < changed_counts[0]
        and joint["mean"]["cost"] < point[0, cost])
    return {"baseline_wait_supported_on_development_calibration": baseline_supported,
        "prefix_retention_max_t_critical": critical, "selection": selected,
        "base": statistics[0], "candidate_statistics": statistics,
        "fresh_confirmation_admitted": admitted,
        "interpretation": "development_selection_only; familywise_waiting_checks_not_a_policy_safety_certificate"}
