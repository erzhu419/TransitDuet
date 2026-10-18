"""Honest paired fleet-risk regression; proposal and service scores stay fixed."""

import numpy as np

from freq_hrl.domains.transit import native_abstention_transfer as base

ARMS = ("raw", "stage247", "zero_abstain", "bias_calibrated", "paired_risk", "matched_thin")
FIELDS = base.FIELDS
PAIRS = tuple((a, "raw") for a in ARMS[1:])+tuple(("paired_risk", a) for a in
    ("zero_abstain", "bias_calibrated", "matched_thin", "stage247"))
FEATURE_NAMES = ("clock_hours", "historical_excess", "active_headroom", "same_active", "other_active",
    "same_queue", "other_queue", "same_load", "same_remaining", "lf_forecast",
    "paired_mean_gap", "paired_first_gap", "paired_gap_std", "paired_nominal_shift",
    "paired_prefix_fleet_forecast", "paired_full_fleet_forecast",
    "P_fleet_mean", "P_service_mean", "P_wait_mean", "S_fleet_mean", "S_fleet_se",
    "S_positive_fraction", "S_negative_fraction", "S_all_zero", "S_service_mean", "S_wait_mean")


def features(rows, raw, proposal, first, screen):
    states, geometry = [], []
    for row in rows:
        occupancy, direction = row["occupancy_context"], row["direction"]
        buses = occupancy["active_buses"]
        same = [b for b in buses if b["direction"] == direction]
        peak = occupancy["historical_peak"]
        states.append([row["time_s"]/3600., max(peak-12, 0)/12., (peak-len(buses))/12.,
            len(same)/12., (len(buses)-len(same))/12., row["context"]["queues"][str(direction)]/50.,
            row["context"]["queues"][str(not direction)]/50.,
            float(np.mean([b["load_fraction"] for b in same])) if same else 0.,
            float(np.mean([b["predicted_remaining_s"] for b in same]))/1800. if same else 0.,
            row["native_state_v2"][12]])
        actions = []
        for c in row["candidates"]:
            planned = np.asarray(c["commitment"]["prefix_planned_s"])
            gaps = np.diff(planned)/360.
            actions.append([gaps.mean(), gaps[0], gaps.std(),
                ((planned-np.asarray(row["nominal_s"][:len(planned)]))/360.).mean(),
                c["commitment"]["prefix_fleet_cost"], c["fleet_cost"]])
        geometry.append(actions)
    geometry = np.asarray(geometry)
    p, s = (base.paired_values(v, proposal, raw) for v in (first, screen))
    fm = s[..., 1]
    statistics = np.stack((p[..., 1].mean(axis=2), p[..., 2].mean(axis=2), p[..., 3].mean(axis=2),
        fm.mean(axis=2), fm.std(axis=2, ddof=1)/np.sqrt(fm.shape[2]), (fm > 0).mean(axis=2),
        (fm < 0).mean(axis=2), np.all(fm == 0, axis=2), s[..., 2].mean(axis=2), s[..., 3].mean(axis=2)), axis=-1)
    paired_geometry = geometry[np.arange(len(rows))[None], proposal]-geometry[np.arange(len(rows))[None], raw]
    x = np.concatenate((np.broadcast_to(states, (*raw.shape, 10)), paired_geometry, statistics), axis=-1)
    if x.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("Incomplete or nonfinite causal paired-risk features")
    return x


def fit(x, targets, changed):
    """One fixed ridge solve. Duplicate profiles are weights, never extra seeds."""
    if not np.any(changed):
        raise ValueError("No changed training proposals for paired-risk fit")
    a, y = x[changed], targets[changed]
    center, scale = a.mean(axis=0), a.std(axis=0)
    scale[scale < 1e-8] = 1.
    z = np.c_[np.ones(len(a)), (a-center)/scale]
    gram, rhs = z.T@z/len(z), z.T@y/len(z)
    penalty = np.diag(np.r_[0., np.ones(a.shape[1])])
    beta = np.linalg.solve(gram+penalty, rhs)
    error = float(np.max(abs((gram+penalty)@beta-rhs)))
    if error > 1e-10 or not np.isfinite(beta).all():
        raise ValueError("Paired-risk ridge solve failed")
    return {"center": center.tolist(), "scale": scale.tolist(), "coefficients": beta.tolist(),
        "ridge": 1., "features": list(FEATURE_NAMES), "normal_equation_error": error,
        "profile_state_examples": int(changed.sum())}


def predict(model, x):
    return np.asarray(model["coefficients"])[0]+((x-model["center"])/model["scale"])@np.asarray(model["coefficients"])[1:]


def calibrate(scores, targets, eligible, cube):
    """Prefix-clustered mean-bias adjustment, not a conditional risk certificate."""
    numerator = ((targets-scores)*eligible).mean(axis=0)[cube].mean(axis=2).reshape(-1)
    denominator = eligible.mean(axis=0)[cube].mean(axis=2).reshape(-1)
    if denominator.sum() == 0:
        raise ValueError("No eligible independent calibration proposals")
    bias = float(numerator.sum()/denominator.sum())
    # Equal root/regime cohort sizes; preserve the fixed-stratum sampling design.
    g, r, _ = cube.shape
    residual = (numerator-bias*denominator).reshape(g, r)
    se = float(np.sqrt(residual.var(axis=1, ddof=1).sum()/(g*g*r))/denominator.mean())
    return {"bias": bias, "prefix_cluster_se": se,
        "upward_adjustment": max(0., bias+1.6448536269514722*se),
        "prefixes": g*r, "eligible_profile_state_examples": int(eligible.sum()),
        "interpretation": "one_sided_mean_score_adjustment_not_actionwise_coverage_or_safety_guarantee"}


def seal(raw, old, proposal, screen, learned_score, bias_score, cube):
    _, keep = base.screen(proposal, raw, screen)
    learned_keep, bias_keep = keep & (learned_score <= 0), keep & (bias_score <= 0)
    probability, counts = base.thin_probabilities(cube, keep, learned_keep)
    choices = np.stack((raw, old, np.where(keep, proposal, raw),
        np.where(bias_keep, proposal, raw), np.where(learned_keep, proposal, raw), raw), axis=1)
    return {"arms": list(ARMS), "choices": choices.tolist(), "proposal": proposal.tolist(),
        "matched_thin_probabilities": probability.tolist(), "matching": counts}


def action_unions(sealed, originals):
    choices, proposal = np.asarray(sealed["choices"]), np.asarray(sealed["proposal"])
    probability = np.asarray(sealed["matched_thin_probabilities"])
    unions = []
    for i, original in enumerate(originals):
        union = set()
        for p in range(len(choices)):
            actions = set(map(int, choices[p, :-1, i]))
            if probability[p, i] > 0: actions.add(int(proposal[p, i]))
            if len(actions) > 1: union.update(actions)
        unions.append(sorted(union | {int(original)}) if union else [])
    return unions


def effects(sealed, panels):
    choices = np.asarray(sealed["choices"])
    result = np.stack([base.paired_values(panels, choices[:, a], choices[:, 0]).mean(axis=0)
        for a in range(len(ARMS))], axis=2)
    result[:, :, -1] = (base.paired_values(panels, np.asarray(sealed["proposal"]), choices[:, 0])
        *np.asarray(sealed["matched_thin_probabilities"])[:, :, None, None]).mean(axis=0)
    return result


def summarize(sealed, panels, cube):
    actual = effects(sealed, panels)
    block, rng = actual[cube], np.random.default_rng(296)
    g, r, j, d = block.shape[:4]
    means = block.mean(axis=(2, 3))
    centered = block-block.mean(axis=3, keepdims=True)
    covariance = np.einsum("grjdvf,grjdwf->vwf", centered, centered)/(j*j*d*(d-1)*(g*r)**2)
    samples = sum(means[i, rng.choice(r, (5000, r), replace=True)].mean(axis=1)/g for i in range(g))
    comparisons = {}
    for after, before in PAIRS:
        a, b = ARMS.index(after), ARMS.index(before)
        point = (means[:, :, a]-means[:, :, b]).mean(axis=(0, 1))
        se = np.sqrt(np.maximum(covariance[a, a]+covariance[b, b]-2*covariance[a, b], 0.))
        ci = np.quantile(samples[:, a]-samples[:, b], [.025, .975], axis=0)
        mc = np.stack((point-1.959963984540054*se, point+1.959963984540054*se))
        comparisons[after+"_vs_"+before] = {"mean": dict(zip(FIELDS, map(float, point))),
            "prefix_CI95": {f: list(map(float, ci[:, k])) for k, f in enumerate(FIELDS)},
            "conditional_MC_CI95": {f: list(map(float, mc[:, k])) for k, f in enumerate(FIELDS)}}
    halves = {name: {arm: dict(zip(FIELDS, map(float, actual[:, ds, v].mean(axis=(0, 1)))))
        for v, arm in enumerate(ARMS)} for name, ds in (("first16", slice(0, 16)), ("last16", slice(16, 32)))}
    def passed(pair, fields):
        return all(comparisons[pair][interval][f][1] < 0 for interval in ("prefix_CI95", "conditional_MC_CI95") for f in fields)
    retention = {}
    for name, values in {"all32": {a: dict(zip(FIELDS, map(float, means[:, :, v].mean(axis=(0, 1)))))
            for v, a in enumerate(ARMS)}, **halves}.items():
        denominator = values["zero_abstain"]["wait_min"]
        retention[name] = values["paired_risk"]["wait_min"]/denominator if denominator < 0 else None
    both_halves = all(halves[h]["paired_risk"][f] < 0 for h in halves for f in ("cost", "wait_min"))
    service = passed("paired_risk_vs_raw", ("service", "wait_min")) and all(v is not None and v >= .5 for v in retention.values())
    return {"comparisons": comparisons, "halves": halves, "wait_retention_vs_zero_abstain": retention,
        "service_retention_supported": service, "joint_transfer_supported": passed("paired_risk_vs_raw", ("cost", "wait_min")) and both_halves and service,
        "fleet_improvement_vs_zero_abstain_supported": passed("paired_risk_vs_zero_abstain", ("fleet",)),
        "cost_information_vs_thinning_supported": passed("paired_risk_vs_matched_thin", ("cost",))}
