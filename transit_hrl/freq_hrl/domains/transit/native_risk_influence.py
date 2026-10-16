"""Cached fleet-event scatter and single-future influence at frozen KKT geometry."""

import numpy as np

from freq_hrl.domains.transit import native_coupled_noise as coupled

TERMS = coupled.TERMS
PARTS = ("within_event", "within_none", "occurrence")
POPULATIONS = ("all", "excluded", "retained")


def fleet_events(paired, eligible):
    lo, hi = coupled.calibration.shrink.pair.LO, coupled.calibration.shrink.pair.HI
    gap = paired[:, :, hi, 1]-paired[:, :, lo, 1]
    relevant = eligible[:, lo]&eligible[:, hi]
    return np.any((np.abs(gap)>1e-12)&relevant[:, None], axis=-1)


def risk_scores(z, c, paired, eligible, prior_heads, full_heads, prefix_heads, indices, model, radius, level, scale):
    target, precision, rho = (np.asarray(c[k]) for k in ("target", "precision", "reliability"))
    difference, penalty = prior_heads[0]-prior_heads[1], c["penalties"][level]
    factory = lambda zz, y, w, ok: coupled.calibration.objectives(zz, y, w, ok, difference, penalty)[model]
    lf, hf, _, _ = factory(z, target, precision, eligible)
    lp, hp, _, _ = factory(z[indices], target[indices], precision[indices], eligible[indices])
    full, prefix = full_heads.sum(axis=0), prefix_heads.sum(axis=0)
    mf, mp = coupled.curvature.multiplier(lf, full, radius), coupled.curvature.multiplier(lp, prefix, radius)
    tf = coupled.endpoint_operator(hf(full), full, mf, radius)["tangent"]
    tp = coupled.endpoint_operator(hp(prefix), prefix, mp, radius)["tangent"]
    draws = paired*rho[:, None]/scale
    sf = coupled.noise.state_scores(z, draws, precision, eligible, model, penalty)[:, :, 1:2]
    sp = np.zeros_like(sf)
    sp[indices] = coupled.noise.state_scores(z[indices], draws[indices], precision[indices], eligible[indices], model, penalty)[:, :, 1:2]
    return {t: s[:, :, 0] for t, s in coupled.response_scores(sf, sp, tf, tp).items()}


def scatter_parts(scores, events):
    """Within-bank exact sample ANOVA; no event-probability model is fitted."""
    m = scores.shape[1]//2
    x = scores.reshape(len(scores), 2, m, -1)
    event = events.reshape(len(scores), 2, m)
    mean = x.mean(axis=2, keepdims=True)
    components = {}
    group_mean = np.zeros_like(x)
    for name, mask in (("within_event", event), ("within_none", ~event)):
        count = mask.sum(axis=2, keepdims=True)
        mu = np.sum(x*mask[..., None], axis=2, keepdims=True)/np.maximum(count[..., None], 1)
        components[name] = ((x-mu)*mask[..., None]).reshape(scores.shape)
        group_mean += mu*mask[..., None]
    components["occurrence"] = (group_mean-mean).reshape(scores.shape)
    centered = (x-mean).reshape(scores.shape)
    error = float(np.max(np.abs(centered-sum(components.values()))))
    if error>1e-10*max(1., np.max(np.abs(centered))): raise ValueError("Fleet scatter partition failed")
    return {"total": centered, **components}


def event_counts(events, retained):
    m = events.shape[1]//2;out = {}
    for population, mask in (("all", np.ones(len(events), bool)), ("excluded", ~retained), ("retained", retained)):
        counts = events[mask].reshape(-1, 2, m).sum(axis=2)
        out[population] = {"state_future_exposures": int(mask.sum()*2*m), "event_exposures": int(counts.sum()),
            "state_bank_exposures": int(counts.size), "zero_event_state_banks": int((counts==0).sum()),
            "one_event_state_banks": int((counts==1).sum()), "all_event_state_banks": int((counts==m).sum())}
    return out


def query_stats(design, scores, parts, events, retained, anchor_bank):
    m = events.shape[1]//2;denominator = m*(m-1)
    gram = design.T@design
    dot = lambda a, b: np.einsum("ndk,ndk->nd", a@gram, b)
    energies = {t: {p: np.maximum(dot(v, v), 0.)/denominator for p, v in split.items()} for t, split in parts.items()}
    maximum = 0.
    for t in TERMS:
        a = energies[t]
        maximum = max(maximum, abs(float(a["total"].sum()-sum(a[p].sum() for p in PARTS)))/max(1., a["total"].sum()))
    crosses = {name: {p: 2*dot(parts[a][p], parts[b][p])/denominator for p in ("total", *PARTS)} for name, (a, b) in coupled.PAIRS.items()}
    for name, (a, b) in coupled.PAIRS.items():
        for p in ("total", *PARTS):
            c0, ca, cb = (energies[t][p] for t in ("coupled", a, b))
            maximum = max(maximum, abs(float((c0-ca-cb-crosses[name][p]).sum()))/max(1., c0.sum(), ca.sum(), cb.sum()))
    if maximum>1e-9: raise ValueError("Fleet query scatter/cross identity failed")
    populations = {"all": np.ones(len(events), bool), "excluded": ~retained, "retained": retained}
    output = {}
    observed_vectors = {}
    for population, mask in populations.items():
        values, variance = {}, {}
        for t in TERMS:
            s = scores[t][mask]
            gap = (s[:, m:].mean(axis=1)-s[:, :m].mean(axis=1)).sum(axis=0)*(1 if anchor_bank==0 else -1)
            values[t] = float(gap@gram@gap)
            if t=="coupled": observed_vectors[population] = gap
            variance[t] = {p: float(e[mask].sum()) for p, e in energies[t].items()}
        output[population] = {"variance": variance, "observed_squared_sum": values,
            "cross": {name: {p: float(a[mask].sum()) for p, a in c.items()} for name, c in crosses.items()},
            "centered_energy_on_event_draws": float(energies["coupled"]["total"][mask][events[mask]].sum()),
            "centered_energy_on_none_draws": float(energies["coupled"]["total"][mask][~events[mask]].sum())}
    # Delete only a cached label from its state/bank mean; all model operators stay fixed.
    centered = parts["coupled"]["total"]
    sign = np.r_[np.ones(m), -np.ones(m)]*(1 if anchor_bank==0 else -1)
    perturbation = centered*sign[None, :, None]/(m-1)
    influence = {}
    e = energies["coupled"]["total"]*denominator
    scatter = e.reshape(len(e), 2, m).sum(axis=2, keepdims=True)
    after_scatter = np.maximum(scatter-m/(m-1)*e.reshape(len(e), 2, m), 0.)
    variance_delta = (after_scatter/((m-1)*(m-2))-scatter/(m*(m-1))).reshape(e.shape)
    for population, mask in populations.items():
        delta = 2*np.einsum("ndk,k->nd", perturbation@gram, observed_vectors[population])+dot(perturbation, perturbation)
        influence[population] = {"energy": energies["coupled"]["total"]*mask[:, None],
            "variance_delta": variance_delta*mask[:, None], "observed_squared_delta": delta*mask[:, None]}
    return {"feasible_action_cases": len(design), "populations": output, "decomposition_residual": maximum}, influence
