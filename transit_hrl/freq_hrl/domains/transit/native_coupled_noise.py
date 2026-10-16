"""Joint full/prefix local KKT noise with shared state/future labels."""

import numpy as np

from freq_hrl.domains.transit import native_calibration_ablation as calibration
from freq_hrl.domains.transit import native_curvature_response as curvature
from freq_hrl.domains.transit import native_joint_noise as noise
from freq_hrl.domains.transit import native_tangent_noise as previous

TERMS = ("prefix", "minus_full", "coupled", "one_ended", "operator_change")
PAIRS = {"endpoints": ("prefix", "minus_full"), "operator_change": ("one_ended", "operator_change")}


def response_scores(full, prefix, tf, tp):
    apply = lambda matrix, scores: np.einsum("ij,ndhj->ndhi", matrix, scores)
    p, f = apply(-tp, prefix), apply(tf, full)
    return {"prefix": p, "minus_full": f, "coupled": p+f,
        "one_ended": apply(-tp, prefix-full), "operator_change": apply(tf-tp, full)}


def cross_covariances(a, b, retained):
    m = a.shape[1]//2
    center = lambda v: np.concatenate([q-q.mean(axis=1, keepdims=True) for q in (v[:, :m], v[:, m:])], axis=1)
    a, b = center(a), center(b)
    output = {}
    for population, mask in (("", np.ones(len(a), bool)), ("excluded_", ~retained), ("retained_", retained)):
        sa, ra, sb, rb = a[mask, :, 0], a[mask, :, 1], b[mask, :, 0], b[mask, :, 1]
        gram = lambda x, y: np.einsum("ndk,ndl->kl", x, y)/(m*(m-1))
        sym = lambda x, y: gram(x, y)+gram(y, x)
        output.update({population+"service": sym(sa, sb), population+"risk": sym(ra, rb),
            population+"joint": sym(sa+ra, sb+rb), population+"cross_sr": sym(sa, rb)+sym(ra, sb)})
    return output


def materialize(scores, retained, low, bank):
    m = scores.shape[1]//2
    gap = (scores[:, m:].mean(axis=1)-scores[:, :m].mean(axis=1))*(1 if bank==0 else -1)
    responses = {}
    for population, mask in (("", np.ones(len(scores), bool)), ("excluded_", ~retained), ("retained_", retained)):
        for head, index in (("service", 0), ("risk", 1)):
            responses[population+head] = gap[mask, index].sum(axis=0)
        responses[population+"joint"] = responses[population+"service"]+responses[population+"risk"]
    return {"responses": responses, "parameter_covariance": noise.noise_covariances(scores, retained), "low_basis": low}


def context(z, c, paired, eligible, prior_heads, full_heads, prefix_heads, indices, model, radius, bank, level, scale):
    target, precision, rho = (np.asarray(c[k]) for k in ("target", "precision", "reliability"))
    difference, penalty = prior_heads[0]-prior_heads[1], c["penalties"][level]
    factory = lambda zz, y, w, ok: calibration.objectives(zz, y, w, ok, difference, penalty)[model]
    lf, hf, _, _ = factory(z, target, precision, eligible)
    lp, hp, _, _ = factory(z[indices], target[indices], precision[indices], eligible[indices])
    full, prefix = full_heads.sum(axis=0), prefix_heads.sum(axis=0)
    mf, mp = curvature.multiplier(lf, full, radius), curvature.multiplier(lp, prefix, radius)
    of, op = previous.operators(hf(full), full, mf, radius), previous.operators(hp(prefix), prefix, mp, radius)
    tf, tp = of["tangent"], op["tangent"]
    optimum_residual = max(float(np.linalg.norm(loss(u)[1]+mu*u, ord=np.inf)) for loss, u, mu in ((lf, full, mf), (lp, prefix, mp)))
    if optimum_residual>1e-8: raise ValueError("Cached endpoint is not KKT stationary")
    draws = np.concatenate([paired[b] for b in (0, 1)], axis=1)*rho[:, None]/scale
    sf = noise.state_scores(z, draws, precision, eligible, model, penalty)
    sp = np.zeros_like(sf)
    sp[indices] = noise.state_scores(z[indices], draws[indices], precision[indices], eligible[indices], model, penalty)
    scores = response_scores(sf, sp, tf, tp)
    retained = np.zeros(len(z), bool);retained[indices] = True
    low = curvature.low_subspace(hp(prefix))[0]
    terms = {name: materialize(s, retained, low, bank) for name, s in scores.items()}
    cross = {name: cross_covariances(scores[a], scores[b], retained) for name, (a, b) in PAIRS.items()}
    error = max(noise.covariance_error(t["parameter_covariance"]) for t in terms.values())
    response_error = 0.
    for name, (a, b) in PAIRS.items():
        for field in noise.COVARIANCES:
            c0, ca, cb = (terms[t]["parameter_covariance"][field] for t in ("coupled", a, b))
            d = c0-ca-cb-cross[name][field]
            error = max(error, float(np.linalg.norm(d, ord=np.inf)/max(1., np.linalg.norm(c0, ord=np.inf), np.linalg.norm(ca, ord=np.inf), np.linalg.norm(cb, ord=np.inf))))
        for field in noise.POSITIVE:
            c0, ca, cb = (terms[t]["responses"][field] for t in ("coupled", a, b))
            response_error = max(response_error, float(np.max(np.abs(c0-ca-cb))))
    change = (paired[1-bank].mean(axis=1)-paired[bank].mean(axis=1))*rho/scale
    lf2 = factory(z, target+change, precision, eligible)[0]
    lp2 = factory(z[indices], (target+change)[indices], precision[indices], eligible[indices])[0]
    direct = -tp@(lp2(prefix)[1]-lp(prefix)[1])+tf@(lf2(full)[1]-lf(full)[1])
    direct_error = float(np.max(np.abs(direct-terms["coupled"]["responses"]["joint"])))
    if error>1e-8 or max(response_error, direct_error)>1e-9*max(1., np.linalg.norm(direct)):
        raise ValueError("Coupled score/covariance identities failed")
    energies = {t: max(0., float(np.trace(c["parameter_covariance"]["joint"]))) for t, c in terms.items()}
    return {"terms": terms, "cross_covariance": cross, "diagnostics": {"full": of["diagnostics"], "prefix": op["diagnostics"],
        "radius": radius, "parameter_noise_energy": energies,
        "noise_norm_RMS_over_radius": {t: float(np.sqrt(e)/radius) for t, e in energies.items()},
        "parameter_noise_cross": {t: float(np.trace(c["joint"])) for t, c in cross.items()},
        "endpoint_optimality_residual": optimum_residual, "covariance_residual": error,
        "response_identity_residual": response_error, "direct_gradient_residual": direct_error}}


def region_stats(z, mask, feasible, context, physical):
    terms = {n: noise.region_stats(z, mask, feasible, c, physical) for n, c in context["terms"].items()}
    x = z[mask & feasible]*physical
    low = (x@context["terms"]["coupled"]["low_basis"])@context["terms"]["coupled"]["low_basis"].T
    cross, maximum = {}, 0.
    for band, design in (("full", x), ("low", low), ("rest", x-low)):
        gram = design.T@design;cross[band] = {}
        for name, (a, b) in PAIRS.items():
            nc = {n: float(np.einsum("ij,ji->", gram, c)) for n, c in context["cross_covariance"][name].items()}
            pa, pb = ({n: design@v for n, v in context["terms"][t]["responses"].items()} for t in (a, b))
            oc = {n: float(2*pa[n]@pb[n]) for n in noise.POSITIVE}
            cross[band][name] = {"noise_squared_sum": nc, "observed_squared_sum": oc}
            for field, signed in (("noise_squared_sum", nc), ("observed_squared_sum", oc)):
                for head in noise.POSITIVE:
                    c0, ca, cb = (terms[t]["bands"][band][field][head] for t in ("coupled", a, b))
                    maximum = max(maximum, abs(c0-ca-cb-signed[head])/max(1., abs(c0), abs(ca), abs(cb)))
    if maximum>1e-8: raise ValueError("Query coupled-noise identity failed")
    return {"terms": terms, "cross": cross, "decomposition_residual": maximum}
