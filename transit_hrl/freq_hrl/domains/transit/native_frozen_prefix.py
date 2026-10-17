"""Five frozen controls on causal rows, with prefix-level paired inference."""

import numpy as np

from freq_hrl.domains.transit import native_service_level as level

VARIANTS = ("raw", "stage247", "uniform", "action_level", "observed_level")
FIELDS = ("cost", "fleet", "service", "wait_min")
PRIMARY = (("observed_level", "uniform"), ("observed_level", "stage247"))


def infer(rows, package, legacy):
    data = legacy.learning.base.causal_data(rows, package["reference"])
    original = data["originals"]
    total = level.service.probe.policy.joint.relative(data["heads"], original)[..., 0]
    risk = level.service.probe.policy.joint.relative(data["components"], original)[..., 1]
    prior = total-risk
    eligible = (data["raw_logits"] <= 0)&(np.arange(6)[None] != original[:, None])
    x = level.service.probe.policy.joint.relative(data["x"][..., :198], original)@np.asarray(package["basis"])
    z = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=-1)
    observed = level.service.state.observed_context(rows)
    old_path = legacy.learning.base.candidates(data, package["stage247_model"])
    old = old_path[package["stage247_selection"] or "zero"]["chosen"]
    output = []
    for bank in package["banks"]:
        cached, model = bank["geometry"], bank["level"]
        coordinates = level.features(cached, z, observed, eligible)
        offsets = {"uniform": np.full(len(rows), np.asarray(cached["maps"]["linear"]["weights"])[0, 0])}
        for kind in level.KINDS:
            m = model["maps"][kind]
            offsets[kind] = (level.base.transform(coordinates[kind][:, None], m)@np.asarray(m["weights"]))[:, 0, 0]
        choices = {"raw": data["raw_choices"], "stage247": old}
        for kind, scalar in offsets.items():
            delta = np.broadcast_to(scalar[:, None], total.shape).copy()
            delta[np.arange(len(rows)), original] = 0.
            service = prior+np.clip(delta, -cached["radius"], cached["radius"])
            choices[kind] = level.service.probe.policy.joint.binding.policy.selected(total+(risk-risk)+(service-prior), data["raw_logits"], original)
        output.append({"bank": bank["bank"], "choices": {v: choices[v] for v in VARIANTS}})
    return output


def active_union(profiles, originals):
    output = []
    for i, original in enumerate(originals):
        actions = set()
        for profile in profiles:
            row = [profile["choices"][v][i] for v in VARIANTS]
            if len(set(row)) > 1:
                actions.update(row)
        output.append(sorted(actions|{int(original)}) if actions else [])
    return output


def prefix_effects(profiles, labels, draws):
    """Average fixed model profiles, retaining paired MC units per decision."""
    effects = np.zeros((6, draws, len(VARIANTS), len(FIELDS)))
    for p in range(6):
        for profile in profiles:
            raw = int(profile["choices"]["raw"][p])
            for vi, variant in enumerate(VARIANTS):
                chosen = int(profile["choices"][variant][p])
                if chosen == raw:
                    continue
                for d in range(draws):
                    effects[p, d, vi] += np.asarray(labels[p, d][chosen])-np.asarray(labels[p, d][raw])
    effects /= len(profiles)
    centered = effects-effects.mean(axis=1, keepdims=True)
    covariance = np.einsum("pdvf,pdwf->vwf", centered, centered)/(36*draws*(draws-1))
    return effects.mean(axis=(0, 1)), covariance


def summarize_prefixes(records, *, new, bootstrap=5000):
    means = np.asarray([r["mean"] for r in records])
    mc_cov = np.asarray([r["MC_covariance"] for r in records]).sum(axis=0)/len(records)**2
    names = {"vs_raw_"+v: (v, "raw") for v in VARIANTS[1:]}
    names.update({a+"_vs_"+b: (a, b) for a, b in PRIMARY})
    samples = None
    if new:
        rng = np.random.default_rng(287)
        strata = sorted({(r["root"], r["scenario"]) for r in records})
        samples = np.zeros((bootstrap, len(VARIANTS), len(FIELDS)))
        for stratum in strata:
            indices = np.flatnonzero([(r["root"], r["scenario"]) == stratum for r in records])
            if len(indices) != 8:
                raise ValueError("New-prefix inference requires eight frozen scenes per stratum")
            samples += means[rng.choice(indices, (bootstrap, len(indices)), replace=True)].mean(axis=1)/len(strata)
    result = {}
    for name, (after, before) in names.items():
        a, b = VARIANTS.index(after), VARIANTS.index(before)
        value = (means[:, a]-means[:, b]).mean(axis=0)
        variance = mc_cov[a, a]+mc_cov[b, b]-2*mc_cov[a, b]
        row = {"mean": dict(zip(FIELDS, map(float, value))),
            "conditional_MC_se": dict(zip(FIELDS, map(float, np.sqrt(np.maximum(variance, 0.)))))}
        if samples is not None:
            bounds = np.quantile(samples[:, a]-samples[:, b], [.025, .975], axis=0)
            row["prefix_CI95"] = {f: list(map(float, bounds[:, i])) for i, f in enumerate(FIELDS)}
        result[name] = row
    return {"prefixes": len(records), "states": len(records)*6,
        "fixed_profile_count": records[0]["profiles"], "contrasts": result}
