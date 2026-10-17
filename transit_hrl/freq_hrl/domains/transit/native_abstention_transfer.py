"""Frozen finite-MC abstention rule and paired new-prefix inference."""

import numpy as np

ARMS = ("raw", "stage247", "zero_abstain", "matched_thin")
FIELDS = ("cost", "fleet", "service", "wait_min")
PAIRS = (("stage247", "raw"), ("zero_abstain", "raw"), ("matched_thin", "raw"),
    ("zero_abstain", "matched_thin"), ("zero_abstain", "stage247"))


def propose(logits, originals, raw, draws):
    mean = draws.mean(axis=1)
    relative = mean-mean[np.arange(len(raw)), raw][:, None]
    eligible = logits <= 0
    eligible[np.arange(len(raw)), originals] = True
    if not np.all(eligible[np.arange(len(raw)), raw]):
        raise ValueError("Raw action is outside the frozen native menu")
    costs = np.where(eligible & (relative[..., 3] <= 0), relative[..., 0], np.inf)
    chosen = costs.argmin(axis=1)
    for preferred in (originals, raw):
        tied = costs[np.arange(len(raw)), preferred] == costs[np.arange(len(raw)), chosen]
        chosen[tied] = preferred[tied]
    return chosen


def paired_values(panels, choices, raw):
    return np.stack([panels[np.arange(len(c)), :, c]-panels[np.arange(len(c)), :, r]
        for c, r in zip(choices, raw)])


def screen(proposal, raw, draws):
    fleet = paired_values(draws, proposal, raw)[..., 1]
    eligible = (proposal != raw) & (fleet.mean(axis=2) <= 0)
    keep = eligible & ~np.all(fleet == 0, axis=2)
    return eligible, keep


def thin_probabilities(cube, eligible, keep):
    pool, retained = eligible[:, cube], keep[:, cube]
    n, k = pool.sum(axis=2), retained.sum(axis=2)
    fraction = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
    probability = np.zeros_like(eligible, dtype=float)
    probability[:, cube] = pool*fraction[:, :, None]
    error = float(np.max(abs(probability[:, cube].sum(axis=2)-k)))
    if error > 1e-12:
        raise ValueError("Expected thinning does not preserve screen-only counts")
    return probability, {"eligible_changes": int(eligible.sum()), "kept_changes": int(keep.sum()),
        "expected_thin_changes": float(probability.sum()), "matched_count_max_abs_difference": error,
        "strata": int(n.size), "randomizable_kept_changes": int(k[(k > 0)&(k < n)].sum())}


def seal(raw, old, proposal, screening, cube):
    eligible, keep = screen(proposal, raw, screening)
    probability, counts = thin_probabilities(cube, eligible, keep)
    choices = np.stack((raw, old, np.where(keep, proposal, raw), raw), axis=1)
    return {"arms": list(ARMS), "choices": choices.tolist(), "proposal": proposal.tolist(),
        "matched_thin_probabilities": probability.tolist(), "matching": counts}


def action_unions(sealed, originals):
    choices = np.asarray(sealed["choices"])
    proposal, probability = np.asarray(sealed["proposal"]), np.asarray(sealed["matched_thin_probabilities"])
    unions = []
    for i, original in enumerate(originals):
        union = set()
        for p in range(len(choices)):
            actions = set(map(int, choices[p, :3, i]))
            if probability[p, i] > 0:
                actions.add(int(proposal[p, i]))
            if len(actions) > 1:
                union.update(actions)
        unions.append(sorted(union | {int(original)}) if union else [])
    return unions


def effects(sealed, panels):
    choices = np.asarray(sealed["choices"])
    actual = np.stack([paired_values(panels, choices[:, v], choices[:, 0]).mean(axis=0)
        for v in range(len(ARMS))], axis=2)
    actual[:, :, 3] = (paired_values(panels, np.asarray(sealed["proposal"]), choices[:, 0])
        *np.asarray(sealed["matched_thin_probabilities"])[:, :, None, None]).mean(axis=0)
    return actual


def summarize(sealed, panels, cube, *, seed=294):
    actual = effects(sealed, panels)
    block = actual[cube]
    means = block.mean(axis=(2, 3))
    g, r, j, d = block.shape[:4]
    centered = block-block.mean(axis=3, keepdims=True)
    covariance = np.einsum("grjdvf,grjdwf->vwf", centered, centered)/(j*j*d*(d-1)*(g*r)**2)
    rng = np.random.default_rng(seed)
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
            "conditional_MC_se": dict(zip(FIELDS, map(float, se))),
            "conditional_MC_CI95": {f: list(map(float, mc[:, k])) for k, f in enumerate(FIELDS)}}
    halves = {name: {arm: dict(zip(FIELDS, map(float, actual[:, ds, vi].mean(axis=(0, 1)))))
        for vi, arm in enumerate(ARMS)} for name, ds in (("first_half", slice(0, d//2)), ("last_half", slice(d//2, d)))}
    passed = lambda pair, fields: all(comparisons[pair][interval][f][1] < 0
        for interval in ("prefix_CI95", "conditional_MC_CI95") for f in fields)
    half_pass = all(halves[h]["zero_abstain"][f] < 0 for h in halves for f in ("cost", "wait_min"))
    joint = passed("zero_abstain_vs_raw", ("cost", "wait_min"))
    return {"comparisons": comparisons, "halves": halves, "primary_both_CI_passed": joint,
        "primary_both_halves_passed": half_pass, "joint_transfer_supported": joint and half_pass,
        "cost_information_supported": passed("zero_abstain_vs_matched_thin", ("cost",)),
        "fleet_information_supported": passed("zero_abstain_vs_matched_thin", ("fleet",)),
        "descriptive_group_means": means.mean(axis=1).tolist()}
