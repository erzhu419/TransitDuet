"""Fixed-candidate calibration and paired, shared-future control attribution."""

import numpy as np

from freq_hrl.domains.transit import native_service_level as previous
from freq_hrl.domains.transit.native_signed_risk import ZERO

service, policy = previous.service, previous.service.probe.policy
SERVICES = ("raw", "observed_context_df_offset_clip", "action_level_clip", "observed_level_clip",
    *(f"observed_level_{m}_contrast_clip" for m in service.MAPS))
ARM_RISK = {(r if s == "raw" else f"{s}__{r}"): r for r in service.CONTROLS for s in SERVICES}
PAIRS = {}
for r in service.CONTROLS:
    arm = lambda s: f"{s}__{r}"
    PAIRS.update({f"{name}__{r}": (arm(after), arm(before)) for name, after, before in (
        ("action_vs_uniform", "action_level_clip", "observed_context_df_offset_clip"),
        ("observed_vs_uniform", "observed_level_clip", "observed_context_df_offset_clip"),
        ("observed_vs_action", "observed_level_clip", "action_level_clip"),
        *((f"{m}_contrast_vs_level", f"observed_level_{m}_contrast_clip", "observed_level_clip") for m in service.MAPS))})
FIELDS = ("service_MSE", "risk_MSE", "service_risk_cross", "margin_MSE", "service_bias", "risk_bias", "margin_bias", "margin_sign_error")


def candidates(data, risks, services):
    original = data["originals"]
    eligible = (data["raw_logits"] <= 0)&(np.arange(6)[None] != original[:, None])
    total = policy.joint.relative(data["heads"], original)[..., 0]
    indices, predictions = {}, {}
    for r in service.CONTROLS:
        score = total+(risks[r]-risks["raw_prior"])
        chosen = np.where(eligible.any(axis=1), np.where(eligible, score, np.inf).argmin(axis=1), original)
        indices[r] = chosen
        ix = np.arange(len(original)), chosen
        for s in SERVICES:
            a = r if s == "raw" else f"{s}__{r}"
            predictions[a] = np.stack((services[s][ix], risks[r][ix], (score+(services[s]-services["raw"]))[ix]), axis=-1)
    return eligible, indices, predictions


def errors(values, originals, eligible, indices, predictions):
    present = eligible.any(axis=1)
    records, maximum = {}, 0.
    for h, ds in service.probe.SLICES.items():
        mean = values[:, ds].mean(axis=1)
        relative = mean-mean[np.arange(len(mean)), originals][:, None]
        records[h] = {}
        for a, prediction in predictions.items():
            actual = relative[np.arange(len(mean)), indices[ARM_RISK[a]]]
            se, re, te = prediction[:, 0]-actual[:, 2], prediction[:, 1]-actual[:, 1], prediction[:, 2]-actual[:, 0]
            terms = (se**2, re**2, 2*se*re, te**2, se, re, te, (prediction[:, 2] < 0) != (actual[:, 0] < 0))
            maximum = max(maximum, float(np.max(np.abs(te**2-se**2-re**2-2*se*re), initial=0.)))
            records[h][a] = {"cases": int(present.sum()), "sums": {f: float(v[present].sum()) for f, v in zip(FIELDS, terms)}}
    if maximum > 1e-10:
        raise ValueError("Candidate service/risk/margin identity failed")
    return records, maximum


def summarize_errors(rows):
    count = sum(r["cases"] for r in rows)
    return {"eligible_state_exposures": count, **{f: sum(r["sums"][f] for r in rows)/count for f in FIELDS}}


def add_units(bucket, states, deltas):
    bucket["cases"] += len(states)
    fleet = deltas[..., 1]
    bucket["fleet_up"] += (fleet > ZERO).sum(axis=(0, 2))
    bucket["fleet_down"] += (fleet < -ZERO).sum(axis=(0, 2))
    for state, delta in zip(states, deltas):
        state = tuple(state)
        if state not in bucket["units"]:
            bucket["units"][state] = np.zeros_like(delta)
            bucket["state_cases"][state] = 0
        bucket["units"][state] += delta
        bucket["state_cases"][state] += 1


def new_bucket():
    return {"cases": 0, "units": {}, "state_cases": {},
        "fleet_up": np.zeros(len(PAIRS), int), "fleet_down": np.zeros(len(PAIRS), int)}


def summarize_units(bucket):
    states = sorted(bucket["units"])
    values = np.stack([bucket["units"][s] for s in states])
    n, draws = bucket["cases"], values.shape[2]
    means = values.mean(axis=2).sum(axis=0)/n
    # Sum reused model contributions before estimating variance of unique seeds.
    ses = np.sqrt(values.var(axis=2, ddof=1).sum(axis=0)/draws)/n
    output = {}
    for c, name in enumerate(PAIRS):
        prefixes = []
        for prefix in sorted({s[:2] for s in states}):
            mask = np.asarray([s[:2] == prefix for s in states])
            contribution = values[mask, c].mean(axis=1).sum(axis=0)/n
            prefixes.append({"prefix": list(prefix), "state_exposures": sum(bucket["state_cases"][s] for s in states if s[:2] == prefix),
                **dict(zip(policy.FIELDS, map(float, contribution)))})
        output[name] = {"state_exposures": n, "unique_states": len(states), "unique_state_future_units": len(states)*draws,
            "mean": dict(zip(policy.FIELDS, map(float, means[c]))), "conditional_MC_se": dict(zip(policy.FIELDS, map(float, ses[c]))),
            "normal_95_MC": {f: [float(m-1.96*se), float(m+1.96*se)] for f, m, se in zip(policy.FIELDS, means[c], ses[c])},
            "fleet_up_exposures": int(bucket["fleet_up"][c]), "fleet_down_exposures": int(bucket["fleet_down"][c]),
            "net_fleet_up_unique_units": int(np.count_nonzero(values[:, c, :, 1] > ZERO)),
            "net_fleet_down_unique_units": int(np.count_nonzero(values[:, c, :, 1] < -ZERO)), "prefix_contributions": prefixes}
    return output
