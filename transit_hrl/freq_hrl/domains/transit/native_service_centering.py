"""Separate a frozen service correction's state level from action contrasts."""

import numpy as np

from freq_hrl.domains.transit import native_service_offset as previous

MAPS, CONTROLS = previous.MAPS, previous.CONTROLS
NEW_SERVICES = tuple(f"{k}_{p}_clip" for k in MAPS for p in ("level", "centered", "offset_centered"))
SERVICES = (*previous.SERVICES, *NEW_SERVICES)
ARM_RISK = {**previous.ARM_RISK, **{f"{s}__{r}": r for s in NEW_SERVICES for r in CONTROLS}}
ARMS = tuple(ARM_RISK)
REPORT_ARMS = (*previous.REPORT_ARMS, *(f"{s}__{r}" for s in NEW_SERVICES for r in CONTROLS))
FIELDS = ("MSE", "level_MSE", "contrast_MSE", "level_bias")


def center(values, eligible):
    count = eligible.sum(axis=1)
    mean = np.divide(np.sum(np.where(eligible, values, 0.), axis=1), count,
        out=np.zeros(len(count)), where=count > 0)
    return mean, values-mean[:, None]


def ablate(services, parts, eligible, originals, radius):
    output, clips = dict(services), {}
    for k in MAPS:
        delta = services[k]-services["raw"]
        mean, centered = center(delta, eligible)
        level = np.broadcast_to(mean[:, None], delta.shape).copy()
        for label, value in (("level", level), ("centered", centered),
                ("offset_centered", np.asarray(parts[k][0])+centered)):
            value[np.arange(len(value)), originals] = 0.
            name = f"{k}_{label}_clip"
            output[name] = services["raw"]+np.clip(value, -radius, radius)
            clips[name] = np.abs(value) > radius
    return output, clips


def error_geometry(values, services, originals, eligible):
    weights = previous.previous.base.base.state_weights(eligible)
    present = eligible.any(axis=1)
    records, maximum = {}, 0.
    for h, draws in previous.previous.probe.SLICES.items():
        mean = values[:, draws, :, 2].mean(axis=1)
        target = mean-mean[np.arange(len(mean)), originals][:, None]
        records[h] = {}
        for s, prediction in services.items():
            error = prediction-target
            level, contrast = center(error, eligible)
            terms = (np.sum(weights*error**2), np.sum(level[present]**2),
                np.sum(weights*contrast**2), np.sum(level[present]))
            maximum = max(maximum, abs(terms[0]-terms[1]-terms[2]))
            # Each state, not each pair, receives equal weight in ranking error.
            a, b = np.triu_indices(prediction.shape[1], 1)
            truth = np.sign(target[:, a]-target[:, b])
            pred = np.sign(prediction[:, a]-prediction[:, b])
            valid = eligible[:, a]&eligible[:, b]&(truth != 0)
            counts = valid.sum(axis=1)
            losses = np.where(pred == 0, .5, pred != truth)
            rates = np.divide(np.sum(np.where(valid, losses, 0.), axis=1), counts,
                out=np.zeros(len(counts)), where=counts > 0)
            rank_sum, rank_cases = rates.sum(), np.count_nonzero(counts)
            records[h][s] = {"cases": int(present.sum()), "sums": dict(zip(FIELDS, map(float, terms))),
                "rank_cases": rank_cases, "rank_error_sum": float(rank_sum)}
    if maximum > 1e-10:
        raise ValueError("Service level/contrast MSE identity failed")
    return records, maximum


def summarize(rows):
    count = sum(r["cases"] for r in rows)
    pairs = sum(r["rank_cases"] for r in rows)
    return {"eligible_state_exposures": count, **{f: sum(r["sums"][f] for r in rows)/count for f in FIELDS},
        "rank_state_exposures": pairs,
        "pair_inversion_rate": sum(r["rank_error_sum"] for r in rows)/pairs if pairs else None}
