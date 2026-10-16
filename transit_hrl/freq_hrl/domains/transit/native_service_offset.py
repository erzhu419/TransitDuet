"""Frozen service intercept/slopes ablations and exact error accounting."""

import numpy as np

from freq_hrl.domains.transit import native_service_transfer as previous

MAPS, CONTROLS = previous.MAPS, previous.CONTROLS
NEW_SERVICES = tuple(f"{k}_{p}_clip" for k in MAPS for p in ("offset", "slopes"))
SERVICES = (*previous.SERVICES, *NEW_SERVICES)
ARM_RISK = {**{r: r for r in CONTROLS}, **{f"{s}__{r}": r for s in SERVICES[1:] for r in CONTROLS}}
ARMS = tuple(ARM_RISK)
REPORT_ARMS = (*CONTROLS, *(f"{s}__{r}" for s in SERVICES[1:] if s == "mc" or s.endswith("_clip") for r in CONTROLS))
PART_FIELDS = ("raw_MSE", "offset_energy", "slopes_energy", "offset_slopes_cross", "raw_offset_cross", "raw_slopes_cross",
    "full_MSE", "raw_bias", "offset_mean", "slopes_mean", "full_bias")
TYPES = ("unchanged", "original_to_other", "other_to_original", "other_to_other")


def ablate(model, services, originals):
    output, clips, parts = dict(services), {}, {}
    for k in MAPS:
        delta = services[k]-services["raw"]
        offset = np.full(delta.shape, float(np.asarray(model["maps"][k]["weights"])[0, 0]))
        offset[np.arange(len(offset)), originals] = 0.
        slopes = delta-offset
        parts[k] = (offset, slopes)
        for label, value in (("offset", offset), ("slopes", slopes)):
            name = f"{k}_{label}_clip"
            output[name] = services["raw"]+np.clip(value, -model["radius"], model["radius"])
            clips[name] = np.abs(value) > model["radius"]
    return output, clips, parts


def error_parts(values, originals, prior, eligible, parts):
    w = previous.base.base.state_weights(eligible)
    output, maximum = {}, 0.
    for h, ds in previous.probe.SLICES.items():
        mean = values[:, ds, :, 2].mean(axis=1)
        raw_error = prior-(mean-mean[np.arange(len(mean)), originals][:, None])
        output[h] = {}
        for k, (offset, slopes) in parts.items():
            full = raw_error+offset+slopes
            arrays = (raw_error**2, offset**2, slopes**2, 2*offset*slopes, 2*raw_error*offset, 2*raw_error*slopes,
                full**2, raw_error, offset, slopes, full)
            error = float(np.max(np.abs(arrays[6]-sum(arrays[:6])), initial=0.))
            maximum = max(maximum, error)
            output[h][k] = {"cases": int(np.count_nonzero(eligible.sum(axis=1))),
                "sums": {f: float(np.sum(w*v)) for f, v in zip(PART_FIELDS, arrays)}}
    if maximum > 1e-12:
        raise ValueError("Service offset/slopes MSE identity failed")
    return output, maximum


def summarize_parts(rows):
    n = sum(r["cases"] for r in rows)
    result = {f: sum(r["sums"][f] for r in rows)/n for f in PART_FIELDS}
    result.update(offset_only_MSE=result["raw_MSE"]+result["offset_energy"]+result["raw_offset_cross"],
        slopes_only_MSE=result["raw_MSE"]+result["slopes_energy"]+result["raw_slopes_cross"])
    if abs(result["full_MSE"]-sum(result[f] for f in PART_FIELDS[:6])) > 1e-12:
        raise ValueError("Aggregate service MSE identity failed")
    return {"eligible_state_exposures": n, **result}


def switch_stats(values, choices, originals):
    ix = np.arange(len(values))
    output = {"all16": {}}
    for a, chosen in choices.items():
        control = choices[ARM_RISK[a]]
        different = chosen != control
        kind = np.zeros(len(ix), int)
        kind[different&(control == originals)] = 1
        kind[different&(chosen == originals)] = 2
        kind[different&(control != originals)&(chosen != originals)] = 3
        mean = (values[ix, :, chosen]-values[ix, :, control]).mean(axis=1)
        output["all16"][a] = {"cases": len(ix), "types": {label: {"cases": int(np.count_nonzero(kind == i)),
            "sums": dict(zip(previous.probe.policy.FIELDS, map(float, mean[kind == i].sum(axis=0))))}
            for i, label in enumerate(TYPES)}}
    return output


def summarize_switches(rows):
    n = sum(r["cases"] for r in rows)
    return {label: {"cases": sum(r["types"][label]["cases"] for r in rows),
        **{f: sum(r["types"][label]["sums"][f] for r in rows)/n for f in previous.probe.policy.FIELDS}}
        for label in TYPES}
