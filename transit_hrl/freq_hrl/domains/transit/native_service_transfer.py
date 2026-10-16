"""Service-only residual fitting with frozen risk geometry and controllers."""

import numpy as np

from freq_hrl.domains.transit import native_component_probe as probe
from freq_hrl.domains.transit import native_state_risk as state

base = state.base
MAPS = ("linear", "action_context_df", "observed_context_df")
SERVICES = ("raw", *(k+s for k in MAPS for s in ("", "_clip")), "mc")
CONTROLS = probe.CONTROLS
ARMS = (*CONTROLS, *(f"{s}__{r}" for s in SERVICES[1:] for r in CONTROLS))
BENCHMARKS = dict(zip((f"mc__{r}" for r in CONTROLS),
    ("all16__mc_service_raw", "all16__mc_service_linear9", "all16__mc_service_observed")))


def designs(maps, z, observed, eligible):
    output = {"linear": base.transform(z[..., 1:], maps["linear"])}
    for k in MAPS[1:]:
        m = maps[k]
        context = state.context(z, observed, eligible, k[:-3])
        output[k] = base.transform(state.lift(z, context, m["context_stats"]), m)
    return output


def fit(cached, linear, z, observed, target, prior, eligible):
    maps = {"linear": linear, **{k+"_df": m for k, m in cached["maps"].items()}}
    weights = base.base.state_weights(eligible)[eligible]
    residual = target-prior
    radius = float(np.sqrt(np.average(residual[eligible]**2, weights=weights)))
    fitted, maximum = {}, 0.
    for k, design in designs(maps, z, observed, eligible).items():
        penalty = .01 if k == "linear" else cached["complexity"][k[:-3]]["penalty"]
        beta, error = base.base.ridge(design[eligible], residual[eligible, None], weights, penalty)
        fitted[k] = {**maps[k], "weights": beta, "penalty": penalty}
        maximum = max(maximum, error)
    return {"maps": fitted, "radius": radius, "head_fits": 3, "normal_error": maximum}


def predict(model, z, observed, prior, eligible, originals):
    services, clips = {"raw": prior.copy()}, {}
    for k, design in designs(model["maps"], z, observed, eligible).items():
        delta = (design@np.asarray(model["maps"][k]["weights"]))[..., 0]
        delta[np.arange(len(z)), originals] = 0.
        services[k] = prior+delta
        services[k+"_clip"] = prior+np.clip(delta, -model["radius"], model["radius"])
        clips[k+"_clip"] = np.abs(delta) > model["radius"]
    return services, clips


def decisions(data, risks, services):
    total = probe.policy.joint.relative(data["heads"], data["originals"])[..., 0]
    return {(r if s == "raw" else f"{s}__{r}"): probe.policy.joint.binding.policy.selected(
        service+risks[r] if s == "mc" else total+(risks[r]-risks["raw_prior"])+(service-services["raw"]),
        data["raw_logits"], data["originals"])
        for s, service in services.items() for r in CONTROLS}


def metrics(services, clips, prior, values, originals, eligible, radius):
    target = values[..., 2]-values[np.arange(len(values)), :, originals, 2][:, :, None]
    return {s: base.metrics(v, prior, target, eligible, radius, clips.get(s)) for s, v in services.items()}
