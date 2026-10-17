"""Learn state-level service calibration independently of frozen action contrasts."""

import numpy as np

from freq_hrl.domains.transit import native_effective_risk as complexity
from freq_hrl.domains.transit import native_service_centering as previous

service = previous.previous.previous
base = service.base
KINDS = ("action_level", "observed_level")
NEW_SERVICES = tuple(f"{k}{suffix}_clip" for k in KINDS for suffix in ("", *(f"_{m}_contrast" for m in service.MAPS)))
SERVICES = (*previous.SERVICES, *NEW_SERVICES)
REPORT_SERVICES = ("raw", "linear_clip", "action_context_df_clip", "observed_context_df_clip",
    "observed_context_df_offset_clip", "observed_context_df_level_clip", "observed_context_df_centered_clip",
    "observed_context_df_offset_centered_clip", "mc", *NEW_SERVICES)
ARM_RISK = {**previous.ARM_RISK, **{f"{s}__{r}": r for s in NEW_SERVICES for r in service.CONTROLS}}
ARMS = tuple(ARM_RISK)
REPORT_ARMS = (*service.CONTROLS, *(f"{s}__{r}" for s in REPORT_SERVICES[1:] for r in service.CONTROLS))


def features(cached, z, observed, eligible):
    action_design = base.transform(z[..., 1:], cached["maps"]["linear"])
    action = np.stack([previous.center(action_design[..., j], eligible)[0] for j in range(1, 9)], axis=-1)
    stats = cached["maps"]["observed_context_df"]["context_stats"]
    return {"action_level": action,
        "observed_level": np.tanh((observed-np.asarray(stats["center"]))/np.asarray(stats["scale"]))}


def fit(cached, coordinates, target, prior, eligible):
    present = eligible.any(axis=1)
    mask = present[:, None]
    level, _ = previous.center(target-prior, eligible)
    designs, models, spectra = {}, {}, {}
    for k, x in coordinates.items():
        models[k] = base.normalization(x[:, None], mask)
        designs[k] = base.transform(x[:, None], models[k])
        spectra[k] = complexity.spectrum(designs[k], mask)
    target_df = min(complexity.effective_df(e, .01) for e in spectra.values())
    maximum, df_error = 0., 0.
    for k in KINDS:
        penalty = complexity.matched_penalty(spectra[k], target_df)
        beta, error = base.base.ridge(designs[k][mask], level[present, None], np.ones(present.sum()), penalty)
        actual_df = complexity.effective_df(spectra[k], penalty)
        models[k].update(weights=beta, penalty=penalty, effective_df=actual_df)
        maximum, df_error = max(maximum, error), max(df_error, abs(actual_df-target_df))
    uniform_error = abs(float(level[present].mean())-float(np.asarray(cached["maps"]["linear"]["weights"])[0, 0]))
    if max(df_error, uniform_error) > 1e-8:
        raise ValueError("Changed training mean or unmatched state-level complexity")
    return {"maps": models, "radius": cached["radius"], "head_fits": 2, "parameters_per_head": 9,
        "eligible_fit_states": int(present.sum()), "target_df": target_df,
        "normal_error": maximum, "df_error": df_error, "uniform_error": uniform_error}


def predict(model, coordinates, services, eligible, originals):
    output, clips = dict(services), {}
    contrasts = {m: previous.center(services[m]-services["raw"], eligible)[1] for m in service.MAPS}
    for k in KINDS:
        m = model["maps"][k]
        level = (base.transform(coordinates[k][:, None], m)@np.asarray(m["weights"]))[:, 0, 0]
        additions = {"": np.zeros_like(services["raw"]), **{f"_{name}_contrast": value for name, value in contrasts.items()}}
        for suffix, contrast in additions.items():
            delta = level[:, None]+contrast
            delta[np.arange(len(delta)), originals] = 0.
            name = f"{k}{suffix}_clip"
            output[name] = services["raw"]+np.clip(delta, -model["radius"], model["radius"])
            clips[name] = np.abs(delta) > model["radius"]
    return output, clips
