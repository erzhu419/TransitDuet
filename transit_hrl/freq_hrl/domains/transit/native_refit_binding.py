"""OOF calibration with cost/wait floors on the actual refitted controller."""

import numpy as np

from freq_hrl.domains.transit import native_best_action_learning as learned
from freq_hrl.domains.transit import native_precision_calibration as precision
from freq_hrl.domains.transit import native_prefix_policy_calibration as policy

WAIT_VARIANTS = precision.WAIT_VARIANTS
GATES = ("glm", *(f"{mode}_{v}" for mode in ("raw", "risk", "oof", "final", "joint") for v in WAIT_VARIANTS))
VARIANTS = {g: ("ridge", g) for g in GATES}


def final_data(rows, panels, indices, bundle, *, batch=None):
    costs, logits = precision.bundle_prediction(rows, bundle, batch=batch)
    _, masses = learned.half_targets(rows, panels, indices)
    return {"originals": np.asarray([r["original_choice"] for r in rows]),
        "roots": np.asarray([r["root"] for r in rows]), "costs": costs, "offset": logits["glm"],
        **{v: logits[v] for v in WAIT_VARIANTS},
        "values": np.stack([panels[learned.state_key(r)]["values"][list(indices)].mean(axis=0) for r in rows]),
        **{name: np.asarray([masses[learned.state_key(r)][field] for r in rows])
            for name, field in (("positive", "positive_wait_min"), ("negative", "negative_wait_min"))}}


def event_candidates(datasets, variant, risk_alpha):
    crossings = []
    for data in datasets:
        residual = data[variant]-data["offset"]
        mask = (np.arange(6)[None, :] != data["originals"][:, None]) & (residual != 0)
        q = -data["offset"][mask]/residual[mask]
        crossings.extend(q[(q >= 0) & (q <= 1)].tolist())
    events = np.unique(np.r_[0., 1., crossings])
    candidates = {0., 1., float(risk_alpha), *events.tolist()}
    for low, high in zip(events[:-1], events[1:]):
        left, right = np.nextafter(low, high), np.nextafter(high, low)
        candidates.update((float(left), float(right), float((low+high)/2), float(np.clip(risk_alpha, left, right))))
    return sorted(candidates)


def floors_met(result, bounds):
    return all(result[root][name] <= bounds[root][name] for root in bounds for name in ("cost", "wait_min"))


def bind(oof, final, variant, risk_alpha, *, enforced):
    datasets = {"oof": oof, "final": final}
    bounds = {name: policy.outcomes(data, data[variant], 1.) for name, data in datasets.items()}
    candidates = event_candidates(datasets.values(), variant, risk_alpha)
    feasible = []
    for alpha in candidates:
        outcomes = {name: policy.outcomes(data, data[variant], alpha) for name, data in datasets.items()}
        if all(floors_met(outcomes[name], bounds[name]) for name in enforced):
            feasible.append((policy.physical_risk(oof, oof[variant], alpha), -alpha, outcomes))
    loss, negative_alpha, outcomes = min(feasible, key=lambda x: (x[0], x[1]))
    alpha = -negative_alpha
    changed = {}
    for name, data in datasets.items():
        raw_choices = policy.selected(data["costs"], data[variant], data["originals"])
        chosen = policy.selected(data["costs"], policy.blended(data["offset"], data[variant], alpha), data["originals"])
        changed[name] = int(np.sum(raw_choices != chosen))
    return {"alpha": alpha, "oof_physical_mass_loss_min": loss, "enforced_scopes": list(enforced),
        "raw_bounds": bounds, "selected": outcomes,
        "floor_met": {name: floors_met(outcomes[name], bounds[name]) for name in datasets},
        "changed_choices_from_raw": changed, "candidate_alphas": len(candidates), "feasible_alphas": len(feasible)}


def calibrate(oof, final, reused):
    result = {"risk": reused["risk"], "oof": reused["policy"], "final": {}, "joint": {}}
    for v in WAIT_VARIANTS:
        risk_alpha = reused["risk"][v]["alpha"]
        result["final"][v] = bind(oof, final, v, risk_alpha, enforced=("final",))
        result["joint"][v] = bind(oof, final, v, risk_alpha, enforced=("oof", "final"))
    diagnostic = {v: {"raw_final_bounds": policy.outcomes(final, final[v], 1.),
        "oof_only_on_final": policy.outcomes(final, final[v], reused["policy"][v]["alpha"])} for v in WAIT_VARIANTS}
    return result, diagnostic


def predict(rows, bundle, calibration, *, batch=None):
    batch = learned.neural.batches(rows) if batch is None else batch
    costs, raw = precision.bundle_prediction(rows, bundle, batch=batch)
    logits = {"glm": raw["glm"], **{f"raw_{v}": raw[v] for v in WAIT_VARIANTS}}
    for mode in ("risk", "oof", "final", "joint"):
        for v in WAIT_VARIANTS:
            logits[f"{mode}_{v}"] = policy.blended(raw["glm"], raw[v], calibration[mode][v]["alpha"])
    originals = np.asarray([r["original_choice"] for r in rows])
    x = learned.centered_cost(batch[0], rows, bundle["best"])
    return {"costs": {"ridge": costs, "best": x@np.asarray(bundle["best"]["coefficients"])},
        "wait_logits": logits, "choices": {g: policy.selected(costs, logits[g], originals) for g in GATES}}
