"""Cross calibration/deployment target precision with common prefix-held labels."""

import numpy as np

from freq_hrl.domains.transit import native_prefix_policy_calibration as base

WAIT_VARIANTS = base.WAIT_VARIANTS
CALIBRATIONS = ("legacy", "c2v4", "c4v4")
GATE_SUFFIXES = ("glm", *(f"raw_{v}" for v in WAIT_VARIANTS),
    *(f"{c}_{m}_{v}" for c in CALIBRATIONS for m in ("risk", "policy") for v in WAIT_VARIANTS))
GATES = tuple(f"f{precision}_{suffix}" for precision in (2, 4) for suffix in GATE_SUFFIXES)
VARIANTS = {g: (f"f{precision}", g) for precision in (2, 4) for g in GATES if g.startswith(f"f{precision}_")}


def fit_bundle(rows, panels, indices, *, batch=None):
    targets, masses = base.base.base.half_targets(rows, panels, indices)
    critic = base.base.base.ridge.fit_value(targets)
    models, fit = base.base.fit_inner_wait(targets, masses, batch=batch)
    return {"ridge": critic, "wait": models}, {"draws": list(indices), "training_states": len(rows),
        "training_groups": critic["training_groups"], "wait_training_groups": models["glm"]["training_groups"], **fit}


def bundle_prediction(rows, bundle, *, batch=None):
    batch = base.base.base.neural.batches(rows) if batch is None else batch
    logits = base.base.wait_logits(rows, bundle["wait"], batch=batch)
    originals = np.asarray([r["original_choice"] for r in rows])
    for z in logits.values(): z[np.arange(len(rows)), originals] = 0.
    critic = bundle["ridge"]
    x = base.base.base.centered_cost(batch[0], rows, critic)
    costs = ((x@np.asarray(critic["coefficients"]))*critic["target_scale"])[..., :2].sum(axis=-1)
    return costs, logits


def validation_rows(rows):
    return [rows[i] for _, _, held in base.prefix_folds(rows) for _ in (0, 1) for i in held]


def common_validation(rows, panels, indices, data):
    repeated = validation_rows(rows)
    originals = np.asarray([r["original_choice"] for r in repeated])
    roots = np.asarray([r["root"] for r in repeated])
    if not np.array_equal(data["originals"], originals) or not np.array_equal(data["roots"], roots):
        raise ValueError("Reused two-draw OOF changed prefix order or action identities")
    _, masses = base.base.base.half_targets(repeated, panels, indices)
    return {**data, "values": np.stack([panels[base.base.base.state_key(r)]["values"][list(indices)].mean(axis=0) for r in repeated]),
        **{name: np.asarray([masses[base.base.base.state_key(r)][field] for r in repeated])
            for name, field in (("positive", "positive_wait_min"), ("negative", "negative_wait_min"))}}


def calibrate_data(data):
    mask = np.arange(6)[None, :] != data["originals"][:, None]
    result = {"risk": {}, "policy": {}}
    for v in WAIT_VARIANTS:
        raw = data[v]
        risk = base.base.shrinkage(data["offset"][mask], (raw-data["offset"])[mask],
            data["positive"][mask], data["negative"][mask])
        result["risk"][v] = risk
        result["policy"][v] = base.policy_shrinkage(data, raw, risk["alpha"])
    return result


def fit_cell_models(train, panels, indices, reused, *, batch=None):
    batch = base.base.base.neural.batches(train) if batch is None else batch
    repeated = validation_rows(train)
    legacy_means = np.stack([panels[base.base.base.state_key(r)]["values"][list(indices[2:] if (i//6) % 2 == 0 else indices[:2])].mean(axis=0)
        for i, r in enumerate(repeated)])
    if not np.array_equal(reused["values"], legacy_means):
        raise ValueError("Reused two-draw OOF labels do not match their prefix/future order")
    pieces, ledger = [], []
    for group, training, validation in base.prefix_folds(train):
        rows, held = [train[i] for i in training], [train[i] for i in validation]
        bundle, fit = fit_bundle(rows, panels, indices, batch=tuple(b[training] for b in batch))
        costs, logits = bundle_prediction(held, bundle, batch=tuple(b[validation] for b in batch))
        piece = {"originals": np.asarray([r["original_choice"] for r in held]),
            "roots": np.asarray([r["root"] for r in held]), "costs": costs, "offset": logits["glm"],
            **{v: logits[v] for v in WAIT_VARIANTS}}
        # Repeat the same four-draw predictor to match the two OOF half-models.
        pieces.extend((piece, piece))
        ledger.append({"held_group": list(group), **fit})
        print(f"PRECISION_INNER_COMPLETE {group[0]}/{group[1]}", flush=True)
    c4 = common_validation(train, panels, indices, {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]})
    c2 = common_validation(train, panels, indices, reused)
    calibration = {"c2v4": calibrate_data(c2), "c4v4": calibrate_data(c4)}
    final, final_fits = [], []
    for draws in (indices[:2], indices[2:]):
        model, fit = fit_bundle(train, panels, draws, batch=batch)
        final.append(model)
        final_fits.append(fit)
    return calibration, final, {"inner_fits": ledger, "final_fits": final_fits}, {"c2v4": c2, "c4v4": c4}


def predict(rows, final2, final4, calibrations, *, batch=None):
    batch = base.base.base.neural.batches(rows) if batch is None else batch
    originals = np.asarray([r["original_choice"] for r in rows])
    costs, logits = {}, {}
    for precision, bundle in ((2, final2), (4, final4)):
        price, raw = bundle_prediction(rows, bundle, batch=batch)
        costs[f"f{precision}"] = price
        logits[f"f{precision}_glm"] = raw["glm"]
        for v in WAIT_VARIANTS:
            logits[f"f{precision}_raw_{v}"] = raw[v]
            for cal in CALIBRATIONS:
                for mode in ("risk", "policy"):
                    alpha = calibrations[cal][mode][v]["alpha"]
                    logits[f"f{precision}_{cal}_{mode}_{v}"] = base.blended(raw["glm"], raw[v], alpha)
    choices = {name: base.selected(costs[price], logits[gate], originals) for name, (price, gate) in VARIANTS.items()}
    x = base.base.base.centered_cost(batch[0], rows, final4["best"])
    costs["best"] = x@np.asarray(final4["best"]["coefficients"])
    return {"costs": costs, "wait_logits": logits, "choices": choices}
