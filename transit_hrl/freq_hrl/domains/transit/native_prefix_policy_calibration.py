"""Prefix-and-future crossfit with selected-action cost/wait floors."""

import numpy as np

from freq_hrl.domains.transit import native_gap_calibration as base

WAIT_VARIANTS = base.WAIT_VARIANTS
NEW_GATES = tuple(f"prefix_{mode}_{v}" for mode in ("risk", "policy") for v in WAIT_VARIANTS)
GATES = (*base.GATES, *NEW_GATES)
VARIANTS = {**base.VARIANTS, **{f"ridge_{g}": ("ridge", g) for g in NEW_GATES}}


def prefix_folds(rows):
    groups = sorted({(r["root"], r["scene_seed"]) for r in rows})
    for held in groups:
        training = [i for i, r in enumerate(rows) if (r["root"], r["scene_seed"]) != held]
        validation = [i for i, r in enumerate(rows) if (r["root"], r["scene_seed"]) == held]
        if not training or len(validation) != 6:
            raise ValueError("Require complete six-callback prefixes and nonempty inner training")
        yield held, training, validation


def selected(costs, logits, originals):
    filtered = np.where(logits <= 0, costs, np.inf)
    filtered[np.arange(len(costs)), originals] = 0.
    choice = filtered.argmin(axis=1)
    # Match native exact identity-first ties, then ascending candidate index.
    tied = filtered[np.arange(len(costs)), originals] == filtered[np.arange(len(costs)), choice]
    choice[tied] = originals[tied]
    return choice


def blended(offset, raw, alpha):
    if alpha == 0.: return offset.copy()
    if alpha == 1.: return raw.copy()
    return offset+alpha*(raw-offset)


def physical_risk(data, raw, alpha):
    mask = np.arange(6)[None, :] != data["originals"][:, None]
    z = blended(data["offset"], raw, alpha)[mask]
    return float(np.mean(data["positive"][mask]*np.logaddexp(0., -z)+data["negative"][mask]*np.logaddexp(0., z)))


def outcomes(data, raw, alpha):
    choices = selected(data["costs"], blended(data["offset"], raw, alpha), data["originals"])
    values = data["values"][np.arange(len(choices)), choices]
    return {str(root): {"cost": float(values[data["roots"] == root, 0].mean()),
        "wait_min": float(values[data["roots"] == root, 3].mean())}
        for root in sorted(set(data["roots"]))}


def policy_shrinkage(data, raw, risk_alpha):
    offset, residual = data["offset"], raw-data["offset"]
    mask = (np.arange(6)[None, :] != data["originals"][:, None]) & (residual != 0)
    crossings = -offset[mask]/residual[mask]
    events = np.unique(np.r_[0., 1., crossings[(crossings >= 0) & (crossings <= 1)]])
    candidates = {0., 1., float(risk_alpha)}
    candidates.update(events.tolist())
    # Menus are constant between zero-logit crossings. Include boundary ties
    # and the convex-risk minimizer in each representable open interval.
    for low, high in zip(events[:-1], events[1:]):
        left, right = np.nextafter(low, high), np.nextafter(high, low)
        candidates.update((float(left), float(right), float((low+high)/2), float(np.clip(risk_alpha, left, right))))
    bounds = outcomes(data, raw, 1.)
    feasible = []
    for alpha in sorted(candidates):
        result = outcomes(data, raw, alpha)
        if all(result[root][name] <= bounds[root][name] for root in bounds for name in ("cost", "wait_min")):
            feasible.append((physical_risk(data, raw, alpha), -alpha, result))
    # Alpha=1 is the exact raw controller and always meets its own floors.
    risk, negative_alpha, result = min(feasible, key=lambda x: (x[0], x[1]))
    return {"alpha": -negative_alpha, "physical_mass_loss_min": risk, "raw_bounds_by_root": bounds,
        "selected_by_root": result, "risk_only_selected_by_root": outcomes(data, raw, risk_alpha),
        "candidate_alphas": len(candidates), "feasible_alphas": len(feasible),
        "zero_crossings": len(events)-2, "selected_action_constraints_met": True}


def collect(rows, panels, indices, *, batch=None):
    if len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError("Require exactly four outer training futures")
    batch = base.base.neural.batches(rows) if batch is None else batch
    pieces, ledger = [], []
    halves = (tuple(indices[:2]), tuple(indices[2:]))
    for group, training, validation in prefix_folds(rows):
        train, held = [rows[i] for i in training], [rows[i] for i in validation]
        train_batch, held_batch = tuple(b[training] for b in batch), tuple(b[validation] for b in batch)
        for direction in (0, 1):
            a, b = halves[direction], halves[1-direction]
            targets, masses = base.base.half_targets(train, panels, a)
            critic = base.base.ridge.fit_value(targets)
            models, fit = base.fit_inner_wait(targets, masses, batch=train_batch)
            logits = base.wait_logits(held, models, batch=held_batch)
            originals = np.asarray([r["original_choice"] for r in held])
            for z in logits.values(): z[np.arange(len(held)), originals] = 0.
            x = base.base.centered_cost(held_batch[0], held, critic)
            costs = ((x@np.asarray(critic["coefficients"]))*critic["target_scale"])[..., :2].sum(axis=-1)
            values = np.stack([panels[base.base.state_key(r)]["values"][list(b)].mean(axis=0) for r in held])
            _, held_masses = base.base.half_targets(held, panels, b)
            pieces.append({"originals": originals, "roots": np.asarray([r["root"] for r in held]),
                "costs": costs, "values": values, "offset": logits["glm"],
                **{v: logits[v] for v in WAIT_VARIANTS},
                **{name: np.asarray([held_masses[base.base.state_key(r)][field] for r in held])
                    for name, field in (("positive", "positive_wait_min"), ("negative", "negative_wait_min"))}})
            ledger.append({"held_group": list(group), "direction": direction,
                "training_draws": list(a), "validation_draws": list(b), "training_states": len(train),
                "held_states": len(held), "training_groups": critic["training_groups"],
                "wait_training_groups": models["glm"]["training_groups"], **fit})
        print(f"PREFIX_POLICY_INNER_COMPLETE {group[0]}/{group[1]}", flush=True)
    return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}, ledger


def calibrate(rows, panels, indices, *, batch=None):
    data, ledger = collect(rows, panels, indices, batch=batch)
    mask = np.arange(6)[None, :] != data["originals"][:, None]
    calibration = {"risk": {}, "policy": {}}
    for variant in WAIT_VARIANTS:
        raw = data[variant]
        risk = base.shrinkage(data["offset"][mask], (raw-data["offset"])[mask],
            data["positive"][mask], data["negative"][mask])
        calibration["risk"][variant] = risk
        calibration["policy"][variant] = policy_shrinkage(data, raw, risk["alpha"])
    return calibration, ledger, data


def predict(rows, models, *, batch=None):
    result = base.predict(rows, models, batch=batch)
    originals = np.asarray([r["original_choice"] for r in rows])
    for mode in ("risk", "policy"):
        for variant in WAIT_VARIANTS:
            name = f"prefix_{mode}_{variant}"
            alpha = models["prefix_calibration"][mode][variant]["alpha"]
            logits = blended(result["wait_logits"]["glm"], result["wait_logits"][variant], alpha)
            result["wait_logits"][name] = logits
            result["choices"][f"ridge_{name}"] = selected(result["costs"]["ridge"], logits, originals)
    return result
