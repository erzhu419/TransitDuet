"""Cost-gap distillation and train-future-only convex residual shrinkage."""

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

from freq_hrl.domains.transit import native_best_action_learning as base

STEPS = base.STEPS
WAIT_VARIANTS = base.WAIT_VARIANTS
GATES = ("glm", *WAIT_VARIANTS, *(f"shrunk_{v}" for v in WAIT_VARIANTS))
VARIANTS = {f"{price}_{gate}": (price, gate)
    for price in ("ridge", "best", "gap") for gate in GATES}
VARIANTS.update({f"{price}_unfiltered": (price, None) for price in ("best", "gap")})


def gap_loss(scores, costs, temperature):
    target = F.softmax(-costs/temperature, dim=-1)
    return F.kl_div(F.log_softmax(-scores/temperature, dim=-1), target, reduction="batchmean")


def fit_gap(rows, critic, *, batch=None):
    batch = base.neural.batches(rows) if batch is None else batch
    x = base.centered_cost(batch[0], rows, critic)
    heads = (x@np.asarray(critic["coefficients"]))*critic["target_scale"]
    offset = heads[..., :2].sum(axis=-1)
    costs = np.asarray([[l["actual_cost_delta"] for l in r["labels"]] for r in rows])
    mask = np.arange(6)[None, :] != np.asarray([r["original_choice"] for r in rows])[:, None]
    temperature = float(np.sqrt(np.mean(costs[mask]**2)))
    if temperature < 1e-8: temperature = 1.
    tx, toffset, target = [torch.as_tensor(a, dtype=torch.float64) for a in (x, offset, costs)]
    with torch.random.fork_rng(devices=[]):
        model = nn.Linear(98, 1, bias=False, dtype=torch.float64)
        nn.init.zeros_(model.weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base.neural.LEARNING_RATE,
            weight_decay=base.neural.WEIGHT_DECAY)
        def loss():
            return gap_loss(toffset+temperature*model(tx).squeeze(-1), target, temperature)
        initial = float(loss().detach())
        for _ in range(STEPS):
            objective = loss()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        final = float(loss().detach())
        coefficients = model.weight.detach().numpy()[0]
    if not np.isfinite(final): raise ValueError("Nonfinite gap objective")
    return {"coefficients": coefficients.tolist(), "temperature_cost": temperature}, {
        "parameter_count": 98, "steps": STEPS, "initial_loss": initial, "final_loss": final,
        "parameter_update_l2": float(np.linalg.norm(coefficients)), "temperature_cost": temperature}


def wait_logits(rows, models, *, batch=None):
    batch = base.neural.batches(rows) if batch is None else batch
    originals = [r["original_choice"] for r in rows]
    reference = models["glm"]
    x = base.wait.design(batch[0], originals, reference["center"], reference["scale"])
    logits = {"glm": x@np.asarray(reference["coefficients"])}
    logits["linear"] = logits["glm"]+x@np.asarray(models["linear"]["coefficients"])
    for variant in WAIT_VARIANTS[1:]:
        logits[variant] = base.neural.logits_many(rows, models[variant], batch=batch)
    return logits


def fit_inner_wait(rows, masses, *, batch=None):
    batch = base.neural.batches(rows) if batch is None else batch
    reference = base.wait.fit_wait(rows, masses, "symmetric_feasibility")
    originals = np.asarray([r["original_choice"] for r in rows])
    mask = np.arange(6)[None, :] != originals[:, None]
    x = base.wait.design(batch[0], originals, reference["center"], reference["scale"])[mask]
    offset = x@np.asarray(reference["coefficients"])
    keys = [base.state_key(r) for r in rows]
    positive, negative = [np.asarray([masses[k][name] for k in keys])[mask]/reference["target_scale_min"]
        for name in ("positive_wait_min", "negative_wait_min")]
    tx, toffset, positive, negative, weights = [torch.as_tensor(a, dtype=torch.float64)
        for a in (x, offset, positive, negative, np.ones(mask.sum()))]
    with torch.random.fork_rng(devices=[]):
        model = nn.Linear(199, 1, bias=False, dtype=torch.float64)
        nn.init.zeros_(model.weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=base.neural.LEARNING_RATE,
            weight_decay=base.neural.WEIGHT_DECAY)
        def loss():
            return base.neural.mass_loss(toffset+model(tx).squeeze(-1), positive, negative, weights)
        initial = float(loss().detach())
        for _ in range(STEPS):
            value = loss()
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
        final = float(loss().detach())
        coefficients = model.weight.detach().numpy()[0].tolist()
    models = {"glm": reference, "linear": {"coefficients": coefficients}}
    fits = {"linear": {"parameter_count": 199, "steps": STEPS, "initial_loss": initial,
        "final_loss": final, "parameter_update_l2": float(np.linalg.norm(coefficients))}}
    # Existing nonlinear fits use the same draw-mass objective and optimizer.
    for variant in WAIT_VARIANTS[1:]:
        fitted = base.neural.fit_wait(rows, masses, reference, variant, batch=batch)
        models[variant] = fitted
        fits[variant] = {"parameter_count": fitted["parameter_count"], "steps": fitted["steps"],
            "initial_loss": fitted["initial_mass_loss"], "final_loss": fitted["final_mass_loss"],
            "parameter_update_l2": fitted["parameter_update_l2"]}
    return models, {"fits": fits, "glm_iterations": reference["optimizer_iterations"]}


def shrinkage(offset, residual, positive, negative):
    def loss(alpha):
        z = offset+alpha*residual
        return float(np.mean(positive*np.logaddexp(0., -z)+negative*np.logaddexp(0., z)))
    def gradient(alpha):
        return float(np.mean(((positive+negative)*expit(offset+alpha*residual)-positive)*residual))
    g0, g1 = gradient(0.), gradient(1.)
    alpha = 0. if g0 >= 0 else (1. if g1 <= 0 else float(brentq(gradient, 0., 1.)))
    return {"alpha": alpha, "reference_loss_min": loss(0.), "raw_loss_min": loss(1.),
        "shrunk_loss_min": loss(alpha), "derivative_at_zero": g0, "derivative_at_one": g1,
        "candidate_cases": int(offset.size)}


def calibrate(rows, panels, indices, *, batch=None):
    if len(indices) != 4 or len(set(indices)) != 4:
        raise ValueError("Calibrate within exactly four outer training futures")
    batch = base.neural.batches(rows) if batch is None else batch
    mask = np.arange(6)[None, :] != np.asarray([r["original_choice"] for r in rows])[:, None]
    pieces, ledger = [], []
    halves = (tuple(indices[:2]), tuple(indices[2:]))
    for direction in (0, 1):
        train, held = halves[direction], halves[1-direction]
        targets, masses = base.half_targets(rows, panels, train)
        models, fit = fit_inner_wait(targets, masses, batch=batch)
        prediction = wait_logits(rows, models, batch=batch)
        _, held_masses = base.half_targets(rows, panels, held)
        positive, negative = [np.asarray([held_masses[base.state_key(r)][name] for r in rows])[mask]
            for name in ("positive_wait_min", "negative_wait_min")]
        pieces.append({"offset": prediction["glm"][mask], "positive": positive, "negative": negative,
            **{v: (prediction[v]-prediction["glm"])[mask] for v in WAIT_VARIANTS}})
        ledger.append({"training_draws": list(train), "validation_draws": list(held), **fit})
    values = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    calibration = {v: shrinkage(values["offset"], values[v], values["positive"], values["negative"])
        for v in WAIT_VARIANTS}
    return calibration, ledger


def fit(rows, panels, indices, *, batch=None):
    batch = base.neural.batches(rows) if batch is None else batch
    targets, masses = base.half_targets(rows, panels, indices)
    models, fit = base.fit_heads(targets, masses, batch=batch)
    models["gap"], fit["fits"]["gap"] = fit_gap(targets, models["ridge"], batch=batch)
    models["calibration"], fit["inner_fits"] = calibrate(rows, panels, indices, batch=batch)
    fit["calibration"] = models["calibration"]
    fit.update(supervised_decision_updates=base.STEPS+STEPS, wait_residual_updates=3*base.STEPS+6*STEPS,
        glm_fits=3, analytic_cost_fits=1, calibration_scalar_fits=3)
    return models, fit


def predict(rows, models, *, batch=None):
    batch = base.neural.batches(rows) if batch is None else batch
    control = base.predict(rows, models, batch=batch)
    x = base.centered_cost(batch[0], rows, models["ridge"])
    costs = {**control["costs"], "gap": control["costs"]["ridge"]+
        models["gap"]["temperature_cost"]*(x@np.asarray(models["gap"]["coefficients"]))}
    logits = control["wait_logits"]
    for variant in WAIT_VARIANTS:
        logits[f"shrunk_{variant}"] = logits["glm"]+models["calibration"][variant]["alpha"]*(logits[variant]-logits["glm"])
    originals = [r["original_choice"] for r in rows]
    choices = {name: np.asarray([base.paired.choice_from_cost(np.where(logits[gate][i] <= 0, costs[price][i], np.inf)
        if gate is not None else costs[price][i], original) for i, original in enumerate(originals)])
        for name, (price, gate) in VARIANTS.items()}
    return {"costs": costs, "wait_logits": logits, "choices": choices}
