"""Causal nonlinear feasibility corrections over a frozen symmetric GLM."""

import numpy as np
from scipy.special import expit
import torch
from torch import nn
from torch.nn.functional import softplus

from freq_hrl.domains.transit import native_wait_feasibility as base

VARIANTS = ("nonlinear_pair", "state_conditioned_pair")
WIDTH, STEPS, SEED = 16, 600, 208
LEARNING_RATE, WEIGHT_DECAY = .003, .01
PARAMETERS = {"nonlinear_pair": 3201, "state_conditioned_pair": 3393}


def batches(rows):
    return (np.stack([base.wait.features(r, "conditional_wait") for r in rows]),
        np.stack([base.wait.condition_features(r) for r in rows]))


def inputs(rows, batch, reference, variant, context_center, context_scale):
    raw, context = batch
    originals = np.asarray([r["original_choice"] for r in rows])
    design = base.design(raw, originals, reference["center"], reference["scale"])
    x = design[..., :-1]
    if variant == "state_conditioned_pair":
        absolute = (context-context_center)/context_scale
        x = np.concatenate([x, np.broadcast_to(absolute[:, None], (*x.shape[:2], 12))], axis=-1)
    elif variant != "nonlinear_pair":
        raise ValueError("Unknown nonlinear wait representation")
    return x, design@np.asarray(reference["coefficients"])


def network(variant):
    first = nn.Linear(198, WIDTH, dtype=torch.float64)
    last = nn.Linear(WIDTH, 1, dtype=torch.float64)
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    if variant == "state_conditioned_pair":
        expanded = nn.Linear(210, WIDTH, dtype=torch.float64)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :198].copy_(first.weight)
            expanded.bias.copy_(first.bias)
        first = expanded
    return nn.Sequential(first, nn.Tanh(), last)


def mass_loss(logits, positive, negative, weights):
    return (weights*(positive*softplus(-logits)+negative*softplus(logits))).sum()/weights.sum()


def fit_wait(rows, masses, reference, variant, *, batch=None):
    batch = batches(rows) if batch is None else batch
    context_center, context_scale = batch[1].mean(axis=0), batch[1].std(axis=0)
    context_scale[context_scale < 1e-8] = 1.
    x, offset = inputs(rows, batch, reference, variant, context_center, context_scale)
    keys = [(r["root"], r["scene_seed"], r["probe_index"]) for r in rows]
    mask = np.arange(6)[None, :] != np.asarray([r["original_choice"] for r in rows])[:, None]
    positive = np.asarray([masses[k]["positive_wait_min"] for k in keys])[mask]
    negative = np.asarray([masses[k]["negative_wait_min"] for k in keys])[mask]
    counts = np.asarray([masses[k]["draw_count"] for k in keys])
    scale = max(float(np.average(positive+negative, weights=np.repeat(counts, 5))), 1e-6)
    if reference["false_safe_cost"] != 1. or reference["training_groups"] != sorted(map(list, {base.paired.scene_group(r) for r in rows})) or not np.isclose(
            scale, reference["target_scale_min"], atol=1e-12, rtol=0):
        raise ValueError("Nonlinear head changed the reference training groups or wait units")
    tx, toffset, positive, negative, weights = [torch.as_tensor(a, dtype=torch.float64)
        for a in (x[mask], offset[mask], positive/scale, negative/scale, np.repeat(counts, 5))]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        model = network(variant)
        initial = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        initial_loss = float(mass_loss(toffset, positive, negative, weights))
        ledger, hidden_gradient_max = [], 0.
        for step in range(STEPS):
            output = toffset+model(tx).squeeze(-1)
            loss = mass_loss(output, positive, negative, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            hidden_gradient_max = max(hidden_gradient_max, float(model[0].weight.grad.norm()))
            optimizer.step()
            if step == 0 or (step+1) % 100 == 0 or step+1 == STEPS:
                ledger.append({"step": step+1, "mass_loss_before_step": float(loss.detach())})
        with torch.no_grad():
            residual = model(tx).squeeze(-1)
            final_loss = float(mass_loss(toffset+residual, positive, negative, weights))
            residual_rms = float(residual.square().mean().sqrt())
            update = float(sum((p-old).square().sum() for p, old in zip(model.parameters(), initial)).sqrt())
            learned = {k: v.numpy().tolist() for k, v in model.state_dict().items()}
    return {"method": "frozen_glm_nonlinear_residual", "variant": variant, "false_safe_cost": 1.,
        "reference": reference, "weights": learned, "context_center": context_center.tolist(), "context_scale": context_scale.tolist(),
        "parameter_count": PARAMETERS[variant], "frozen_parameters": 199, "steps": STEPS, "initialization_seed": SEED,
        "width": WIDTH, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "training_states": len(rows), "training_groups": reference["training_groups"], "training_mean_states": int(np.sum(counts == 8)),
        "sample_weight_sum": float(counts.sum()), "target_scale_min": scale, "initial_mass_loss": initial_loss,
        "final_mass_loss": final_loss, "residual_logit_rms": residual_rms, "parameter_update_l2": update,
        "hidden_gradient_l2_max": hidden_gradient_max, "loss_ledger": ledger}


def logits_many(rows, model, *, batch=None):
    x, offset = inputs(rows, batches(rows) if batch is None else batch, model["reference"], model["variant"],
        model["context_center"], model["context_scale"])
    w = {k: np.asarray(v) for k, v in model["weights"].items()}
    return offset+(np.tanh(x@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"])[..., 0]


def predict_many(rows, model, *, batch=None):
    shares = expit(logits_many(rows, model, batch=batch))
    originals = np.asarray([r["original_choice"] for r in rows])
    shares[np.arange(len(rows)), originals] = 0.
    scores = shares-.5
    scores[np.arange(len(rows)), originals] = 0.
    return [{"predicted_harm_mass_share": p.tolist(), "predicted_feasibility_score": s.tolist()} for p, s in zip(shares, scores)]
