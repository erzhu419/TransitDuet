"""Smooth logit-bounded wait corrections with training-only functional shrinkage."""

import numpy as np
from scipy.special import expit
import torch

from freq_hrl.domains.transit import native_early_stop_wait as stopping

neural = stopping.neural
BOUND = 1.
FUNCTIONAL_RIDGES = {"bounded_only": 0., "shrunk_bounded": .1}


def correction(raw):
    return BOUND*torch.tanh(raw/BOUND)


def penalty(residual, positive, negative, weights, ridge):
    mass_weights = weights*(positive+negative)
    return .5*ridge*(mass_weights*residual.square()).sum()/mass_weights.sum()


def logits_many(rows, model, *, batch=None):
    x, offset = neural.inputs(rows, neural.batches(rows) if batch is None else batch, model["reference"],
        "state_conditioned_pair", model["context_center"], model["context_scale"])
    w = {k: np.asarray(v) for k, v in model["weights"].items()}
    raw = (np.tanh(x@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"])[..., 0]
    return offset+model["residual_bound"]*np.tanh(raw/model["residual_bound"])


def predict_many(rows, model, *, batch=None):
    shares = expit(logits_many(rows, model, batch=batch))
    originals = np.asarray([r["original_choice"] for r in rows])
    shares[np.arange(len(rows)), originals] = 0.
    scores = shares-.5
    scores[np.arange(len(rows)), originals] = 0.
    return [{"predicted_harm_mass_share": p.tolist(), "predicted_feasibility_score": s.tolist()} for p, s in zip(shares, scores)]


def validation_loss(rows, masses, model, batch):
    logits = logits_many(rows, model, batch=batch)
    mask = np.arange(6)[None, :] != np.asarray([r["original_choice"] for r in rows])[:, None]
    p = np.asarray([masses[stopping.key(r)]["positive_wait_min"] for r in rows])[mask]
    n = np.asarray([masses[stopping.key(r)]["negative_wait_min"] for r in rows])[mask]
    weights = np.repeat([masses[stopping.key(r)]["draw_count"] for r in rows], 5)
    return {"loss_sum_min": float(weights@(p*np.logaddexp(0, -logits[mask])+n*np.logaddexp(0, logits[mask]))),
        "mass_sum_min": float(weights@(p+n))}


def fit_path(rows, masses, reference, variant, checkpoints, *, validation=None, batch=None):
    ridge = FUNCTIONAL_RIDGES[variant]
    batch = neural.batches(rows) if batch is None else batch
    center, std = batch[1].mean(axis=0), batch[1].std(axis=0)
    std[std < 1e-8] = 1.
    x, offset = neural.inputs(rows, batch, reference, "state_conditioned_pair", center, std)
    mask = np.arange(6)[None, :] != np.asarray([r["original_choice"] for r in rows])[:, None]
    p = np.asarray([masses[stopping.key(r)]["positive_wait_min"] for r in rows])[mask]
    n = np.asarray([masses[stopping.key(r)]["negative_wait_min"] for r in rows])[mask]
    counts = np.asarray([masses[stopping.key(r)]["draw_count"] for r in rows])
    scale = max(float(np.average(p+n, weights=np.repeat(counts, 5))), 1e-6)
    groups = sorted(map(list, {neural.base.paired.scene_group(r) for r in rows}))
    if reference["training_groups"] != groups or reference["false_safe_cost"] != 1. or not np.isclose(
            scale, reference["target_scale_min"], atol=1e-12, rtol=0):
        raise ValueError("Bounded residual requires its own training-only symmetric reference")
    tx, toffset, p, n, weights = [torch.as_tensor(a, dtype=torch.float64) for a in
        (x[mask], offset[mask], p/scale, n/scale, np.repeat(counts, 5))]
    validation_batch = None if validation is None else neural.batches(validation)
    curve, ledger, gradient = [], [], 0.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(neural.SEED)
        net = neural.network("state_conditioned_pair")
        initial = [v.detach().clone() for v in net.parameters()]
        optimizer = torch.optim.AdamW(net.parameters(), lr=neural.LEARNING_RATE, weight_decay=neural.WEIGHT_DECAY)
        initial_loss = float(neural.mass_loss(toffset, p, n, weights))
        for step in range(max(checkpoints)+1):
            if step:
                residual = correction(net(tx).squeeze(-1))
                objective = neural.mass_loss(toffset+residual, p, n, weights)+penalty(residual, p, n, weights, ridge)
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                gradient = max(gradient, float(net[0].weight.grad.norm()))
                optimizer.step()
            if step not in checkpoints:
                continue
            with torch.no_grad():
                residual = correction(net(tx).squeeze(-1))
                loss = float(neural.mass_loss(toffset+residual, p, n, weights))
                shrink = float(penalty(residual, p, n, weights, ridge))
                model = {"method": "bounded_glm_nonlinear_residual", "variant": "state_conditioned_pair", "correction": variant,
                    "false_safe_cost": 1., "residual_bound": BOUND, "functional_ridge": ridge,
                    "reference": reference, "weights": {k: v.numpy().tolist() for k, v in net.state_dict().items()},
                    "context_center": center.tolist(), "context_scale": std.tolist(), "parameter_count": 3393,
                    "frozen_parameters": 199, "steps": step, "initialization_seed": neural.SEED, "width": neural.WIDTH,
                    "learning_rate": neural.LEARNING_RATE, "weight_decay": neural.WEIGHT_DECAY,
                    "training_states": len(rows), "training_groups": groups, "training_mean_states": int(np.sum(counts == 8)),
                    "sample_weight_sum": float(counts.sum()), "target_scale_min": scale,
                    "initial_mass_loss": initial_loss, "final_mass_loss": loss, "final_functional_penalty": shrink,
                    "final_objective": loss+shrink, "residual_logit_rms": float(residual.square().mean().sqrt()),
                    "residual_logit_max_abs": float(residual.abs().max()),
                    "parameter_update_l2": float(sum((v-old).square().sum() for v, old in zip(net.parameters(), initial)).sqrt()),
                    "hidden_gradient_l2_max": gradient}
            ledger.append({"step": step, "mass_loss_after_step": loss, "functional_penalty": shrink, "objective": loss+shrink})
            if validation is not None:
                curve.append({"step": step, **validation_loss(validation, masses, model, validation_batch)})
    return {**model, "loss_ledger": ledger}, curve


def fit_wait(rows, masses, reference, groups, mean_keys):
    partition = stopping.grouping.partition_groups(groups, {neural.base.paired.scene_group(r) for r in rows if stopping.key(r) in mean_keys})
    curves, inner = {v: [] for v in FUNCTIONAL_RIDGES}, {v: [] for v in FUNCTIONAL_RIDGES}
    scored = []
    for held_groups in partition:
        held = set(map(tuple, held_groups))
        train = [r for r in rows if neural.base.paired.scene_group(r) not in held]
        validation = [r for r in rows if neural.base.paired.scene_group(r) in held and stopping.key(r) in mean_keys]
        glm = neural.base.fit_wait(train, masses, "symmetric_feasibility")
        batch = neural.batches(train)
        scored.extend(stopping.key(r) for r in validation)
        for variant in FUNCTIONAL_RIDGES:
            model, curve = fit_path(train, masses, glm, variant, stopping.CHECKPOINTS, validation=validation, batch=batch)
            curves[variant].append(curve)
            inner[variant].append({"held_groups": held_groups, "validation_mean_states": len(validation),
                "glm_fit": stopping.metadata(glm), "neural_fit": stopping.metadata(model), "curve": curve})
    expected = {stopping.key(r) for r in rows} & mean_keys
    if len(scored) != len(expected) or set(scored) != expected:
        raise ValueError("Bounded inner folds lost or duplicated a mean8 state")
    models, selections = {}, {}
    batch = neural.batches(rows)
    for variant in FUNCTIONAL_RIDGES:
        step, pooled = stopping.select_checkpoint(curves[variant])
        models[variant], _ = fit_path(rows, masses, reference, variant, (step,), batch=batch)
        selections[variant] = {"selected_step": step, "partition": partition, "inner_fits": inner[variant], "pooled_curve": pooled,
            "validation_mean_states": len(scored), "inner_glm_iterations": sum(f["glm_fit"]["optimizer_iterations"] for f in inner[variant]),
            "inner_adamw_steps": len(partition)*max(stopping.CHECKPOINTS), "outer_adamw_steps": step}
    return models, selections
