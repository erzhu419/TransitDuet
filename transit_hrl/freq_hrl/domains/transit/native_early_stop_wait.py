"""Grouped training-only checkpoint selection for the fixed nonlinear wait head."""

import numpy as np
import torch

from freq_hrl.domains.transit import native_nonlinear_wait as neural
from freq_hrl.domains.transit import native_wait_calibration as grouping

CHECKPOINTS = (0, 50, 100, 200, 400, 600)
VARIANT = "state_conditioned_pair"


def key(row):
    return row["root"], row["scene_seed"], row["probe_index"]


def metadata(model):
    return {k: v for k, v in model.items() if k not in
        ("weights", "reference", "context_center", "context_scale", "coefficients", "center", "scale")}


def validation_loss(rows, masses, model, batch=None):
    logits = neural.logits_many(rows, model, batch=batch)
    mask = np.arange(6)[None, :] != np.asarray([r["original_choice"] for r in rows])[:, None]
    positive = np.asarray([masses[key(r)]["positive_wait_min"] for r in rows])[mask]
    negative = np.asarray([masses[key(r)]["negative_wait_min"] for r in rows])[mask]
    weights = np.repeat([masses[key(r)]["draw_count"] for r in rows], 5)
    loss = positive*np.logaddexp(0, -logits[mask])+negative*np.logaddexp(0, logits[mask])
    return {"loss_sum_min": float(weights@loss), "mass_sum_min": float(weights@(positive+negative))}


def fit_path(rows, masses, reference, checkpoints, *, validation=None, batch=None):
    batch = neural.batches(rows) if batch is None else batch
    center, std = batch[1].mean(axis=0), batch[1].std(axis=0)
    std[std < 1e-8] = 1.
    x, offset = neural.inputs(rows, batch, reference, VARIANT, center, std)
    mask = np.arange(6)[None, :] != np.asarray([r["original_choice"] for r in rows])[:, None]
    positive = np.asarray([masses[key(r)]["positive_wait_min"] for r in rows])[mask]
    negative = np.asarray([masses[key(r)]["negative_wait_min"] for r in rows])[mask]
    counts = np.asarray([masses[key(r)]["draw_count"] for r in rows])
    scale = max(float(np.average(positive+negative, weights=np.repeat(counts, 5))), 1e-6)
    groups = sorted(map(list, {neural.base.paired.scene_group(r) for r in rows}))
    if reference["false_safe_cost"] != 1. or reference["training_groups"] != groups or not np.isclose(
            scale, reference["target_scale_min"], atol=1e-12, rtol=0):
        raise ValueError("Checkpoint path requires its own training-only symmetric GLM")
    tensors = [torch.as_tensor(a, dtype=torch.float64) for a in
        (x[mask], offset[mask], positive/scale, negative/scale, np.repeat(counts, 5))]
    tx, toffset, positive, negative, weights = tensors
    validation_batch = None if validation is None else neural.batches(validation)
    curve, ledger, gradient = [], [], 0.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(neural.SEED)
        net = neural.network(VARIANT)
        initial = [p.detach().clone() for p in net.parameters()]
        optimizer = torch.optim.AdamW(net.parameters(), lr=neural.LEARNING_RATE, weight_decay=neural.WEIGHT_DECAY)
        initial_loss = float(neural.mass_loss(toffset, positive, negative, weights))
        for step in range(max(checkpoints)+1):
            if step:
                loss = neural.mass_loss(toffset+net(tx).squeeze(-1), positive, negative, weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient = max(gradient, float(net[0].weight.grad.norm()))
                optimizer.step()
            if step not in checkpoints:
                continue
            with torch.no_grad():
                residual = net(tx).squeeze(-1)
                final_loss = float(neural.mass_loss(toffset+residual, positive, negative, weights))
                model = {"method": "frozen_glm_nonlinear_residual", "variant": VARIANT, "false_safe_cost": 1.,
                    "reference": reference, "weights": {k: v.numpy().tolist() for k, v in net.state_dict().items()},
                    "context_center": center.tolist(), "context_scale": std.tolist(),
                    "parameter_count": neural.PARAMETERS[VARIANT], "frozen_parameters": 199,
                    "steps": step, "initialization_seed": neural.SEED, "width": neural.WIDTH,
                    "learning_rate": neural.LEARNING_RATE, "weight_decay": neural.WEIGHT_DECAY,
                    "training_states": len(rows), "training_groups": groups, "training_mean_states": int(np.sum(counts == 8)),
                    "sample_weight_sum": float(counts.sum()), "target_scale_min": scale,
                    "initial_mass_loss": initial_loss, "final_mass_loss": final_loss,
                    "residual_logit_rms": float(residual.square().mean().sqrt()),
                    "parameter_update_l2": float(sum((p-old).square().sum() for p, old in zip(net.parameters(), initial)).sqrt()),
                    "hidden_gradient_l2_max": gradient}
            ledger.append({"step": step, "mass_loss_after_step": final_loss})
            if validation is not None:
                curve.append({"step": step, **validation_loss(validation, masses, model, validation_batch)})
    return {**model, "loss_ledger": ledger}, curve


def select_checkpoint(curves):
    pooled = []
    for step in CHECKPOINTS:
        points = [next(p for p in curve if p["step"] == step) for curve in curves]
        mass = sum(p["mass_sum_min"] for p in points)
        if mass <= 0:
            raise ValueError("Checkpoint validation requires nonzero mean8 wait signal")
        pooled.append({"step": step, "mass_weighted_nats": sum(p["loss_sum_min"] for p in points)/mass})
    return min(pooled, key=lambda p: (p["mass_weighted_nats"], p["step"]))["step"], pooled


def fit_wait(rows, masses, reference, groups, mean_keys):
    partition = grouping.partition_groups(groups, {neural.base.paired.scene_group(r) for r in rows if key(r) in mean_keys})
    curves, inner, scored = [], [], []
    for held_groups in partition:
        held = set(map(tuple, held_groups))
        train = [r for r in rows if neural.base.paired.scene_group(r) not in held]
        validation = [r for r in rows if neural.base.paired.scene_group(r) in held and key(r) in mean_keys]
        # The outer GLM has seen these validation labels; fit a new inner GLM instead.
        glm = neural.base.fit_wait(train, masses, "symmetric_feasibility")
        model, curve = fit_path(train, masses, glm, CHECKPOINTS, validation=validation)
        scored.extend(key(r) for r in validation)
        curves.append(curve)
        inner.append({"held_groups": held_groups, "validation_mean_states": len(validation),
            "glm_fit": metadata(glm), "neural_fit": metadata(model), "curve": curve})
    expected = {key(r) for r in rows} & mean_keys
    if len(scored) != len(expected) or set(scored) != expected:
        raise ValueError("Inner scene folds lost or duplicated mean8 validation states")
    step, pooled = select_checkpoint(curves)
    model, _ = fit_path(rows, masses, reference, (step,))
    return model, {"selected_step": step, "partition": partition, "inner_fits": inner, "pooled_curve": pooled,
        "validation_mean_states": len(scored), "inner_glm_iterations": sum(f["glm_fit"]["optimizer_iterations"] for f in inner),
        "inner_adamw_steps": len(inner)*max(CHECKPOINTS), "outer_adamw_steps": step}
