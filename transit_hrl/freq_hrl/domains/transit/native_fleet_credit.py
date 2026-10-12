"""Separate reference-action fleet risk from candidate-relative credit."""

import numpy as np
import torch
from torch import nn

from freq_hrl.domains.transit import native_fleet_residual as previous

floor = previous.floor
METHODS = ("shared_pair_scaled", "separated_shared_scale", "separated_pair_scaled")
VARIANTS = ("cached_absolute", "fleet_from_zero", *METHODS)
PARAMETERS = {m: 1601 for m in METHODS}


def initialize(input_dim, method):
    base = nn.Sequential(nn.Linear(floor.occupancy.INPUT_DIM, 16), nn.Tanh(), nn.Linear(16, 3))
    hidden = nn.Linear(input_dim, 16)
    with torch.no_grad():
        hidden.weight.zero_()
        hidden.weight[:, :floor.occupancy.INPUT_DIM].copy_(base[0].weight)
        hidden.bias.copy_(base[0].bias)
    if method == "shared_pair_scaled":
        models = nn.ModuleDict({"shared": nn.Sequential(hidden, nn.Tanh(), nn.Linear(16, 1))})
    else:
        models = nn.ModuleDict()
        for name, part in (("risk", slice(0, 8)), ("contrast", slice(8, 16))):
            layer = nn.Linear(input_dim, 8)
            with torch.no_grad():
                layer.weight.copy_(hidden.weight[part])
                layer.bias.copy_(hidden.bias[part])
            models[name] = nn.Sequential(layer, nn.Tanh(), nn.Linear(8, 1, bias=name == "risk"))
    with torch.no_grad():
        for model in models.values():
            model[2].weight.zero_()
            if model[2].bias is not None:
                model[2].bias.zero_()
    return models


def fit_fleet(rows, method, *, steps=100):
    if method not in METHODS:
        raise ValueError("Unknown fleet-credit method")
    raw = np.stack([floor.features(r, "tail_summary") for r in rows])
    center, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.
    labels = [{l["candidate"]: l for l in r["labels"]} for r in rows]
    target = np.asarray([[ls[i]["future_fleet_cost"] for i in range(6)] for ls in labels])
    a, b = np.triu_indices(6, 1)
    absolute_unit = max(float(np.sqrt(np.mean(target**2))), 1e-6)
    paired_unit = max(float(np.sqrt(np.mean((target[:, a]-target[:, b])**2))), 1e-6)
    loss_unit = absolute_unit if method == "separated_shared_scale" else paired_unit
    x = torch.as_tensor((raw-center)/scale, dtype=torch.float32)
    y = torch.as_tensor(target/absolute_unit, dtype=torch.float32)
    batch, original = torch.arange(len(rows)), torch.tensor([r["original_choice"] for r in rows])
    ledger = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(floor.occupancy.SEED)
        models = initialize(raw.shape[-1], method)
        parameters = list(models.parameters())
        initial = [v.detach().clone() for v in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=floor.occupancy.LEARNING_RATE,
            weight_decay=floor.occupancy.WEIGHT_DECAY)
        for step in range(steps):
            if method == "shared_pair_scaled":
                predicted = models["shared"](x).squeeze(-1).clamp_min(0.)
                anchor = ((predicted-y)**2).mean()
            else:
                risk = models["risk"](x[batch, original]).squeeze(-1).clamp_min(0.)
                effect = models["contrast"](x).squeeze(-1)
                effect = effect-effect[batch, original, None]
                # The pair target trains only action credit, not reference-state risk.
                predicted = (risk.detach()[:, None]+effect).clamp_min(0.)
                anchor = ((risk-y[batch, original])**2).mean()
            pair = (((predicted[:, a]-predicted[:, b]-(y[:, a]-y[:, b]))*absolute_unit/loss_unit)**2).mean()
            optimizer.zero_grad(set_to_none=True)
            (anchor+pair).backward()
            optimizer.step()
            if step == 0 or step+1 == steps:
                ledger.append({"step": step+1, "absolute_anchor_mse": float(anchor.detach()),
                    "paired_fleet_mse": float(pair.detach())})
        weights = {name: {k: v.detach().numpy().tolist() for k, v in model.state_dict().items()}
            for name, model in models.items()}
        update = float(torch.sqrt(sum((v.detach()-old).square().sum() for v, old in zip(parameters, initial))))
    return {"method": method, "weights": weights, "center": center.tolist(), "scale": scale.tolist(),
        "target_scale": absolute_unit, "paired_target_scale": paired_unit, "paired_loss_scale": loss_unit,
        "steps": steps, "parameter_count": sum(v.numel() for v in parameters), "initialization_seed": floor.occupancy.SEED,
        "initialization": "same16_hidden_rows_zero_output_shared16_or_disjoint8plus8",
        "baseline_pair_gradient": method == "shared_pair_scaled", "parameter_update_l2": update,
        "loss_ledger": ledger, "training_states": len(rows),
        "training_groups": sorted(map(list, {floor.paired.scene_group(r) for r in rows}))}


def forward_numpy(inputs, weights):
    w = {k: np.asarray(v) for k, v in weights.items()}
    return (np.tanh(inputs@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w.get("2.bias", 0.)).ravel()


def predict_future(row, model):
    if model["method"] == "fleet_from_zero":
        return previous.predict_future(row, model)
    x = (floor.features(row, "tail_summary")-model["center"])/model["scale"]
    if model["method"] == "shared_pair_scaled":
        future = forward_numpy(x, model["weights"]["shared"])
    else:
        risk = max(float(forward_numpy(x[row["original_choice"]], model["weights"]["risk"])[0]), 0.)
        effect = forward_numpy(x, model["weights"]["contrast"])
        future = risk+(effect-effect[row["original_choice"]])
    return np.maximum(future, 0.)*model["target_scale"]


def predict(row, reference, variant, models):
    result = floor.predict(row, reference)
    if variant != "cached_absolute":
        absolute = floor.floor_cost(row)+predict_future(row, models[variant])
        heads = np.asarray(result["predicted_heads"])
        heads[:, 0] = absolute-absolute[row["original_choice"]]
        result.update(predicted_heads=heads.tolist(), absolute_fleet_cost=absolute.tolist())
    return result


def evaluate(rows, reference, variant, models, policy):
    def prediction(row):
        return predict(row, reference, variant, models)
    def selection(row, _):
        p = prediction(row)
        costs = np.asarray(p["predicted_heads"]).sum(axis=1)
        if policy == "wait_feasible":
            costs = np.where(np.asarray(p["predicted_wait_delta_min"]) <= 0, costs, np.inf)
        elif policy != "cost_only":
            raise ValueError("Unknown fleet-credit selector")
        return floor.paired.choice_from_cost(costs, row["original_choice"])
    result = floor.paired.evaluate_action_value(rows, None,
        predictor=lambda row, _: np.asarray(prediction(row)["predicted_heads"]), selector=selection)
    a, b = np.triu_indices(6, 1)
    for row, observed in zip(rows, result):
        observed.update(prediction(row))
        actual = np.asarray([l["future_fleet_cost"]+floor.floor_cost(row) for l in row["labels"]])
        error = np.asarray(observed["absolute_fleet_cost"])-actual
        observed.update(fleet_absolute_mse=float(np.mean(error**2)), fleet_pair_mse=float(np.mean((error[a]-error[b])**2)),
            original_fleet_mse=float(error[row["original_choice"]]**2),
            selected_fleet_delta_error=float(error[observed["selected"]]-error[row["original_choice"]]))
    return result
