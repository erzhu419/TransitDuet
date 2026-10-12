"""Fleet-only corrections, keeping a frozen critic's nonfleet and wait heads."""

import numpy as np
import torch
from torch import nn

from freq_hrl.domains.transit import native_floor_value as floor

METHODS = ("fleet_from_zero", "fleet_residual")
VARIANTS = ("cached_absolute", "physics_prior", *METHODS)


def prior_future(row):
    return np.maximum([c["fleet_cost"] for c in row["candidates"]], floor.floor_cost(row))-floor.floor_cost(row)


def fit_fleet(rows, method, *, steps=100):
    if method not in METHODS:
        raise ValueError("Unknown fleet-only fit")
    raw = np.stack([floor.features(r, "tail_summary") for r in rows])
    center, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.
    labels = [{l["candidate"]: l for l in r["labels"]} for r in rows]
    target = np.asarray([[ls[i]["future_fleet_cost"] for i in range(6)] for ls in labels])
    unit = max(float(np.sqrt(np.mean(target**2))), 1e-6)
    x = torch.as_tensor((raw-center)/scale, dtype=torch.float32)
    y = torch.as_tensor(target/unit, dtype=torch.float32)
    prior = np.stack([prior_future(r) for r in rows]) if method == "fleet_residual" else np.zeros_like(target)
    p = torch.as_tensor(prior/unit, dtype=torch.float32)
    a, b = np.triu_indices(6, 1)
    ledger = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(floor.occupancy.SEED)
        base = nn.Sequential(nn.Linear(floor.occupancy.INPUT_DIM, floor.occupancy.WIDTH), nn.Tanh(),
            nn.Linear(floor.occupancy.WIDTH, 3))
        expanded = nn.Linear(raw.shape[-1], floor.occupancy.WIDTH)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :floor.occupancy.INPUT_DIM].copy_(base[0].weight)
            expanded.bias.copy_(base[0].bias)
        network = nn.Sequential(expanded, nn.Tanh(), nn.Linear(floor.occupancy.WIDTH, 1))
        with torch.no_grad():
            network[2].weight.zero_()
            network[2].bias.zero_()
        initial = [v.detach().clone() for v in network.parameters()]
        optimizer = torch.optim.AdamW(network.parameters(), lr=floor.occupancy.LEARNING_RATE,
            weight_decay=floor.occupancy.WEIGHT_DECAY)
        for step in range(steps):
            # clamp_min has a gradient at zero: a zero prior can learn future risk.
            predicted = (p+network(x).squeeze(-1)).clamp_min(0.)
            anchor = ((predicted-y)**2).mean()
            pair = ((predicted[:, a]-predicted[:, b]-(y[:, a]-y[:, b]))**2).mean()
            optimizer.zero_grad(set_to_none=True)
            (anchor+pair).backward()
            optimizer.step()
            if step == 0 or step+1 == steps:
                ledger.append({"step": step+1, "absolute_fleet_mse": float(anchor.detach()),
                    "paired_fleet_mse": float(pair.detach())})
        weights = {k: v.detach().numpy().tolist() for k, v in network.state_dict().items()}
        update = float(torch.sqrt(sum((v.detach()-old).square().sum() for v, old in zip(network.parameters(), initial))))
    return {"method": method, "weights": weights, "center": center.tolist(), "scale": scale.tolist(),
        "target_scale": unit, "steps": steps, "parameter_count": sum(v.numel() for v in network.parameters()),
        "initialization_seed": floor.occupancy.SEED, "initialization": "shared_scalar_hidden_zero_fleet_projection",
        "objective": "absolute_fleet_MSE_plus_paired_fleet_MSE", "parameter_update_l2": update,
        "loss_ledger": ledger, "training_states": len(rows),
        "training_groups": sorted(map(list, {floor.paired.scene_group(r) for r in rows}))}


def predict_future(row, model):
    x = (floor.features(row, "tail_summary")-model["center"])/model["scale"]
    w = {k: np.asarray(v) for k, v in model["weights"].items()}
    output = (np.tanh(x@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"]).ravel()
    prior = prior_future(row) if model["method"] == "fleet_residual" else np.zeros(6)
    return np.maximum(prior+output*model["target_scale"], 0.)


def predict(row, reference, variant, models):
    result = floor.predict(row, reference)
    if variant == "cached_absolute":
        return result
    future = prior_future(row) if variant == "physics_prior" else predict_future(row, models[variant])
    absolute = floor.floor_cost(row)+future
    heads = np.asarray(result["predicted_heads"])
    heads[:, 0] = absolute-absolute[row["original_choice"]]
    result.update(predicted_heads=heads.tolist(), absolute_fleet_cost=absolute.tolist())
    return result


def evaluate(rows, reference, variant, models, policy):
    def prediction(row, _):
        return predict(row, reference, variant, models)
    def selection(row, _):
        p = prediction(row, None)
        costs = np.asarray(p["predicted_heads"]).sum(axis=1)
        if policy == "wait_feasible":
            costs = np.where(np.asarray(p["predicted_wait_delta_min"]) <= 0, costs, np.inf)
        elif policy != "cost_only":
            raise ValueError("Unknown fleet-only selector")
        return floor.paired.choice_from_cost(costs, row["original_choice"])
    result = floor.paired.evaluate_action_value(rows, None,
        predictor=lambda row, _: np.asarray(prediction(row, None)["predicted_heads"]), selector=selection)
    for row, observed in zip(rows, result):
        observed.update(prediction(row, None))
        actual = np.asarray([l["future_fleet_cost"]+floor.floor_cost(row) for l in row["labels"]])
        error = np.asarray(observed["absolute_fleet_cost"])-actual
        observed["fleet_absolute_mse"] = float(np.mean(error**2))
        observed["selected_fleet_delta_error"] = float(error[observed["selected"]]-error[row["original_choice"]])
    return result
