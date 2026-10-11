"""Anchored fleet values above the causal peak and a learned wait constraint."""

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_occupancy_value as occupancy

METHODS = ("raw_absolute", "floor_absolute")
POLICIES = ("cost_only", "wait_feasible")
HEADS = (*paired.HEADS, "wait_delta_min")
CHECKPOINT_STEPS = (100, 300, 600, 900, 1200, 1500)


def floor_cost(row):
    return max(row["occupancy_context"]["historical_peak"]-12, 0)**2/12.


def fit_value(rows, method, *, steps=occupancy.STEPS, validation_rows=None):
    if method not in METHODS:
        raise ValueError("Unknown anchored fleet value method")
    raw = np.stack([occupancy.features(r, "occupancy_regret") for r in rows])
    labels = [{l["candidate"]: l for l in r["labels"]} for r in rows]
    # The absolute target fixes the offset that pair differences cannot identify.
    targets = np.asarray([[[l[i][h] for h in ("future_fleet_cost", *HEADS[1:])]
        for i in range(6)] for l in labels])
    center, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.
    a, b = np.triu_indices(6, 1)
    target_scale = np.r_[np.sqrt(np.mean(targets[..., 0]**2)),
        np.sqrt(np.mean((targets[:, a, 1:]-targets[:, b, 1:])**2, axis=(0, 1)))]
    target_scale = np.maximum(target_scale, 1e-6)
    actual_cost = np.asarray([[l[i]["actual_cost_delta"] for i in range(6)] for l in labels])
    native_scale = max(float(np.sqrt(np.mean(actual_cost**2))), 1e-6)
    x = torch.as_tensor((raw-center)/scale, dtype=torch.float32)
    y = torch.as_tensor(targets/target_scale, dtype=torch.float32)
    units = torch.as_tensor(target_scale[:2]/native_scale, dtype=torch.float32)
    actual = torch.as_tensor(actual_cost/native_scale, dtype=torch.float32)
    a, b = (torch.as_tensor(v, dtype=torch.long) for v in (a, b))
    ledger, checkpoints = [], []
    if validation_rows is not None:
        if {paired.scene_group(r) for r in rows} & {paired.scene_group(r) for r in validation_rows}:
            raise ValueError("Checkpoint validation must hold out complete training scenes")
        vraw = np.stack([occupancy.features(r, "occupancy_regret") for r in validation_rows])
        vl = [{l["candidate"]: l for l in r["labels"]} for r in validation_rows]
        vt = np.asarray([[[l[i][h] for h in ("future_fleet_cost", *HEADS[1:])]
            for i in range(6)] for l in vl])
        vx = torch.as_tensor((vraw-center)/scale, dtype=torch.float32)
        vy = torch.as_tensor(vt/target_scale, dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(occupancy.SEED)
        network = nn.Sequential(nn.Linear(occupancy.INPUT_DIM, occupancy.WIDTH), nn.Tanh(),
            nn.Linear(occupancy.WIDTH, 3))
        initial = [p.detach().clone() for p in network.parameters()]
        optimizer = torch.optim.AdamW(network.parameters(), lr=occupancy.LEARNING_RATE,
            weight_decay=occupancy.WEIGHT_DECAY)
        for step in range(steps):
            output = network(x)
            predicted = (torch.cat((softplus(output[..., :1]), output[..., 1:]), dim=-1)
                if method == "floor_absolute" else output)
            anchor = ((predicted[..., 0]-y[..., 0])**2).mean()
            pair_mse = ((predicted[:, a]-predicted[:, b]-(y[:, a]-y[:, b]))**2).mean()
            regret = occupancy.expected_regret(predicted[..., :2]*units, actual)
            loss = anchor+pair_mse+regret
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0 or (step+1) % 300 == 0 or step+1 == steps:
                ledger.append({"step": step+1, "absolute_fleet_mse": float(anchor.detach()),
                    "paired_mse": float(pair_mse.detach()), "native_regret": float(regret.detach())})
            if validation_rows is not None and (step+1 in CHECKPOINT_STEPS or step+1 == steps):
                with torch.no_grad():
                    voutput = network(vx)
                    vp = (torch.cat((softplus(voutput[..., :1]), voutput[..., 1:]), dim=-1)
                        if method == "floor_absolute" else voutput)
                    va = ((vp[..., 0]-vy[..., 0])**2).mean()
                    vm = ((vp[:, a]-vp[:, b]-(vy[:, a]-vy[:, b]))**2).mean()
                    checkpoints.append({"step": step+1, "validation_loss": float(va+vm),
                        "absolute_fleet_mse": float(va), "paired_mse": float(vm)})
        weights = {k: v.detach().numpy().tolist() for k, v in network.state_dict().items()}
        update_l2 = float(torch.sqrt(sum(((p.detach()-old)**2).sum() for p, old in zip(network.parameters(), initial))))
    model = {"method": method, "heads": list(HEADS), "center": center.tolist(), "scale": scale.tolist(),
        "target_scale": target_scale.tolist(), "regret_cost_scale": native_scale, "weights": weights,
        "steps": steps, "initialization_seed": occupancy.SEED,
        "parameter_count": sum(p.numel() for p in network.parameters()), "loss_ledger": ledger,
        "parameter_update_l2": update_l2, "training_states": len(rows), "pairwise_examples": len(rows)*15,
        "training_groups": [list(g) for g in sorted({paired.scene_group(r) for r in rows})],
        "target": "absolute_future_fleet_above_causal_peak_and_paired_nonfleet_wait_under_original_continuation"}
    if validation_rows is not None:
        model.update(checkpoint_ledger=checkpoints, validation_states=len(validation_rows),
            validation_groups=[list(g) for g in sorted({paired.scene_group(r) for r in validation_rows})])
    return model


def predict(row, model):
    x = (occupancy.features(row, "occupancy_regret")-model["center"])/model["scale"]
    w = {k: np.asarray(v) for k, v in model["weights"].items()}
    heads = np.tanh(x@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"]
    if model["method"] == "floor_absolute":
        heads[:, 0] = np.logaddexp(0., heads[:, 0])
    heads *= model["target_scale"]
    absolute_fleet = floor_cost(row)+heads[:, 0]
    delta = heads-heads[row["original_choice"]].copy()
    return {"predicted_heads": delta[:, :2].tolist(), "predicted_wait_delta_min": delta[:, 2].tolist(),
        "absolute_fleet_cost": absolute_fleet.tolist(), "causal_fleet_floor": floor_cost(row)}


def predict_value(row, model):
    return np.asarray(predict(row, model)["predicted_heads"])


def choose_action(row, model, policy):
    p = predict(row, model)
    cost = np.asarray(p["predicted_heads"]).sum(axis=1)
    if policy == "wait_feasible":
        cost = np.where(np.asarray(p["predicted_wait_delta_min"]) <= 0, cost, np.inf)
    elif policy != "cost_only":
        raise ValueError("Unknown anchored fleet decision policy")
    return paired.choice_from_cost(cost, row["original_choice"])
