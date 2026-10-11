"""Causal occupancy representation and decision-aware paired cost learning."""

import numpy as np
import torch
from torch import nn

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit.native_supply_plan import occupancy_peak

METHODS = {
    "base_mse": (False, False), "occupancy_mse": (True, False),
    "base_regret": (False, True), "occupancy_regret": (True, True),
}
STEPS, WIDTH, SEED = 1500, 16, 178
LEARNING_RATE, WEIGHT_DECAY = .003, .01
OCCUPANCY_NAMES = ["historical_excess", "active_headroom", "same_queue_concentration", "other_queue_concentration"]
OCCUPANCY_NAMES += [f"{direction}_{name}" for direction in ("same", "other") for name in
    ("remaining_mean", "remaining_std", "age_mean", "load_mean", "speed_mean", "hold_mean", "dwell_mean")]
OCCUPANCY_NAMES += ["active_at_first", "active_at_departures_max", "active_at_departures_mean",
    "local_peak_excess", "prefix_future_fleet_cost", "full_future_fleet_cost"]
INPUT_DIM = 72+len(OCCUPANCY_NAMES)


def mean(values):
    return float(np.mean(values)) if len(values) else 0.


def occupancy_features(row):
    context = row["occupancy_context"]
    buses, floor = context["active_buses"], context["historical_peak"]
    floor_cost = max(floor-12, 0)**2/12.
    state = [(floor-12)/12., (floor-len(buses))/12.]
    for direction in (row["direction"], not row["direction"]):
        queues = [q[2] for q in context["queue_by_station"] if q[1] == direction]
        state.append(max(queues, default=0)/max(sum(queues), 1))
    for direction in (row["direction"], not row["direction"]):
        group = [b for b in buses if b["direction"] == direction]
        remaining = [b["predicted_remaining_s"] for b in group]
        state.extend([mean(remaining)/1800., float(np.std(remaining))/1800. if group else 0.,
            mean([b["elapsed_s"] for b in group])/1800., mean([b["load_fraction"] for b in group]),
            mean([b["speed_mps"] for b in group])/10., mean([b["holding_s"] for b in group])/60.,
            mean([b["dwelling_s"] for b in group])/60.])
    duration = mean([b["mean_total_s"] for b in buses if b["direction"] == row["direction"]])
    active_ends = row["time_s"]+np.asarray([b["predicted_remaining_s"] for b in buses])
    features = []
    for candidate in row["candidates"]:
        c = candidate["commitment"]
        planned = np.asarray(c["prefix_planned_s"])
        active = (active_ends[:, None] > planned).sum(axis=0)
        starts = np.r_[np.full(len(buses), row["time_s"]), planned]
        ends = np.r_[active_ends, planned+max(duration, 1.)]
        peak = max(floor, occupancy_peak(starts, ends))
        features.append([*state, active[0]/12., float(active.max())/12., float(active.mean())/12.,
            (peak-floor)/12., c["prefix_fleet_cost"]-floor_cost, candidate["fleet_cost"]-floor_cost])
    return np.asarray(features)


def features(row, method):
    occupancy, _ = METHODS[method]
    extra = occupancy_features(row) if occupancy else np.zeros((len(row["candidates"]), len(OCCUPANCY_NAMES)))
    return np.c_[paired.action_features(row), extra]


def expected_regret(predicted, actual_cost):
    """Use native total-cost units for both cost heads and the decision loss."""
    probabilities = torch.softmax(-predicted.sum(dim=-1), dim=-1)
    regret = actual_cost-actual_cost.min(dim=-1, keepdim=True).values
    return (probabilities*regret).sum(dim=-1).mean()


def fit_value(rows, method, *, steps=STEPS):
    raw = np.stack([features(row, method) for row in rows])
    labels = [{l["candidate"]: l for l in row["labels"]} for row in rows]
    targets = np.asarray([[[l[i][h] for h in paired.HEADS] for i in range(6)] for l in labels])
    center, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.
    cost_scale = max(float(np.sqrt(np.mean(targets.sum(axis=-1)**2))), 1e-6)
    x = torch.as_tensor((raw-center)/scale, dtype=torch.float32)
    y = torch.as_tensor(targets/cost_scale, dtype=torch.float32)
    a, b = (torch.as_tensor(v, dtype=torch.long) for v in np.triu_indices(6, 1))
    ledger = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        network = nn.Sequential(nn.Linear(INPUT_DIM, WIDTH), nn.Tanh(), nn.Linear(WIDTH, 2))
        initial = [p.detach().clone() for p in network.parameters()]
        optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        for step in range(steps):
            predicted = network(x)
            mse = ((predicted[:, a]-predicted[:, b]-(y[:, a]-y[:, b]))**2).mean()
            regret = expected_regret(predicted, y.sum(dim=-1))
            loss = mse+regret if METHODS[method][1] else mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0 or (step+1) % 300 == 0 or step+1 == steps:
                ledger.append({"step": step+1, "mse": float(mse.detach()), "regret": float(regret.detach())})
        weights = {k: v.detach().numpy().tolist() for k, v in network.state_dict().items()}
        update_l2 = float(torch.sqrt(sum(((p.detach()-old)**2).sum() for p, old in zip(network.parameters(), initial))))
    return {"method": method, "heads": list(paired.HEADS), "center": center.tolist(), "scale": scale.tolist(),
        "cost_scale": cost_scale, "weights": weights, "steps": steps, "initialization_seed": SEED,
        "parameter_count": sum(p.numel() for p in network.parameters()), "loss_ledger": ledger,
        "parameter_update_l2": update_l2,
        "training_states": len(rows), "pairwise_examples": len(rows)*15,
        "training_groups": [list(g) for g in sorted({paired.scene_group(r) for r in rows})],
        "target": "single_action_terminal_fleet_nonfleet_cost_under_original_greedy_continuation"}


def predict_value(row, model):
    x = (features(row, model["method"])-np.asarray(model["center"]))/np.asarray(model["scale"])
    w = {k: np.asarray(v) for k, v in model["weights"].items()}
    hidden = np.tanh(x@w["0.weight"].T+w["0.bias"])
    heads = (hidden@w["2.weight"].T+w["2.bias"])*model["cost_scale"]
    return heads-heads[row["original_choice"]].copy()
