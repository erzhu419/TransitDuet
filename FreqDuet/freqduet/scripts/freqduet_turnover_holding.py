"""Seven-bin holding response conditioned on observable completed vehicle turnover."""

import copy
import math

import numpy as np

from freqduet_closed_loop_forecast import holding_features
from freqduet_holding_distribution import ACTIONS, probabilities_from_features, sample_action


def fit_turnover_tilt(rows, now):
    import torch

    past = rows.loc[rows.time_s < now].tail(4096)
    past = past.loc[past.context.map(lambda values: bool(np.any(values)))]
    result = dict(coef=[0.] * 7, scale=[1.] * 7, samples=len(past),
        latest_time_s=float(past.time_s.max()) if len(past) else None)
    if not len(past):
        return result
    features = np.stack(past.context)
    scale = np.maximum(np.sqrt(np.mean(features ** 2, axis=0)), 1e-8)
    x = torch.as_tensor(features / scale, dtype=torch.float64)
    base = torch.as_tensor(np.log(np.stack(past.base_probabilities)), dtype=torch.float64)
    actions = torch.as_tensor(ACTIONS / 45, dtype=torch.float64)
    labels = torch.as_tensor([int(np.argmin(abs(ACTIONS - y))) for y in past.action_s], dtype=torch.long)
    weights = torch.zeros(7, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=100, line_search_fn="strong_wolfe",
        tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        logits = base + (x @ weights)[:, None] * actions
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum") + 5 * weights.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    result.update(coef=weights.detach().numpy().tolist(), scale=scale.tolist())
    return result


def tilted_probabilities(base_model, features, context, correction):
    probabilities = probabilities_from_features(base_model, features)
    tilt = np.asarray(context) / np.asarray(correction["scale"]) @ np.asarray(correction["coef"])
    logits = tilt * ACTIONS / 45
    mass = probabilities * np.exp(logits - logits.max())
    return mass / mass.sum()


def turnover_features(previous, current, time):
    elapsed = max(0.0, time - current["launch_s"])
    memory = math.exp(-elapsed / 900)
    known = previous is not None
    scored = known and previous["scored_count"] > 0
    if not scored or current["decisions"] != 0:
        return np.zeros(7)
    return np.array([float(scored),
        memory * previous["mean_residual_s"] / 45 if scored else 0,
        memory * previous["last_residual_s"] / 45 if scored else 0,
        memory * previous["duration_s"] / 3600 if known else 0,
        max(0, current["launch_s"] - previous["end_s"]) / 3600 if known else 0,
        max(0, current["launch_s"] - current["scheduled_s"]) / 3600, elapsed / 3600])


def completed_profiles(labels, launches, terminals):
    launch = launches.set_index("trip_id").to_dict("index")
    profiles = {}
    for row in terminals.sort_values("time_s").itertuples():
        past = labels.loc[(labels.trip_id == row.trip_id) & (labels.time_s < row.time_s)].dropna(subset=["residual_s"])
        entry = dict(trip_id=int(row.trip_id), end_s=float(row.time_s),
            duration_s=float(row.time_s - launch[row.trip_id]["time_s"]),
            scored_count=len(past), mean_residual_s=float(past.residual_s.mean()) if len(past) else 0.,
            last_residual_s=float(past.residual_s.iloc[-1]) if len(past) else 0.)
        profiles.setdefault(int(row.bus_id), []).append(entry)
    return launch, profiles


def augment_labels(labels, launches, terminals):
    launch, profiles = completed_profiles(labels, launches, terminals)
    features, has_previous = [], []
    ordinals = labels.groupby("trip_id").cumcount()
    for row, ordinal in zip(labels.itertuples(), ordinals):
        current = launch[row.trip_id]
        if current["time_s"] >= row.time_s:
            raise ValueError("holding context precedes its observed launch")
        past = [entry for entry in profiles.get(row.bus_id, []) if entry["end_s"] < row.time_s]
        previous = past[-1] if past else None
        has_previous.append(previous is not None and previous["scored_count"] > 0)
        context = turnover_features(previous, dict(launch_s=current["time_s"], scheduled_s=current["scheduled_s"], decisions=int(ordinal)), row.time_s)
        features.append(np.r_[row.features, context].tolist())
    result = labels.copy()
    result["augmented_features"] = features
    result["has_scored_previous"] = has_previous
    return result


def freeze_turnover(labels, launches, terminals, observed):
    now = observed["time_s"]
    labels = labels.loc[labels.time_s < now]
    launches = launches.loc[launches.time_s < now]
    terminals = terminals.loc[terminals.time_s < now]
    launch, profiles = completed_profiles(labels, launches, terminals)
    result = {}
    for vehicle in observed["vehicles"]:
        bus, trip = vehicle["bus_id"], vehicle["trip_id"]
        past = profiles.get(bus, [])
        current = None
        if vehicle["on_route"]:
            entry = launch[trip]
            all_holds = labels.loc[labels.trip_id == trip]
            holds = all_holds.dropna(subset=["residual_s"])
            current = dict(trip_id=trip, launch_s=entry["time_s"], scheduled_s=entry["scheduled_s"],
                decisions=len(all_holds), count=len(holds), total=float(holds.residual_s.sum()), last=float(holds.residual_s.iloc[-1]) if len(holds) else 0.)
        result[str(bus)] = dict(previous=past[-1] if past else None, current=current)
    return result


class TurnoverResponse:
    def __init__(self, model, history, routes, draw):
        self.model, self.history, self.routes = model, copy.deepcopy(history), routes
        self.rng = np.random.default_rng(20261015 + draw)

    def current(self, row):
        state = self.history[str(row["bus_id"])]
        if state["current"] is None or state["current"]["trip_id"] != row["trip_id"]:
            state["current"] = dict(trip_id=row["trip_id"], launch_s=row["launch_time_s"],
                scheduled_s=row["scheduled_launch_s"], decisions=0, count=0, total=0., last=0.)
        return state

    def __call__(self, values):
        state = self.current(values)
        context = turnover_features(state["previous"], state["current"], values["time_s"])
        features = holding_features(values)
        base = probabilities_from_features(self.model["base"], features)
        state["current"]["base_mean_s"] = float(base @ ACTIONS)
        probabilities = tilted_probabilities(self.model["base"], features, context, self.model["correction"])
        return sample_action(probabilities, self.rng.random())

    def observe(self, row):
        state = self.current(row)
        current = state["current"]
        if row["stage"] == "action" and row["evidence_valid"]:
            current["decisions"] += 1
            current["count"] += 1
            residual = row["action_s"] - current["base_mean_s"]
            current["total"] += residual
            current["last"] = residual
        terminal = self.routes[str(row["direction"])]["station_ids"][-1]
        if row["stage"] == "arrival" and row["station_id"] == terminal:
            state["previous"] = dict(trip_id=current["trip_id"], end_s=row["time_s"],
                duration_s=row["time_s"] - current["launch_s"],
                scored_count=current["count"],
                mean_residual_s=current["total"] / current["count"] if current["count"] else 0.,
                last_residual_s=current["last"])
            state["current"] = None
