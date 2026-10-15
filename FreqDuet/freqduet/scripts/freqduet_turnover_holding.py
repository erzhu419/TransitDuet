"""Seven-bin holding response conditioned on observable completed vehicle turnover."""

import copy
import math

import numpy as np

from freqduet_closed_loop_forecast import holding_features
from freqduet_holding_distribution import probabilities_from_features, sample_action


def turnover_features(previous, current, time):
    elapsed = max(0.0, time - current["launch_s"])
    memory = math.exp(-elapsed / 900)
    known = previous is not None
    return np.array([float(known),
        memory * previous["mean_hold_s"] / 45 if known else 0,
        memory * previous["last_hold_s"] / 45 if known else 0,
        memory * previous["duration_s"] / 3600 if known else 0,
        max(0, current["launch_s"] - previous["end_s"]) / 3600 if known else 0,
        max(0, current["launch_s"] - current["scheduled_s"]) / 3600, elapsed / 3600])


def completed_profiles(labels, launches, terminals):
    launch = launches.set_index("trip_id").to_dict("index")
    profiles = {}
    for row in terminals.sort_values("time_s").itertuples():
        past = labels.loc[(labels.trip_id == row.trip_id) & (labels.time_s < row.time_s)]
        entry = dict(trip_id=int(row.trip_id), end_s=float(row.time_s),
            duration_s=float(row.time_s - launch[row.trip_id]["time_s"]),
            mean_hold_s=float(past.action_s.mean()) if len(past) else 0.,
            last_hold_s=float(past.action_s.iloc[-1]) if len(past) else 0.)
        profiles.setdefault(int(row.bus_id), []).append(entry)
    return launch, profiles


def augment_labels(labels, launches, terminals):
    launch, profiles = completed_profiles(labels, launches, terminals)
    features = []
    for row in labels.itertuples():
        current = launch[row.trip_id]
        if current["time_s"] >= row.time_s:
            raise ValueError("holding context precedes its observed launch")
        past = [entry for entry in profiles.get(row.bus_id, []) if entry["end_s"] < row.time_s]
        previous = past[-1] if past else None
        context = turnover_features(previous, dict(launch_s=current["time_s"], scheduled_s=current["scheduled_s"]), row.time_s)
        features.append(np.r_[row.features, context].tolist())
    result = labels.copy()
    result["augmented_features"] = features
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
            holds = labels.loc[labels.trip_id == trip]
            current = dict(trip_id=trip, launch_s=entry["time_s"], scheduled_s=entry["scheduled_s"],
                count=len(holds), total=float(holds.action_s.sum()), last=float(holds.action_s.iloc[-1]) if len(holds) else 0.)
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
                scheduled_s=row["scheduled_launch_s"], count=0, total=0., last=0.)
        return state

    def __call__(self, values):
        state = self.current(values)
        features = np.r_[holding_features(values), turnover_features(state["previous"], state["current"], values["time_s"])]
        return sample_action(probabilities_from_features(self.model, features), self.rng.random())

    def observe(self, row):
        state = self.current(row)
        current = state["current"]
        if row["stage"] == "action" and row["evidence_valid"]:
            current["count"] += 1
            current["total"] += row["action_s"]
            current["last"] = row["action_s"]
        terminal = self.routes[str(row["direction"])]["station_ids"][-1]
        if row["stage"] == "arrival" and row["station_id"] == terminal:
            state["previous"] = dict(trip_id=current["trip_id"], end_s=row["time_s"],
                duration_s=row["time_s"] - current["launch_s"],
                mean_hold_s=current["total"] / current["count"] if current["count"] else 0.,
                last_hold_s=current["last"])
            state["current"] = None
