"""Paired wait credit with causal occupancy-by-timetable interactions."""

import numpy as np

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_floor_value as value

REPRESENTATIONS = ("geometry_wait", "conditional_wait")
STATE_NAMES = ("same_queue", "same_queue_squared", "other_queue", "same_onboard",
    "same_mean_load", "same_max_load", "same_queue_concentration", "same_active",
    "other_active", "same_remaining_mean", "queue_load_pressure", "lf_forecast")
GEOMETRY_NAMES = ("mean_gap", "first_gap", "last_gap", "gap_std", "mean_nominal_shift",
    "mean_squared_nominal_shift", "early_nominal_shift", "mean_squared_gap")
FEATURE_COUNTS = {"geometry_wait": 102, "conditional_wait": 198}


def geometry_features(row):
    result = []
    for candidate in row["candidates"]:
        planned = np.asarray(candidate["commitment"]["prefix_planned_s"])
        gaps = np.diff(planned)/360.
        shifts = (planned-np.asarray(row["nominal_s"][:len(planned)]))/360.
        result.append([gaps.mean(), gaps[0], gaps[-1], gaps.std(), shifts.mean(),
            np.mean(shifts**2), np.mean((1-np.linspace(0, 1, len(planned)))*shifts), np.mean(gaps**2)])
    return np.asarray(result)


def condition_features(row):
    direction, context = row["direction"], row["context"]
    buses = row["occupancy_context"]["active_buses"]
    same = [b for b in buses if b["direction"] == direction]
    loads = [b["load_fraction"] for b in same]
    queues = [q[2] for q in row["occupancy_context"]["queue_by_station"] if q[1] == direction]
    q = context["queues"][str(direction)]/50.
    load = float(np.mean(loads)) if loads else 0.
    return np.asarray([q, q*q, context["queues"][str(not direction)]/50.,
        context["onboard"][str(direction)]/100., load, max(loads, default=0.),
        max(queues, default=0)/max(sum(queues), 1), len(same)/12., (len(buses)-len(same))/12.,
        float(np.mean([b["predicted_remaining_s"] for b in same]))/1800. if same else 0.,
        q*load, row["native_state_v2"][12]])


def features(row, representation):
    base, geometry = value.features(row, "tail_summary"), geometry_features(row)
    result = np.c_[base, geometry[:, 4:]]
    if representation == "conditional_wait":
        result = np.c_[result, (condition_features(row)[None, :, None]*geometry[:, None, :]).reshape(len(base), -1)]
    elif representation != "geometry_wait":
        raise ValueError("Unknown conditional wait representation")
    return result


def fit_wait(rows, representation, *, sample_weights):
    raw = np.stack([features(r, representation) for r in rows])
    labels = [{l["candidate"]: l["wait_delta_min"] for l in r["labels"]} for r in rows]
    targets = np.asarray([[label[i] for i in range(6)] for label in labels])
    center, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.
    a, b = np.triu_indices(6, 1)
    gaps = targets[:, a]-targets[:, b]
    target_scale = max(float(np.sqrt(np.mean(gaps**2))), 1e-6)
    x = (raw-center)/scale
    z, y = (x[:, a]-x[:, b]).reshape(-1, raw.shape[-1]), gaps.reshape(-1)/target_scale
    weights = np.repeat(sample_weights, len(a))
    coefficients = np.linalg.solve(z.T@(weights[:, None]*z)/weights.sum()+paired.RIDGE*np.eye(z.shape[1]),
        z.T@(weights*y)/weights.sum())
    return {"method": "paired_wait_ridge", "representation": representation, "coefficients": coefficients.tolist(),
        "center": center.tolist(), "scale": scale.tolist(), "target_scale_min": target_scale,
        "ridge": paired.RIDGE, "parameter_count": coefficients.size, "steps": 0,
        "final_pair_loss": float(np.average((z@coefficients-y)**2, weights=weights)),
        "training_states": len(rows), "training_groups": sorted(map(list, {paired.scene_group(r) for r in rows})),
        "sample_weight_sum": float(sum(sample_weights)), "target": "all_candidate_paired_terminal_wait_delta_min",
        "condition_names": list(STATE_NAMES), "geometry_names": list(GEOMETRY_NAMES)}


def predict_wait(row, model):
    x = (features(row, model["representation"])-model["center"])/model["scale"]
    x -= x[row["original_choice"]].copy()
    return (x@np.asarray(model["coefficients"]))*model["target_scale_min"]
