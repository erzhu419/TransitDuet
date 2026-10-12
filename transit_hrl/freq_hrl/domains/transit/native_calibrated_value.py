"""Scene-held-out, action-simultaneous empirical cost/wait margins."""

from collections import defaultdict
import math

import numpy as np

from freq_hrl.domains.transit import native_action_value as paired
from freq_hrl.domains.transit import native_floor_value as value

MEMBERS = 4
PARTITION_SEED = 190
QUANTILE = .9
POLICIES = ("ensemble_mean", "calibrated")


def partition_groups(groups):
    strata = defaultdict(list)
    for g in groups:
        strata[g["root"], g["scenario"]].append((g["root"], g["scene_seed"]))
    rng = np.random.default_rng(PARTITION_SEED)
    calibration, excluded = [], [[] for _ in range(MEMBERS)]
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        keys = [keys[i] for i in rng.permutation(len(keys))]
        n = math.ceil(len(keys)/5)
        calibration.extend(keys[:n])
        for i, key in enumerate(keys[n:]):
            excluded[i % MEMBERS].append(key)
    proper = {tuple(g) for values in excluded for g in values}
    return {"calibration_groups": sorted(map(list, calibration)),
        "member_groups": [sorted(map(list, proper-set(ex))) for ex in excluded]}


def training_scales(rows):
    targets = np.asarray([[[l["actual_cost_delta"], l["wait_delta_min"]]
        for l in r["labels"] if l["candidate"] != r["original_choice"]] for r in rows])
    return np.maximum(.1*np.sqrt(np.mean(targets**2, axis=(0, 1))), 1e-6)


def ensemble_prediction(row, models, floors):
    estimates = [value.predict(row, model) for model in models]
    predictions = np.asarray([np.c_[np.asarray(p["predicted_heads"]).sum(axis=1),
        p["predicted_wait_delta_min"]] for p in estimates])
    mean = predictions.mean(axis=0)
    uncertainty = np.maximum(predictions.std(axis=0, ddof=1), floors)
    mean[row["original_choice"]] = 0.
    uncertainty[row["original_choice"]] = 0.
    return {"mean": mean.tolist(), "uncertainty": uncertainty.tolist()}


def calibrate(rows, predictions):
    # Maxima cover adaptive selection among all actions and all six scene callbacks.
    scores = {}
    for row, prediction in zip(rows, predictions, strict=True):
        actual = np.asarray([[l["actual_cost_delta"], l["wait_delta_min"]] for l in row["labels"]])
        other = [i for i in range(6) if i != row["original_choice"]]
        residual = (actual[other]-np.asarray(prediction["mean"])[other])/np.asarray(prediction["uncertainty"])[other]
        key = paired.scene_group(row)
        scores[key] = np.maximum(scores.get(key, np.zeros(2)), residual.max(axis=0))
    ordered = np.sort(np.asarray(list(scores.values())), axis=0)
    rank = min(len(ordered), math.ceil((len(ordered)+1)*QUANTILE))
    return {"multipliers": ordered[rank-1].tolist(), "quantile": QUANTILE, "rank": rank,
        "groups": [list(g) for g in sorted(scores)], "states": len(rows),
        "group_scores": [{"group": list(g), "score": scores[g].tolist()} for g in sorted(scores)]}


def choose(prediction, original, multipliers):
    upper = np.asarray(prediction["mean"])+np.asarray(prediction["uncertainty"])*multipliers
    upper[original] = 0.
    return paired.choice_from_cost(np.where(upper[:, 1] <= 0, upper[:, 0], np.inf), original)


def evaluate(row, prediction, multipliers):
    original = row["original_choice"]
    selected = choose(prediction, original, multipliers)
    label = row["labels"][selected]
    upper = np.asarray(prediction["mean"])+np.asarray(prediction["uncertainty"])*multipliers
    mean = np.asarray(prediction["mean"])
    return {**{k: row[k] for k in ("root", "scene_seed", "scenario", "probe_index", "original_choice")},
        **prediction, "selected": selected, "upper": upper.tolist(),
        **{k: label[k] for k in ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min", "episode_reward_delta")},
        "selected_cost_bias": float(mean[selected, 0]-label["actual_cost_delta"]),
        "selected_wait_bias_min": float(mean[selected, 1]-label["wait_delta_min"]),
        "cost_bound_violation": bool(label["actual_cost_delta"] > upper[selected, 0]),
        "wait_bound_violation": bool(label["wait_delta_min"] > upper[selected, 1])}


def summary(rows):
    return {"states": len(rows), "changed_choices": sum(r["selected"] != r["original_choice"] for r in rows),
        "fallback_choices": sum(r["selected"] == r["original_choice"] for r in rows),
        "positive_cost_choices": sum(r["actual_cost_delta"] > 0 for r in rows),
        "positive_wait_choices": sum(r["wait_delta_min"] > 0 for r in rows),
        **{"mean_"+k: float(np.mean([r[k] for r in rows])) for k in
            ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min", "episode_reward_delta",
                "selected_cost_bias", "selected_wait_bias_min")},
        "selected_cost_bound_violations": sum(r["cost_bound_violation"] for r in rows),
        "selected_wait_bound_violations": sum(r["wait_bound_violation"] for r in rows)}
