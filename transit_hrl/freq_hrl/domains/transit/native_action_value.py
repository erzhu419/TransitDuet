"""Paired physical-cost learning over the existing native timetable actions."""

from itertools import combinations

import numpy as np

RIDGE = .1
HEADS = ("fleet_cost_delta", "nonfleet_cost_delta")
STATE_FEATURES = ("bias", "clock_fraction", "direction", "lf_forecast", "lf_slope",
    "same_queue", "same_onboard", "active_fleet", "holding_ratio")
ACTION_FEATURES = ("mean_gap", "first_gap", "last_gap", "gap_std", "anchor_shift",
    "prefix_wait", "prefix_fleet", "full_fleet")


def scene_group(row):
    return row["root"], row["scene_seed"]


def action_features(row):
    """Only preaction LF/physical context and proposed committed geometry."""
    state, context = row["native_state_v2"], row["context"]
    same = str(row["direction"])
    s = np.asarray([1., row["time_s"]/61380., float(row["direction"]), state[12], state[11],
        context["queues"][same]/50., context["onboard"][same]/100.,
        sum(context["active"].values())/12., state[4]])
    actions = []
    for candidate in row["candidates"]:
        c = candidate["commitment"]
        planned = np.asarray(c["prefix_planned_s"])
        gaps = np.diff(planned)/360.
        actions.append([gaps.mean(), gaps[0], gaps[-1], gaps.std(),
            (planned[-1]-row["nominal_s"][len(planned)-1])/360.,
            c["prefix_wait_cost"], c["prefix_fleet_cost"], candidate["fleet_cost"]])
    x = np.asarray(actions)[:, None, :] * s[None, :, None]
    return x.reshape(len(actions), -1)


def fit_action_value(rows):
    x, y = [], []
    for row in rows:
        features = action_features(row)
        labels = {l["candidate"]: l for l in row["labels"]}
        for a, b in combinations(range(len(features)), 2):
            x.append(features[a]-features[b])
            y.append([labels[a][h]-labels[b][h] for h in HEADS])
    x, y = np.asarray(x), np.asarray(y)
    # RMS and the ridge system are fitted anew using training scenes only.
    scale = np.sqrt(np.mean(x*x, axis=0))
    scale[scale == 0] = 1.
    z = x/scale
    weights = np.linalg.solve(z.T@z/len(z)+RIDGE*np.eye(z.shape[1]), z.T@y/len(z))
    return {"coefficients": (weights/scale[:, None]).tolist(), "heads": list(HEADS),
        "feature_names": [f"{s}:{a}" for s in STATE_FEATURES for a in ACTION_FEATURES],
        "ridge": RIDGE, "training_states": len(rows), "pairwise_examples": len(x),
        "training_groups": [list(g) for g in sorted({scene_group(r) for r in rows})],
        "target": "single_action_terminal_cost_difference_under_original_greedy_continuation"}


def predict_action_value(row, model):
    x = action_features(row)
    x -= x[row["original_choice"]].copy()
    return x @ np.asarray(model["coefficients"])


def choose_action(row, model):
    cost = predict_action_value(row, model).sum(axis=1)
    return choice_from_cost(cost, row["original_choice"])


def choice_from_cost(cost, original):
    # An exactly tied value retains the source action (including a zero model).
    order = [original, *[i for i in range(len(cost)) if i != original]]
    return min(order, key=lambda i: cost[i])


def evaluate_action_value(rows, model, *, predictor=predict_action_value):
    evaluated = []
    for row in rows:
        predicted = predictor(row, model)
        selected = choice_from_cost(predicted.sum(axis=1), row["original_choice"])
        labels = {l["candidate"]: l for l in row["labels"]}
        observed = np.asarray([labels[i]["actual_cost_delta"] for i in range(len(predicted))])
        costs = predicted.sum(axis=1)
        pairs = [(a, b) for a, b in combinations(range(len(costs)), 2)
            if abs(observed[a]-observed[b]) > 1e-12]
        evaluated.append({**{k: row[k] for k in ("root", "scenario", "scene_seed", "probe_index", "direction")},
            "original_choice": row["original_choice"], "selected": selected,
            "predicted_heads": predicted.tolist(),
            **{k: labels[selected][k] for k in ("actual_cost_delta", *HEADS, "wait_delta_min", "episode_reward_delta")},
            "best_realized_cost_delta": float(observed.min()),
            "regret": float(observed[selected]-observed.min()),
            "strict_pairs": len(pairs),
            "concordant_pairs": int(sum((costs[a]-costs[b])*(observed[a]-observed[b]) > 0 for a, b in pairs)),
            "tied_predicted_pairs": int(sum(costs[a] == costs[b] for a, b in pairs))})
    return evaluated


def summarize_choices(rows):
    return {"states": len(rows), "changed_choices": sum(r["selected"] != r["original_choice"] for r in rows),
        "improved_states": sum(r["actual_cost_delta"] < 0 for r in rows),
        "worsened_states": sum(r["actual_cost_delta"] > 0 for r in rows),
        **{f"mean_{k}": float(np.mean([r[k] for r in rows])) for k in
            ("actual_cost_delta", *HEADS, "wait_delta_min", "episode_reward_delta", "regret", "best_realized_cost_delta")},
        "strict_pairs": sum(r["strict_pairs"] for r in rows),
        "concordant_pairs": sum(r["concordant_pairs"] for r in rows),
        "tied_predicted_pairs": sum(r["tied_predicted_pairs"] for r in rows)}


def grouped_validation(rows, split, *, fit_model=fit_action_value, predictor=predict_action_value):
    key = {"scene": scene_group, "regime": lambda r: r["scenario"], "lower_root": lambda r: r["root"]}[split]
    folds, predictions = [], []
    for group in sorted({key(r) for r in rows}):
        train = [r for r in rows if key(r) != group]
        test = [r for r in rows if key(r) == group]
        model = fit_model(train)
        evaluated = evaluate_action_value(test, model, predictor=predictor)
        folds.append({"held_out": group, "training_groups": model["training_groups"],
            "training_states": len(train), **summarize_choices(evaluated)})
        predictions.extend(evaluated)
    return {"split": split, "folds": folds, "predictions": predictions,
        "overall": summarize_choices(predictions),
        "by_root": {str(r): summarize_choices([p for p in predictions if p["root"] == r])
            for r in sorted({row["root"] for row in rows})}}
