"""Prequential travel/holding innovations and descriptive dependence accounting."""

import numpy as np
import pandas as pd

from freqduet_holding_distribution import ACTIONS, probabilities_from_features

START, STOP = 3600, 49500
TIME_BINS = (0, 300, 900, 1800)


def issued_origin(clocks, time):
    index = np.searchsorted(clocks, time, side="left") - 1
    return None if index < 0 or time > STOP else int(clocks[index])


def identify_holding(teacher, holds, events, routes):
    keys = ["bus_id", "trip_id", "station_id", "time_s"]
    actions = events.loc[events.stage == "action"]
    joined = holds.merge(actions[keys + ["direction", "load", "action_s"]], on=keys,
                         validate="one_to_one", how="left", suffixes=("", "_event"))
    if joined.direction.isna().any() or not np.allclose(joined.action_s, joined.action_s_event):
        raise ValueError("holding ledger does not exactly match executed action identities/clocks")
    labels = []
    for row in teacher:
        x = row["features"]
        direction = str(bool(x[10]))
        ids = routes[direction]["station_ids"]
        index = int(round(x[8] * (len(ids) - 1)))
        if not np.isclose(index / (len(ids) - 1), x[8]):
            raise ValueError("teacher station phase is not a deployed stop")
        labels.append(dict(**row, direction=(direction == "True"), station_id=ids[index]))
    labels = pd.DataFrame(labels)
    match = ["time_s", "direction", "station_id"]
    result = labels.merge(joined, on=match, how="left", validate="one_to_one",
                          suffixes=("", "_ledger"))
    if result.bus_id.isna().any() or not np.allclose(result.action_s, result.action_s_ledger):
        raise ValueError("teacher holding label has no unique executed action")
    if not np.allclose(np.stack(result.features)[:, 5] * 50, result.load):
        raise ValueError("teacher and executed action passenger loads differ")
    return result


def identified_legs(events, routes):
    keys = ["bus_id", "trip_id", "station_id"]
    result, incomplete = [], 0
    for direction, route in routes.items():
        subset = events.loc[events.direction == (direction == "True")]
        dep = subset.loc[subset.stage == "departure", keys + ["time_s"]].rename(columns={"time_s": "start_s"})
        arr = subset.loc[subset.stage == "arrival", keys + ["time_s"]].rename(columns={"time_s": "end_s"})
        arr["station_id"] = arr.station_id.map(dict(zip(route["station_ids"][1:], route["station_ids"][:-1])))
        arr = arr.dropna(subset=["station_id"])
        pairs = dep.merge(arr, on=keys, how="left", validate="one_to_one")
        incomplete += int(pairs.end_s.isna().sum())
        pairs = pairs.dropna(subset=["end_s"]).copy()
        if (pairs.end_s <= pairs.start_s).any():
            raise ValueError("nonpositive identified travel leg")
        pairs["direction"] = (direction == "True")
        pairs["arrival_station_id"] = pairs.station_id.map(dict(zip(route["station_ids"][:-1], route["station_ids"][1:])))
        pairs["duration_s"] = pairs.end_s - pairs.start_s
        result.append(pairs)
    return pd.concat(result, ignore_index=True), incomplete


def finish_records(frame):
    frame = frame.sort_values(["time_s", "bus_id", "trip_id", "station_id"]).reset_index(drop=True)
    frame["hour"] = (frame.time_s // 3600).astype(int)
    frame["residual"] = frame.actual - frame.expected
    # This retrospective transform is diagnostic only, never a forecast input.
    frame["centered"] = frame.residual - frame.groupby(["direction", "station_id", "hour"]).residual.transform("mean")
    return frame


def holding_records(labels, captures):
    clocks = sorted(int(c["pool"]["time_s"]) for c in captures.values())
    rows = []
    for row in labels.itertuples():
        origin = issued_origin(clocks, row.time_s)
        if origin is None:
            continue
        model = captures[f"clock_{origin}"]["distribution_model"]
        if model["latest_time_s"] is not None and model["latest_time_s"] >= origin:
            raise ValueError("nonprefix holding fit")
        probs = probabilities_from_features(model, row.features)
        expected = float(probs @ ACTIONS)
        if np.min(abs(ACTIONS - row.action_s)) > 1e-5:
            raise ValueError("non-seven-bin observed holding")
        rows.append(dict(time_s=row.time_s, end_s=row.time_s, origin=origin,
            bus_id=int(row.bus_id), trip_id=int(row.trip_id), direction=bool(row.direction),
            station_id=int(row.station_id), actual=row.action_s, expected=expected,
            variance=float(probs @ (ACTIONS - expected) ** 2), support=model["samples"]))
    return finish_records(pd.DataFrame(rows))


def travel_records(legs, models):
    clocks, rows, profiles = sorted(models), [], {}
    for row in legs.itertuples():
        origin = issued_origin(clocks, row.start_s)
        if origin is None:
            continue
        model = models[origin]
        if model["now_s"] != origin or (model["latest_completion_s"] is not None and model["latest_completion_s"] >= origin):
            raise ValueError("nonprefix travel fit")
        key = f"{bool(row.direction)}:{int(row.station_id)}"
        profile = np.asarray(model["profiles"][key])
        lag = min(int((row.start_s - origin) // model["bin_s"]),
                  int(np.ceil(model["horizon_s"] / model["bin_s"])) - 1)
        deviation = np.roll(profile, -lag) - model["means"][key]
        identity = (int(row.trip_id), int(row.station_id))
        if identity in profiles:
            raise ValueError("duplicated travel label across forecast origins")
        profiles[identity] = deviation
        rows.append(dict(time_s=row.start_s, end_s=row.end_s, origin=origin,
            bus_id=int(row.bus_id), trip_id=int(row.trip_id), direction=bool(row.direction),
            station_id=int(row.station_id), arrival_station_id=int(row.arrival_station_id),
            actual=row.duration_s, expected=model["means"][key], variance=float(np.mean(deviation ** 2)),
            support=model["counts"][key]))
    frame = finish_records(pd.DataFrame(rows))
    width = max(map(len, profiles.values()))
    deviations = np.zeros((len(frame), width))
    lengths = np.zeros(len(frame))
    for i, row in enumerate(frame.itertuples()):
        deviation = profiles[(row.trip_id, row.station_id)]
        deviations[i, :len(deviation)], lengths[i] = deviation, len(deviation)
    return frame, deviations, lengths


def moments(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.array([len(x), x.sum(), y.sum(), x @ x, y @ y, x @ y], dtype=float)


def describe_moments(values):
    n, sx, sy, xx, yy, xy = values
    mx, my = sx / n, sy / n
    vx, vy, cov = max(0, xx / n - mx * mx), max(0, yy / n - my * my), xy / n - mx * my
    return dict(pairs=int(n), mean_left=mx, mean_right=my, product=xy / n, covariance=cov,
                correlation=cov / np.sqrt(vx * vy) if vx > 1e-12 and vy > 1e-12 else np.nan)


def pair_audit(frame, deviations=None, lengths=None):
    arrays = {k: frame[k].to_numpy() for k in ["time_s", "bus_id", "trip_id", "direction", "station_id", "origin", "residual", "centered"]}
    stats, model_stats = {}, {}

    def add(relation, lag, left, right):
        left, right = np.asarray(left, int), np.asarray(right, int)
        if not len(left):
            return
        # Different issued origins do not define a joint draw in the deployed model.
        valid = arrays["origin"][left] == arrays["origin"][right]
        a, b = left[valid], right[valid]
        covariance = (np.sum(deviations[a] * deviations[b], axis=1) / lengths[a]
                      if deviations is not None else np.zeros(len(a)))
        key = (relation, lag)
        model_stats[key] = model_stats.get(key, np.zeros(2)) + [len(a), covariance.sum()]
        for scope, x, y in (("all_pairs", left, right), ("same_origin", a, b)):
            if not len(x):
                continue
            for transform in ("residual", "centered"):
                key = (relation, lag, transform, scope)
                stats[key] = stats.get(key, np.zeros(7)) + np.r_[
                    moments(arrays[transform][x], arrays[transform][y]),
                    (arrays["time_s"][y] - arrays["time_s"][x]).sum()]

    for group_keys, base in ((["bus_id"], "same_bus"), (["direction", "station_id"], "same_directed_stop")):
        for _, group in frame.groupby(group_keys):
            indices = group.index.to_numpy()
            for lag in (1, 2, 3):
                a, b = indices[:-lag], indices[lag:]
                if base == "same_bus":
                    for same, suffix in ((True, "same_trip"), (False, "other_trip")):
                        valid = (arrays["trip_id"][a] == arrays["trip_id"][b]) == same
                        add(base + "_" + suffix, str(lag), a[valid], b[valid])
                else:
                    valid = arrays["bus_id"][a] != arrays["bus_id"][b]
                    add(base + "_different_bus", str(lag), a[valid], b[valid])
    times = arrays["time_s"]
    for i in range(len(frame)):
        # Every unordered pair within 30 minutes, counted once, without a large pair table.
        end = np.searchsorted(times, times[i] + TIME_BINS[-1], side="right")
        candidates = np.arange(i + 1, end)
        candidates = candidates[arrays["bus_id"][candidates] != arrays["bus_id"][i]]
        dt = times[candidates] - times[i]
        for lower, upper in zip(TIME_BINS[:-1], TIME_BINS[1:]):
            in_bin = (dt >= lower if lower == 0 else dt > lower) & (dt <= upper)
            for same, relation in ((True, "different_bus_same_direction"), (False, "different_bus_opposite_direction")):
                selected = candidates[in_bin & ((arrays["direction"][candidates] == arrays["direction"][i]) == same)]
                add(relation, f"{lower}-{upper}s", np.full(len(selected), i), selected)
    rows = []
    for (relation, lag, transform, scope), values in stats.items():
        n, covariance = model_stats[(relation, lag)]
        rows.append(dict(relation=relation, lag=lag, transform=transform, pair_scope=scope,
            mean_lag_s=values[6] / values[0], **describe_moments(values[:6]),
            same_origin_pairs=int(n),
            same_origin_model_covariance=covariance / n if n and transform == "residual" else np.nan))
    return pd.DataFrame(rows)


def grouped_square(frame, keys, value):
    return float(np.square(frame.groupby(keys)[value].sum()).sum())


def variance_parts(frame, value):
    r = frame[value].to_numpy()
    diagonal = float(r @ r)
    same_bus = grouped_square(frame, ["bus_id"], value) - diagonal
    same_stop = grouped_square(frame, ["direction", "station_id"], value) - grouped_square(frame, ["bus_id", "direction", "station_id"], value)
    bus_dir = grouped_square(frame, ["bus_id", "direction"], value) - diagonal
    dir_square = grouped_square(frame, ["direction"], value)
    same_dir = dir_square - diagonal - bus_dir - same_stop
    opposite = float(r.sum() ** 2) - dir_square - (same_bus - bus_dir)
    result = dict(sum_residual=float(r.sum()), squared_sum=float(r.sum() ** 2), diagonal=diagonal,
        same_bus_cross=same_bus, same_stop_other_bus_cross=same_stop,
        same_direction_other_cross=same_dir, opposite_direction_other_cross=opposite)
    if not np.isclose(sum(result[k] for k in ["diagonal", "same_bus_cross", "same_stop_other_bus_cross", "same_direction_other_cross", "opposite_direction_other_cross"]), result["squared_sum"], atol=1e-6):
        raise ValueError("residual-sum covariance partition does not close")
    return result


def window_audit(frame):
    rows, temporal = [], []
    for horizon in (900, 1800):
        for start in range(START, STOP - horizon + 1, horizon):
            group = frame.loc[(frame.time_s > start) & (frame.time_s <= start + horizon)]
            for transform in ("residual", "centered"):
                rows.append(dict(start_s=start, horizon_s=horizon, transform=transform, events=len(group),
                    independent_variance=group.variance.sum(), **variance_parts(group, transform)))
        for transform in ("residual", "centered"):
            subset = [r for r in rows if r["horizon_s"] == horizon and r["transform"] == transform]
            values = np.array([r["sum_residual"] for r in subset])
            for lag in (1, 2, 3):
                temporal.append(dict(horizon_s=horizon, transform=transform, lag=lag,
                    **describe_moments(moments(values[:-lag], values[lag:]))))
    return pd.DataFrame(rows), pd.DataFrame(temporal)


def incoming_travel_holding(travel, holding):
    keys = ["bus_id", "trip_id", "direction", "station_id"]
    incoming = travel.rename(columns={"station_id": "departure_station_id", "arrival_station_id": "station_id"})
    paired = holding.merge(incoming, on=keys, suffixes=("_holding", "_travel"), validate="one_to_one")
    if (paired.end_s_travel > paired.time_s_holding).any():
        raise ValueError("holding preceded the allegedly incoming completed leg")
    rows = []
    for transform in ("residual", "centered"):
        rows.append(dict(transform=transform, **describe_moments(moments(
            paired[transform + "_travel"], paired[transform + "_holding"])),
            mean_completion_to_action_s=float((paired.time_s_holding - paired.end_s_travel).mean())))
    return pd.DataFrame(rows)
