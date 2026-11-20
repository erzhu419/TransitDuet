"""Causal non-harmonic features, matched rule, and common schedule readout."""

import math

import numpy as np


class MinuteRates:
    def __init__(self, prior):
        self.current = dict(prior)
        self.previous = dict(prior)
        self.pending = dict.fromkeys(prior, 0.0)

    def add(self, counts, bin_closed):
        for key, value in counts.items():
            self.pending[key] += value
        if bin_closed:
            self.previous, self.current = self.current, self.pending
            self.pending = dict.fromkeys(self.current, 0.0)

    def station(self, sid, direction):
        key = (sid, direction)
        now = self.current.get(key, 0.0)
        return now, now - self.previous.get(key, 0.0)

    def directional(self, direction):
        return [self.station(sid, d) for sid, d in self.current if d == direction]


def reference_rate(daily_pax):
    return max(0.5 * daily_pax / 783.0, 0.2)


def raw_upper(rates, daily_pax):
    scale = reference_rate(daily_pax)
    out = []
    for direction in (True, False):
        rate, change = np.asarray(rates.directional(direction), dtype=float).T
        out.extend(np.clip([rate.sum() / scale,
                            np.sqrt(np.mean(rate**2)) * len(rate) / scale,
                            rate.std() * len(rate) / scale,
                            change.sum() * 5 / scale], -3.5, 3.5))
    return out


def raw_lower(rates, here, downstream, direction):
    rate, change = rates.station(here, direction)
    nearby = [rates.station(sid, direction) for sid in downstream]
    near, near_change = np.mean(nearby, axis=0) if nearby else (0.0, 0.0)
    return np.clip(np.asarray([rate, near, 5 * change, 5 * near_change]) / 3.0,
                   -3.5, 3.5).tolist()


def desired_headways(action, available, ready, floor):
    factor, tilt = map(float, action[:2])
    return {
        d: float(np.clip(available[d] * factor + (1.0 if d else -1.0) * tilt
                         * float(np.clip(0.55 + 0.35 * (ready[d] - ready[not d]), 0.15, 1.0)),
                         floor, 455.0))
        for d in available
    }


def rule_plan(actions, available, ready, floor, rates, daily_pax):
    ideal = {}
    for d, gap in available.items():
        observed = sum(value for value, _ in rates.directional(d))
        ratio = math.sqrt(reference_rate(daily_pax) / max(observed, 0.1))
        ideal[d] = float(np.clip(gap * ratio, floor, 455.0))

    def score(a):
        gaps = desired_headways(a, available, ready, floor)
        error = sum(((gaps[d] - ideal[d]) / ideal[d])**2 for d in gaps)
        return error, abs(float(a[1])), abs(float(a[0]) - 1.0)

    return min(actions, key=score)


def rule_hold(bus, ctx, rates, bins, remaining, reward):
    if ctx["fwd"] is None:
        return 0.0
    stops = bus.effective_station
    i = stops.index(bus.last_station)
    downstream = sum(rates.station(int(st.station_id), bool(bus.direction))[0]
                     for st in stops[i + 1:-1])
    legal = [u for u in bins if u <= remaining + 1e-6]
    return float(max(legal, key=lambda u: reward(ctx, u, downstream / 60.0)[0]))


def dispatch_metrics(timetables):
    offsets = np.asarray([
        tt._actual_launch_time - tt._freqduet_base_launch
        for tt in timetables if tt.launched
    ], dtype=float)
    return {
        "public_abs_deviation_s_per_trip": float(np.abs(offsets).mean()),
        "public_late_s_per_trip": float(np.maximum(offsets, 0).mean()),
        "public_early_s_per_trip": float(np.maximum(-offsets, 0).mean()),
    }
