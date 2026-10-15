"""Prequential return-time intervals using only previously completed trips."""

from collections import defaultdict, deque
import math

import numpy as np


def lead_band(remaining_s):
    return "short" if remaining_s <= 900 else "medium" if remaining_s <= 1800 else "long"


class ReturnIntervals:
    def __init__(self):
        self.completed = defaultdict(lambda: deque(maxlen=128))
        self.pending, self.seen, self.rows = {}, set(), []

    def interval(self, now, predicted):
        band = lead_band(predicted - now)
        past = [row for row in self.completed[band] if row[0] < now]
        result = dict(band=band, calibration_n=len(past), available=len(past) >= 20,
                      latest_completion_s=max((row[0] for row in past), default=None),
                      lower_s=None, upper_s=None, radius_s=None)
        if result["available"]:
            residuals = sorted(abs(row[1]) for row in past)
            rank = min(math.ceil(.9 * (len(past) + 1)), len(past))
            radius = residuals[rank - 1]
            result.update(lower_s=max(now, predicted - radius), upper_s=predicted + radius, radius_s=radius)
        return result

    def issue(self, now, trip_id, predicted):
        interval = self.interval(now, predicted)
        key = (int(trip_id), interval["band"])
        if key in self.seen:
            return
        self.seen.add(key)
        row = dict(origin_s=now, trip_id=int(trip_id), predicted_return_s=predicted, **interval)
        self.rows.append(row)
        self.pending[key] = row

    def resolve(self, trip_id, actual, observed_time):
        if actual >= observed_time:
            raise ValueError("return label is not strictly earlier than the current observation")
        for key in [key for key in self.pending if key[0] == trip_id]:
            row = self.pending.pop(key)
            residual = actual - row["predicted_return_s"]
            row.update(actual_return_s=actual, residual_s=residual)
            if row["available"]:
                lower, upper = row["lower_s"], row["upper_s"]
                row.update(covered=lower <= actual <= upper, width_s=upper - lower,
                           interval_score_s=upper - lower + 20 * (max(lower - actual, 0) + max(actual - upper, 0)))
            self.completed[key[1]].append((actual, residual))


def interval_summary(rows):
    results = []
    for band in ("short", "medium", "long"):
        resolved = [row for row in rows if row["band"] == band and "actual_return_s" in row]
        available = [row for row in resolved if row["available"]]
        results.append(dict(band=band, forecasts=len(resolved), calibrated=len(available),
                            availability=len(available) / max(len(resolved), 1),
                            coverage=float(np.mean([row["covered"] for row in available])) if available else None,
                            mean_width_s=float(np.mean([row["width_s"] for row in available])) if available else None,
                            interval_score_s=float(np.mean([row["interval_score_s"] for row in available])) if available else None,
                            point_mae_s=float(np.mean([abs(row["residual_s"]) for row in resolved])) if resolved else None))
    return results
