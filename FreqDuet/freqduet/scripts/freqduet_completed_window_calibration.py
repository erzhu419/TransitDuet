"""Prequential endpoint calibration from strictly completed, disjoint windows."""

from collections import defaultdict, deque
import math

import numpy as np


class CompletedWindows:
    def __init__(self, min_windows=8, history_size=16):
        self.minimum = min_windows
        self.history = defaultdict(lambda: deque(maxlen=history_size))
        self.pending = []
        self.last_end = defaultdict(lambda: -float("inf"))
        self.admitted, self.overlapping = 0, 0

    def issue(self, origin, horizon, draws):
        output = {}
        for metric, values in draws.items():
            history = self.history[(horizon, metric)]
            if any(row["end"] >= origin for row in history):
                raise ValueError("calibration label was not strictly available before issue")
            raw = float(np.mean(values))
            lo, hi = np.quantile(values, [.05, .95])
            mature = len(history) >= self.minimum
            bias = float(np.mean([row["raw_residual"] for row in history])) if mature else 0.0
            base_lo, base_hi = float(lo + bias), float(hi + bias)
            radius = None
            if mature:
                scores = sorted(row["score"] for row in history)
                rank = min(len(scores), math.ceil(.9 * (len(scores) + 1)))
                radius = float(scores[rank - 1])
            output[metric] = dict(raw=raw, point=max(0.0, raw + bias), bias=bias,
                base_low=base_lo, base_high=base_hi,
                low=max(0.0, base_lo - radius) if mature else None,
                high=max(0.0, base_hi + radius) if mature else None,
                radius=radius, mature=mature, samples=len(history),
                latest_completion=max((row["end"] for row in history), default=None))
        record = dict(origin=float(origin), horizon=int(horizon), end=float(origin + horizon),
                      metrics=output)
        self.pending.append(record)
        return output

    def release(self, now, truth_for_window):
        remaining = []
        for record in self.pending:
            if record["end"] >= now:
                remaining.append(record)
                continue
            horizon = record["horizon"]
            if record["origin"] < self.last_end[horizon]:
                self.overlapping += 1
                continue
            truth = truth_for_window(record["origin"], horizon)
            for metric, issued in record["metrics"].items():
                value = float(truth[metric])
                score = max(issued["base_low"] - value, value - issued["base_high"], 0.0)
                self.history[(horizon, metric)].append(dict(end=record["end"],
                    raw_residual=value - issued["raw"], score=score))
            self.last_end[horizon] = record["end"]
            self.admitted += 1
        self.pending = remaining
