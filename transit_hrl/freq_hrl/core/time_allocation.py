"""Allocate service intervals without changing a window's time or event budget."""

import numpy as np


def budgeted_time_points(nominal, preferred_intervals, *, minimum, maximum):
    """Center interval adjustments and shrink uniformly into physical bounds.

    Endpoints and event count stay fixed; internal points can move cumulatively.
    The returned integer-clock intervals can differ from continuous ones by 1 s.
    """
    nominal = np.asarray(nominal, dtype=np.float64)
    base = np.diff(nominal)
    preferred = np.asarray(preferred_intervals, dtype=np.float64)
    if (nominal.ndim != 1 or len(nominal) < 2 or preferred.shape != base.shape
            or not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(preferred))
            or not 0 < minimum <= maximum or np.any(base < minimum) or np.any(base > maximum)):
        raise ValueError("Expected an ordered feasible event window and one preference per interval")
    delta = preferred - base
    delta -= delta.mean()
    scale = 1.0
    positive, negative = delta > 0, delta < 0
    if positive.any():
        scale = min(scale, float(np.min((maximum - base[positive]) / delta[positive])))
    if negative.any():
        scale = min(scale, float(np.min((minimum - base[negative]) / delta[negative])))
    points = np.r_[nominal[0], nominal[0] + np.cumsum(base + scale * delta)]
    points[-1] = nominal[-1]
    return np.rint(points).astype(np.int64)
