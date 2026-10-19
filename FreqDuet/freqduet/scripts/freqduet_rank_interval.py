"""Fixed 10%-per-tail rank bounds for nonnegative remaining durations."""

from bisect import bisect_left

import numpy as np

import audit_freqduet_v120_component_eta as model


def completed_values(history, direction, station, now, elapsed):
    k = direction, station
    if k not in history.groups:
        return np.asarray([], dtype=float), None
    times = history.times[k]
    end = bisect_left(times, now)
    start = bisect_left(times, now-model.WINDOW_S)
    if start == end:
        start = 0
    survivors = [(t, d-elapsed) for t, d in history.groups[k][start:end] if d >= elapsed]
    return np.asarray([v for _, v in survivors]), max((t for t, _ in survivors), default=None)


def rank_interval(values):
    ordered = np.sort(values)
    n = len(ordered)
    # Integer ranks keep the fixed 0.1 tails exact at n=9,19,... .
    lower_rank = (n+1)//10
    upper_rank = n+1-lower_rank
    finite = upper_rank <= n
    return dict(status='finite_rank_interval' if finite else
        'insufficient_upper_rank' if n else 'no_completed_survivor',
        n=n, lower_rank=lower_rank, upper_rank=upper_rank, finite=finite,
        lower_s=float(ordered[lower_rank-1]) if lower_rank else 0.,
        upper_s=float(ordered[upper_rank-1]) if finite else None)
