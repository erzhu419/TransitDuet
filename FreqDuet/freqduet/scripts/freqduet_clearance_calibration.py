"""Two outward tail offsets; native finite support and point forecasts stay fixed."""

from collections import defaultdict

import numpy as np

from freqduet_interval_validation import describe, mean


def eligible(row, cutoff):
    return row['kind'] == 'service' and row['time_s'] >= cutoff and row['rank']['finite']


def weighted_upper_tail(values, weights):
    order = np.argsort(values, kind='stable')
    cdf = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(values)[order][np.searchsorted(cdf, .9*cdf[-1], side='left')])


def fit(cells, actors, modes, days):
    expected = {(s, d, m) for s in actors for d in days for m in modes}
    keys = [(c['seed'], c['day'], c['mode']) for c in cells]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise AssertionError('calibration grid differs from registration')
    by_day, represented = defaultdict(list), defaultdict(set)
    selected_n, unresolved = 0, 0
    for c in cells:
        episodes = defaultdict(list)
        for r in c['rows']:
            if eligible(r, c['cutoff']):
                selected_n += 1
                if r['label_status'] != 'resolved':
                    unresolved += 1
                else:
                    episodes[tuple(r['episode'])].append(r)
        if episodes:
            by_day[c['day']].append(episodes)
            represented[c['seed'], c['mode']].add(c['day'])
    support = {f'{s}_{m}': len(represented[s, m]) for s in actors for m in modes}
    result = dict(status='calibration_inconclusive', selected_queries=selected_n,
        finite_unresolved_n=unresolved, represented_days=len(by_day),
        nonempty_cells=sum(len(v) for v in by_day.values()), days_by_actor_mode=support)
    if unresolved or set(by_day) != set(days) or min(support.values()) < 9:
        return result
    lower, upper, weights = [], [], []
    # Equal day mass, then nonempty actor/mode cells, episodes and queries.
    for day in days:
        for episodes in by_day[day]:
            for rows in episodes.values():
                weight = 1./(len(days)*len(by_day[day])*len(episodes)*len(rows))
                for r in rows:
                    lower.append(max(0., r['rank']['lower_s']-r['truth_s']))
                    upper.append(max(0., r['truth_s']-r['rank']['upper_s']))
                    weights.append(weight)
    result.update(status='sealed', lower_extension_s=weighted_upper_tail(lower, weights),
        upper_extension_s=weighted_upper_tail(upper, weights), tail_probability=.1,
        estimator='fixed weighted empirical 0.9 quantile of each positive endpoint error')
    return result


def apply(row, model, cutoff):
    if model['status'] != 'sealed':
        raise AssertionError('confirmation requires sealed calibration')
    if not eligible(row, cutoff):
        return dict(row, calibration_applied=False)
    bounds = dict(row['rank'], status='clearance_calibrated_interval',
        lower_s=max(0., row['rank']['lower_s']-model['lower_extension_s']),
        upper_s=row['rank']['upper_s']+model['upper_extension_s'])
    resolved = row['label_status'] == 'resolved'
    return dict(row, base_rank=row['rank'], rank=bounds, calibration_applied=True,
        covered=None if not resolved else bounds['lower_s'] <= row['truth_s'] <= bounds['upper_s'],
        below=None if not resolved else row['truth_s'] < bounds['lower_s'],
        above=None if not resolved else row['truth_s'] > bounds['upper_s'])


def interval_score(row):
    lo, hi, truth = row['rank']['lower_s'], row['rank']['upper_s'], row['truth_s']
    return hi-lo+10.*(max(0., lo-truth)+max(0., truth-hi))


SCOPES = ('service_all', 'service_hour13', 'service_clearance', 'service_active', 'ready_all')


def select(row, scope, cutoff):
    kind, group = scope.split('_', 1)
    return row['kind'] == kind and (group == 'all' or
        group == 'hour13' and row['hour'] == 13 or
        group == 'clearance' and row['time_s'] >= cutoff or
        group == 'active' and row['time_s'] < cutoff)


def paired_stats(base, candidate):
    episodes = defaultdict(list)
    for b, c in zip(base, candidate):
        if b['rank']['finite'] != c['rank']['finite']:
            raise AssertionError('calibration changed native abstention')
        if b['rank']['finite'] and b['label_status'] == 'resolved':
            episodes[tuple(b['episode'])].append((interval_score(b), interval_score(c)))
    values = [np.mean(v, axis=0) for v in episodes.values()]
    return dict(episodes=len(values), base_score=mean([v[0] for v in values]),
        candidate_score=mean([v[1] for v in values]),
        score_delta=mean([v[1]-v[0] for v in values]))


def cell_stats(rows, model, cutoff):
    candidate = [apply(r, model, cutoff) for r in rows]
    result = {}
    for scope in SCOPES:
        chosen = [i for i, r in enumerate(rows) if select(r, scope, cutoff)]
        b, c = [[items[i] for i in chosen] for items in (rows, candidate)]
        result[scope] = dict(base=describe(b), candidate=describe(c), paired=paired_stats(b, c))
    return result
