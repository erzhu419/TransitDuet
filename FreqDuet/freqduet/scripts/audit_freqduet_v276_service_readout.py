#!/usr/bin/env python3
"""Read existing V276 traffic archives once; never replay or optimize a policy."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import pickle

import numpy as np


def gaps_from_arrivals(arrivals):
    last, gaps = {}, []
    for row in arrivals:
        key = bool(row['direction']), int(row['station_id'])
        now = float(row['time_s'])
        if key in last:
            assert now >= last[key], 'arrival archive is not chronological within stop'
            gaps.append(dict(direction=key[0], station=key[1], hour=int(now // 3600),
                             gap_s=now-last[key]))
        last[key] = now
    return gaps


def cv_components(arrivals):
    gaps = gaps_from_arrivals(arrivals)
    values = np.array([r['gap_s'] for r in gaps])
    mean, variance = float(values.mean()), float(values.var())
    denominator = len(values) * max(mean, 1.) ** 2
    groups = defaultdict(lambda: dict(count=0, sum_gap_s=0., contribution=0.))
    for row in gaps:
        key = row['direction'], row['station'], row['hour']
        group = groups[key]
        group['count'] += 1
        group['sum_gap_s'] += row['gap_s']
        group['contribution'] += (row['gap_s']-mean)**2 / denominator
    cv2 = variance / max(mean, 1.)**2
    assert abs(sum(r['contribution'] for r in groups.values())-cv2) < 1e-12
    return dict(count=len(values), mean_s=mean, variance_s2=variance, cv2=cv2), groups


def describe(values):
    if not values:
        return dict(count=0)
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), mae=float(np.abs(a).mean()),
                p10=float(np.quantile(a, .1)), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)))


def prediction_readout(trace, expected):
    predictions = trace['resource_predictions']
    assert len(predictions) == expected
    changed = {(r['time_s'], r['bus'], r['trip'], r['station']) for r in trace['graph_queries']
               if r['shadow_action_s'] != r['native_action_s']}
    recorded = {(p['query_s'], p['bus'], p['trip'], p['station']) for p in predictions}
    assert len(changed) == len(recorded) == expected and changed == recorded
    errors, deltas, missing = defaultdict(list), defaultdict(list), defaultdict(int)
    vehicles, own_queries = set(), 0
    for p in predictions:
        for v in p['vehicles']:
            vehicles.add((v['bus'], v['trip']))
            own = (v['bus'], v['trip']) == (p['bus'], p['trip'])
            own_queries += own
            for scope in ('all_active', 'current_bus') if own else ('all_active',):
                for metric in ('return_s', 'next_launch_s', 'remaining_trip_h_s'):
                    actual = v['actual'][metric]
                    for branch in ('full', 'fallback'):
                        key = scope + '/' + branch + '/' + metric
                        predicted = v[branch][metric]
                        if predicted is None or actual is None:
                            missing[key] += 1
                        else:
                            errors[key].append(predicted-actual)
                        reference = v['reference'][metric]
                        if predicted is not None and reference is not None:
                            deltas[key].append(predicted-reference)
    assert own_queries == expected
    return dict(intervention_queries=expected, unique_vehicle_trips=len(vehicles),
        forecast_minus_executed_actual={k:describe(v) for k,v in sorted(errors.items())},
        model_branch_minus_model_reference={k:describe(v) for k,v in sorted(deltas.items())},
        unavailable_labels=dict(missing),
        interpretation='Repeated queries share vehicle/trip outcomes; no independent-query inference. '
            'Full/fallback forecasts are compared to the actually replanned path, not counterfactual truth. '
            'Reference differences are model-only; no observed baseline counterfactual label.')


def contrast(base, candidate):
    b, bg = cv_components(base['arrivals'])
    c, cg = cv_components(candidate['arrivals'])
    def grouped(axes):
        groups = defaultdict(lambda: dict(base_count=0, candidate_count=0,
            base_contribution=0., candidate_contribution=0.))
        for name, rows in (('base', bg), ('candidate', cg)):
            for key, row in rows.items():
                group = groups[tuple(key[i] for i in axes)]
                group[name+'_count'] += row['count']
                group[name+'_contribution'] += row['contribution']
        result = []
        for key, row in groups.items():
            row = dict(key=list(key), **row)
            row['delta_cv2'] = row['candidate_contribution']-row['base_contribution']
            result.append(row)
        assert abs(sum(r['delta_cv2'] for r in result)-(c['cv2']-b['cv2'])) < 1e-12
        return sorted(result, key=lambda r:r['delta_cv2'], reverse=True)
    variance_term = (c['variance_s2']-b['variance_s2'])/max(b['mean_s'], 1.)**2
    normalization_term = c['variance_s2']*(1/max(c['mean_s'],1.)**2-1/max(b['mean_s'],1.)**2)
    assert abs(variance_term+normalization_term-(c['cv2']-b['cv2'])) < 1e-12
    return dict(base=b, candidate=c, delta_cv2=c['cv2']-b['cv2'],
        variance_term=variance_term, normalization_term=normalization_term,
        by_station=grouped((0,1)), by_hour=grouped((2,)),
        largest_station_hour_increases=grouped((0,1,2))[:8],
        largest_station_hour_decreases=grouped((0,1,2))[-8:],
        interpretation='Exact additive accounting with each branch own pooled mean/denominator; '
            'not causal per-station effects or matched within-hour samples.')


def run(a):
    manifest = json.loads((a.replacement/'summary/results.json').read_text())
    assert manifest['status'] == 'twelve_development_paths_complete'
    summaries, rows = {}, manifest['by_day']
    for day in (2076001,2076002,2076003,2076004):
        traces, cells = {}, {}
        for index, row in enumerate(rows):
            if row['day'] != day:
                continue
            # The manifest preserves the original and three replacement origins.
            origin = Path(manifest['path_origins'][str(index)]['root'])
            assert origin in (a.original, a.replacement)
            with gzip.open(origin/'traces'/f'p{index}.pkl.gz', 'rb') as stream:
                trace = pickle.load(stream)
            value, _ = cv_components(trace['arrivals'])
            assert np.isclose(value['cv2'], row['results']['cv_squared'], atol=1e-12, rtol=0)
            assert len(trace['arrivals']) == 5240 and value['count'] == 5200
            traces[row['arm']], cells[row['arm']] = trace, row
        candidate = traces['native_continuation_apc']
        graph = cells['native_continuation_apc']['graph']
        summaries[str(day)] = dict(
            candidate_vs_v275=contrast(traces['frozen_v275_apc'], candidate),
            candidate_vs_fixed=contrast(traces['fixed360_bilateral025'], candidate),
            resource_prediction=prediction_readout(candidate, graph['resource_prediction_queries']))
        print('V276_READOUT_DAY_COMPLETE', day, flush=True)
    result = dict(source=a.source, status='existing_trajectories_readout_complete', days=summaries,
        endpoints_reproduced=12, intervention_queries=sum(
            r['resource_prediction']['intervention_queries'] for r in summaries.values()),
        physical_starts=0, training_updates=0, main_policy_changed=False,
        v276_gate_unchanged=True, raw_archives_remain_server_side=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('V276_SERVICE_READOUT_COMPLETE', result['intervention_queries'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--replacement', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', required=True)
    run(parser.parse_args())
