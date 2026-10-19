#!/usr/bin/env python3
"""Consumed V123 paths only: quota accounts, exact timing and local geometry."""

import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v116_quota_regularity as geometry
import run_freqduet_v123_eta_request_pilot as frozen


PARTS = ('release', 'raw_shadow', 'request_shift', 'innovation_quota', 'other_elapsed')
DEPARTURE_PARTS = ('arrival_time', 'service_elapsed', 'current_action')
ACCOUNT_FIELDS = ('issued_s', 'executed_s', 'expired_s', 'remaining_s')


def holding_parts(row, actual):
    if row is None:
        return dict(raw_shadow=actual, request_shift=0., innovation_quota=0.)
    if row['chosen_s'] != actual or row['executed_s'] != actual:
        raise AssertionError('trace command does not reproduce completed holding')
    request = row.get('request_shadow_s', row['shadow_s'])
    return dict(raw_shadow=row['shadow_s'], request_shift=request-row['shadow_s'],
        innovation_quota=actual-request)


def read_path(root, seed, day, method):
    name = f'{seed}_{day}_{method}_0'
    pair = json.loads((root/'pairs'/f'{seed}_{day}_{method}.json').read_text())
    if (pair['seed'], pair['day'], pair['method']) != (seed, day, method):
        raise AssertionError('consumed path identity differs from V123 grid')
    rows = frozen.prior.read_trace(root/'traces'/(name+'.jsonl.gz'))
    trace = {geometry.key(r): r for r in rows}
    if len(trace) != len(rows):
        raise AssertionError('consumed decision identities are duplicated')
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        events = {k: [dict(zip(fields, r.tolist())) for r in record[k]] for k in ('arrivals', 'departures')}
        trips = len(record['launches'])
    ends = {geometry.key(r): r for r in events['departures']}
    cumulative = defaultdict(Counter)
    arrivals, departures, holding = [], {}, 0.
    for row in events['arrivals']:
        key = geometry.key(row)
        end = ends[key]
        actual = end['realized_holding']-row['realized_holding']
        np.testing.assert_allclose(sum(cumulative[key[0]].values()), row['realized_holding'], atol=1e-8, rtol=0)
        arrivals.append(dict(row, **{k: cumulative[key[0]][k] for k in PARTS[1:-1]}))
        if method != frozen.BILATERAL and key not in trace and actual != 0:
            raise AssertionError('quota holding lacks a decision receipt')
        for k, v in holding_parts(trace.get(key), actual).items():
            cumulative[key[0]][k] += v
        holding += actual
        departures[key] = dict(end, arrival_time=row['time_s'], current_action=actual,
            service_elapsed=end['time_s']-row['time_s']-actual)
    departures = [departures[geometry.key(r)] for r in events['departures'] if geometry.key(r) in departures]
    a, d = geometry.series(arrivals, PARTS), geometry.series(departures, DEPARTURE_PARTS)
    np.testing.assert_allclose(a['cv_squared'], pair['results']['cv_squared'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(holding/trips, pair['results']['holding_s_per_trip'], atol=1e-6, rtol=0)
    return dict(pair=pair, trace=trace, arrival=a, departure=d, departures=departures)


def forecast_and_expiry(trace):
    counts, forecasts = Counter(), []
    for row in trace.values():
        if row['expired_s']:
            reason = 'reference' if row['reset_reason'] != 'continued' else 'balanced'
            counts[reason+'_expired_s'] += row['expired_s']
            counts[reason+'_expiry_events'] += 1
        flip = row['raw_feedback_error_s'] > 0 and row['feedback_error_s'] <= 0
        counts['prediction_balance_flips'] += int(flip)
        counts['balance_flip_expired_s'] += row['expired_s'] if flip else 0.
        counts['component_requests'] += int(row['component_used'])
        if not row['component_used']:
            continue
        key, evidence = geometry.key(row), row['evidence']
        follower = (evidence['predicted_follower_trip_id'], key[1], key[2])
        if follower not in trace:
            counts['missing_same_follower_ready'] += 1
            continue
        if trace[follower]['evidence']['bus_id'] != evidence['predicted_follower_bus_id']:
            counts['same_follower_bus_mismatch'] += 1
            continue
        gap = trace[follower]['evidence']['ready_time_s']-evidence['ready_time_s']
        if gap < 0:
            counts['negative_same_follower_ready_gap'] += 1
            continue
        forecasts.append((row['request_gap_s']-gap, evidence['follower_gap_prediction_s']-gap))
    errors = np.asarray(forecasts)
    if len(errors):
        counts.update(dict(labelled_component_requests=len(errors), split_bias_s=float(errors[:, 0].mean()),
            raw_bias_s=float(errors[:, 1].mean()), split_mae_s=float(np.abs(errors[:, 0]).mean()),
            raw_mae_s=float(np.abs(errors[:, 1]).mean())))
    return dict(counts)


def local_restores(path):
    groups, cohorts = defaultdict(list), defaultdict(Counter)
    for row in path['departures']:
        groups[geometry.key(row)[1:]].append(row)
    for rows in groups.values():
        times = np.asarray([r['time_s'] for r in rows])
        for i, event in enumerate(rows):
            row = path['trace'].get(geometry.key(event))
            if row is None or row['shadow_s'] <= row['chosen_s']:
                continue
            cut = row['shadow_s']-row['chosen_s']
            shifted = times.copy()
            shifted[i] += cut
            delta = geometry.geometry_change(times, shifted, path['departure'])
            useful = delta['cv_squared_change'] < -1e-12
            harmful = delta['cv_squared_change'] > 1e-12
            names = ['all_raw_shadow_cuts']
            if row['expired_s'] > 0:
                names.append('expiry_here')
            if row.get('raw_feedback_error_s', 0.) > 0 >= row['feedback_error_s']:
                names.append('prediction_balance_flip')
            for name in names:
                cohorts[name].update(dict(decisions=1, cut_s=cut, useful_restores=int(useful),
                    harmful_restores=int(harmful), useful_cut_s=cut*useful,
                    harmful_cut_s=cut*harmful, boundary_events=int(i in (0, len(rows)-1))))
    return {k: dict(v) for k, v in cohorts.items()}


def review(root, out):
    parent = json.loads((root/'summary/status.json').read_text())
    if parent['status'] != 'v123_matched_request_pilot_complete' or parent['rollouts'] != len(frozen.ITEMS):
        raise AssertionError('all original V123 endpoints must be available')
    cells = []
    for seed, day in itertools.product(frozen.SEEDS, frozen.DAYS):
        paths = {m: read_path(root, seed, day, m) for m in frozen.METHODS}
        effects = {}
        for name, (candidate, base) in frozen.CONTRASTS.items():
            a, b = paths[candidate], paths[base]
            effects[name+'_arrival'] = geometry.effect(b['arrival'], a['arrival'], PARTS)
            effects[name+'_departure'] = geometry.effect(b['departure'], a['departure'], DEPARTURE_PARTS)
            expected = next(r for r in parent['actor_day'] if (r['seed'], r['day']) == (seed, day))['effects'][name]
            np.testing.assert_allclose(effects[name+'_arrival']['cv_squared'], expected['cv_squared'], atol=1e-10, rtol=0)
        accounts = {m: {k: paths[m]['pair']['accounting'][k] for k in ACCOUNT_FIELDS}
            for m in (frozen.RAW, frozen.PRIMARY)}
        effects['split_vs_raw_accounts'] = {k: accounts[frozen.PRIMARY][k]-accounts[frozen.RAW][k] for k in ACCOUNT_FIELDS}
        quota = paths[frozen.PRIMARY]['pair']['accounting']
        details = forecast_and_expiry(paths[frozen.PRIMARY]['trace'])
        np.testing.assert_allclose(details.get('reference_expired_s', 0.)+details.get('balanced_expired_s', 0.), quota['expired_s'], atol=0, rtol=0)
        cells.append(dict(seed=seed, day=day, effects=effects, accounts=accounts, forecast_expiry=details,
            local_restores={m: local_restores(paths[m]) for m in (frozen.RAW, frozen.PRIMARY)},
            absolute_moments={m: {kind: {k: p[kind][k] for k in ('mean_s', 'variance_s2', 'cv_squared')}
                for kind in ('arrival', 'departure')} for m, p in paths.items()}))
        print('V123_READONLY_CELL_DONE', seed, day, flush=True)
    for method in (frozen.RAW, frozen.PRIMARY):
        for field in ACCOUNT_FIELDS:
            if sum(r['accounts'][method][field] for r in cells) != parent['accounting'][method][field]:
                raise AssertionError('consumed account coverage differs from original V123')
    frozen.pilot.write_json(out/'status.json', dict(status='v123_readonly_request_mechanism_complete',
        effects=frozen.inference.describe(cells, 'effects'), actor_day=cells,
        axes={str(axis): {str(v): {kind: {k: float(np.mean([r['effects'][kind][k] for r in cells if r[axis] == v]))
            for k in cells[0]['effects'][kind]} for kind in cells[0]['effects']}
            for v in sorted({r[axis] for r in cells})} for axis in ('seed', 'day')},
        consumed_paths=len(frozen.ITEMS), original_endpoints_exact=True, new_rollouts=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Consumed development paths only. Covariance components and inter-arm account identities '
            'are exact descriptive accounting, not causal mediation. Raw shadows use each visited state, '
            'not a matched baseline trajectory. A single shadow restoration holds other completed '
            'departure times fixed and is not an executed full-day intervention. Forecast labels '
            'are same predicted-trip ready times on the altered policy path, not new validation.'))
    print('V123_READONLY_MECHANISM_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v123-root', type=Path, required=True)
    p.add_argument('--out-summary', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v123_request_mechanism.py', '-v'], check=True)
    review(a.v123_root, a.out_summary)
