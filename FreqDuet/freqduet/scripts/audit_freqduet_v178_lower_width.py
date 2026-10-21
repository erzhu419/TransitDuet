#!/usr/bin/env python3
"""Read saved V177 queries only: lower-tail variation and envelope loss accounting."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v177_confirmation as saved
import audit_freqduet_v174_upper_tails as prior


read, write, require, close, stream = saved.read, saved.write, saved.require, saved.fit.close, saved.stream
SCOPES = ('clearance', 'hour13_moving_launch')
CHANNELS = ('running', 'service', 'residence')
METRICS = ('below', 'above', 'point_bias_s', 'running_bias_s', 'service_bias_s',
    'residence_bias_s', 'qhigh_s', 'lower_excess_s', 'below_running_s',
    'below_service_s', 'below_residence_s', 'below_qhigh_s', 'expansion_s',
    'upper_saving_s', 'envelope_score_delta_s', 'repaired_above', 'profitable_expansion')
INTERVALS = ('below', 'above', 'expansion_s', 'upper_saving_s', 'envelope_score_delta_s')


def parents(args):
    registration = read(args.v177_root/'registration.json')
    summary = read(args.v177_root/'summary/status.json')
    model = read(args.v176_root/'model.json')
    require(summary['status'] == 'v177_independent_confirmation_complete' and
        registration['execution_source'] == summary['execution_source'] == 'd1454cfc39' and
        summary['map_source'] == model['execution_source'] == '36108d4f49' and
        registration['sealed_model'] == model, 'requires completed, frozen V177/V176')
    require(summary['paths'] == 1536 and summary['shards'] == 96 and not summary['interval_fit'] and
        summary['map_updates'] == summary['new_training'] == 0 and not summary['main_policy_changed'] and
        all(not summary[k+'_changed'] for k in ('v151', 'v146', 'v167', 'v172', 'v176')),
        'frozen confirmation incomplete or changed')
    r = registration['reservation']
    require(len(r['actors']) == 12 and r['confirmation_days'] == list(range(3352001, 3352065)), 'cohort differs')
    return r, summary, model


def features(q, entry):
    c, stage = q['condition_values'], q['native_stage']
    days = None if entry is None else entry['day_count']
    support = 'unseen' if days is None else '<19' if days < 19 else '19-31' if days < 32 else '32-47' if days < 48 else '48-64'
    age, n = stage.get('latest_survivor_lag_s'), stage.get('survivor_n')
    require(age is None or age >= 0., 'query support age is negative')
    age_band = 'not_applicable' if age is None else '<=900' if age <= 900 else '<=1800' if age <= 1800 else '<=3600' if age <= 3600 else '>3600'
    count_band = 'not_applicable' if n is None else '<9' if n < 9 else '9-18' if n < 19 else '>=19'
    elapsed = q['query_time_s']-q['observable']['demand_end_time_s']
    elapsed_band = 'active' if elapsed < 0 else '[0,900)' if elapsed < 900 else '[900,1800)' if elapsed < 1800 else '>=1800'
    position = q['observable']['position']
    return dict(all='all', stage=c['native_stage_kind'], horizon=c['predicted_horizon_band'],
        position_status=c['position_status'], direction=str(c['direction']), target_stop=str(c['target_stop']),
        key_days=support, native_support=c['native_support_class'], native_survivors=count_band,
        support_age_s=age_band, clearance_elapsed_s=elapsed_band,
        remaining_stops=None if position is None else str(position['remaining_stops']),
        stage_horizon=c['native_stage_kind']+'|'+str(c['predicted_horizon_band']))


def diagnostics(q, p, r, entry):
    label, error = r['label'], r['residual_s']
    require(label['status'] == 'resolved' and error is not None and all(error[c] is not None for c in CHANNELS),
        'finite query missing joint label/components')
    truth, point = label['full_eta_s'], q['point_s']
    close(sum(error[c] for c in CHANNELS), point-truth)
    require(label['available_time_s'] == label['target_ready_time_s'] >= q['query_time_s'], 'label clock differs')
    lo, hi = p['bounds']
    rl, ru = p['temporal_bounds']
    require(lo == rl and hi >= ru, 'frozen envelope geometry differs')
    below, above = float(truth < lo), float(truth > hi)
    width = hi-ru
    saving = 10.*(max(truth-ru, 0.)-max(truth-hi, 0.))
    vector = (below, above, point-truth, *(error[c] for c in CHANNELS), entry['qhigh_s'],
        max(lo-truth, 0.), *(below*error[c] for c in CHANNELS), below*entry['qhigh_s'],
        width, saving, width-saving, float(truth > ru and truth <= hi), float(width-saving < 0.))
    return np.array(vector)


def weighted_profiles(base, groups):
    eligible_episodes = len(base)
    finite_episodes = sum(v[1] > 0 for v in base.values())
    require(eligible_episodes > 0 and finite_episodes > 0, 'empty original target cell')
    totals = defaultdict(lambda: np.zeros(3+len(METRICS)))
    counts = defaultdict(lambda: np.zeros(2, dtype=int))
    for ep, entries in groups.items():
        ne, nf = base[ep][:2]
        for identity, v in entries.items():
            totals[identity][:2] += v[:2]/(ne*eligible_episodes)
            if nf:
                totals[identity][2] += v[1]/(nf*finite_episodes)
                totals[identity][3:] += v[2:]/(nf*finite_episodes)
            counts[identity] += v[:2].astype(int)
    all_values = totals['all', 'all']
    for axis in {identity[0] for identity in totals}:
        actual = sum((v for (a, _), v in totals.items() if a == axis), np.zeros(len(all_values)))
        require(np.allclose(actual, all_values, rtol=1e-9, atol=1e-8), 'state partition does not sum to original cell')
    close(all_values[0], 1.)
    close(all_values[2], 1.)
    return [dict(axis=a, value=v, values=values.tolist(), queries=int(counts[a, v][0]),
        finite_queries=int(counts[a, v][1])) for (a, v), values in totals.items()]


def scan(args, actor, day, mode, tables, model):
    name = f'{actor}_{day}_{mode}'
    original = read(args.v177_root/'cells'/(name+'.json'))
    require(original['execution_source'] == 'd1454cfc39' and original['map_source'] == saved.MAP_SOURCE and
        original['bounds_saved_before_labels'] and not original['main_policy_changed'], 'path source/freeze differs')
    base = {s: {} for s in SCOPES}
    groups = {s: defaultdict(dict) for s in SCOPES}
    sources = [stream(args.v177_root/'confirmation'/d/(name+'.jsonl.gz'))
        for d in ('queries', 'interval_predictions', 'residuals')]
    count = 0
    for q, p, r in zip_longest(*sources):
        require(q is not None and p is not None and r is not None and q['query_key'] == p['query_key'] == r['query_key'] and
            q['query_time_s'] == p['query_time_s'] == r['query_time_s'], 'saved streams misaligned')
        count += 1
        selected = prior.scopes(q) if q['denominator_eligible'] else ()
        if not selected:
            continue
        require((q['actor'], q['day'], q['mode']) == (actor, day, mode) and
            q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'query state or unavailable load differs')
        require(saved.prediction(q, *tables, model['frozen_protocol']) == p, 'saved frozen bounds differ')
        entry = None if p['candidate_key'] is None else tables[0].get(tuple(p['candidate_key']))
        vector = diagnostics(q, p, r, entry) if p['finite'] else None
        scored = saved.values(p, r['label'])
        require(not p['finite'] or scored is not None, 'finite reference/label missing')
        ep, dimensions = tuple(q['episode']), features(q, entry)
        for scope in selected:
            row = base[scope].setdefault(ep, np.zeros(3+len(saved.METRICS)))
            row[:3] += (1, int(p['finite']), int(scored is not None))
            if scored is not None:
                row[3:] += scored
            for axis, value in dimensions.items():
                group = groups[scope][ep].setdefault((axis, value), np.zeros(2+len(METRICS)))
                group[:2] += (1, int(p['finite']))
                if vector is not None:
                    group[2:] += vector
    require(count == original['scoring_counts']['requests'], 'query cohort incomplete')
    cells = []
    for scope in SCOPES:
        reproduced = saved.collapse(base[scope], 0, 0)
        for field, value in reproduced.items():
            expected = original['cells'][scope][field]
            require(expected is None, 'missing state changed') if value is None else close(value, expected)
        profiles = weighted_profiles(base[scope], groups[scope])
        all_values = next(v['values'] for v in profiles if v['axis'] == 'all')
        close(all_values[1], reproduced['finite_fraction'])
        for metric, endpoint in (('below', 'below'), ('above', 'above'), ('expansion_s', 'temporal_width_delta_s'),
                ('envelope_score_delta_s', 'temporal_score_delta_s'), ('repaired_above', 'v172_repaired_above')):
            close(all_values[3+METRICS.index(metric)], reproduced[endpoint])
        cells.append(dict(actor=actor, day=day, mode=mode, scope=scope, profiles=profiles, endpoint_reproduced=True))
    return cells


def bucket(args, actor, mode):
    reservation, _, model = parents(args)
    tables = saved.map_tables(args.v176_root, args.v172_root, args.v167_root, actor, mode)
    rows = []
    for i, day in enumerate(reservation['confirmation_days'], 1):
        rows.extend(scan(args, actor, day, mode, tables, model))
        print('V178_DAY_READ', actor, mode, i, '/64', flush=True)
    write(args.out_root/'cells'/f'{actor}_{mode}.json', dict(execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], cells=rows))
    return rows


def bootstrap_weights(actors, days):
    rng = np.random.RandomState(20260803)
    a = rng.randint(0, actors, size=(5000, actors))
    d = rng.randint(0, days, size=(5000, days))
    ac = (a[:, :, None] == np.arange(actors)).mean(axis=1)
    dc = (d[:, :, None] == np.arange(days)).mean(axis=1)
    return (ac[:, :, None]*dc[:, None, :]).reshape(5000, actors*days)


def profile_readout(matrix, weights):
    mean, draws = matrix.mean(axis=0), weights@matrix
    def ratio(column, denominator):
        point = None if mean[denominator] == 0 else float(mean[column]/mean[denominator])
        supported = bool((draws[:, denominator] > 0).all())
        ci = None if not supported else np.percentile(draws[:, column]/draws[:, denominator], [2.5, 97.5]).tolist()
        return dict(mean=point, ci95=ci)
    return dict(eligible_mass=float(mean[0]), finite_mass=float(mean[1]), score_mass=float(mean[2]),
        observed_cells=int(np.count_nonzero(matrix[:, 0])), finite_cells=int(np.count_nonzero(matrix[:, 2])),
        finite_fraction=ratio(1, 0), conditional={m: ratio(3+METRICS.index(m), 2) for m in INTERVALS},
        conditional_means={m: None if mean[2] == 0. else float(mean[3+i]/mean[2]) for i, m in enumerate(METRICS)},
        contributions={m: dict(mean=float(mean[3+i]), ci95=np.percentile(draws[:, 3+i], [2.5, 97.5]).tolist())
            for i, m in enumerate(METRICS) if m in ('below', 'envelope_score_delta_s')})


def correlation(x, y):
    return None if np.std(x) == 0. or np.std(y) == 0. else float(np.corrcoef(x, y)[0, 1])


def summarize(rows, actors, days, expected):
    index = {(r['actor'], r['day']): r for r in rows}
    require(len(index) == len(rows) == len(actors)*len(days) and set(index) == set(product(actors, days)), 'crossed grid incomplete')
    matrices, query_counts = {}, defaultdict(lambda: [0, 0])
    for i, identity in enumerate(product(actors, days)):
        for p in index[identity]['profiles']:
            key = p['axis'], p['value']
            matrices.setdefault(key, np.zeros((len(rows), 3+len(METRICS))))[i] = p['values']
            query_counts[key][0] += p['queries']
            query_counts[key][1] += p['finite_queries']
    weights, profiles = bootstrap_weights(len(actors), len(days)), []
    for (axis, value), matrix in sorted(matrices.items(), key=lambda v: str(v[0])):
        profiles.append(dict(axis=axis, value=value, queries=query_counts[axis, value][0],
            finite_queries=query_counts[axis, value][1], **profile_readout(matrix, weights)))
    all_matrix = matrices['all', 'all']
    global_stats = next(r for r in profiles if r['axis'] == 'all')
    for metric, old in (('below', 'below'), ('above', 'above'), ('expansion_s', 'temporal_width_delta_s'),
            ('envelope_score_delta_s', 'temporal_score_delta_s')):
        actual, original = global_stats['conditional'][metric], expected['estimates'][old]
        close(actual['mean'], original['mean'])
        for a, b in zip(actual['ci95'], original['ci95']):
            close(a, b)
    variation = {m: prior.variation(all_matrix[:, 3+METRICS.index(m)].reshape(len(actors), len(days)))
        for m in ('below', 'point_bias_s', 'running_bias_s', 'service_bias_s', 'residence_bias_s', 'qhigh_s', 'lower_excess_s')}
    by_day = []
    for i, day in enumerate(days):
        row = all_matrix.reshape(len(actors), len(days), -1)[:, i].mean(axis=0)
        by_day.append(dict(day=day, **dict(zip(METRICS, row[3:].tolist()))))
    daily = {m: np.array([r[m] for r in by_day]) for m in METRICS}
    centered = daily['point_bias_s']-daily['point_bias_s'].mean()
    variance = float(np.mean(centered**2))
    covariance = {c: float(np.mean((daily[c+'_bias_s']-daily[c+'_bias_s'].mean())*centered)) for c in CHANNELS}
    close(sum(covariance.values()), variance)
    return dict(profiles=profiles, descriptive_variance=variation, by_day=by_day,
        day_component_covariance=dict(total_bias_variance=variance, covariance=covariance,
            signed_fractions=None if variance == 0. else {c: v/variance for c, v in covariance.items()},
            lower_rate_correlations={m: correlation(daily['below'], daily[m])
                for m in ('point_bias_s', 'running_bias_s', 'service_bias_s', 'residence_bias_s', 'qhigh_s')}),
        highest_lower_days=sorted(by_day, key=lambda v: v['below'], reverse=True)[:5])


def prepare(args):
    reservation, _, model = parents(args)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(Path(__file__).resolve().parents[1]/'tests'),
        '-p', 'test_v178_*.py', '-v'], check=True)
    actor, day = reservation['actors'][0], reservation['confirmation_days'][0]
    tables = saved.map_tables(args.v176_root, args.v172_root, args.v167_root, actor, 'old')
    cells = scan(args, actor, day, 'old', tables, model)
    write(args.out_root/'prepare/summary/status.json', dict(status='v178_readonly_preflight_passed',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], actor=actor, day=day,
        endpoints_reproduced=all(c['endpoint_reproduced'] for c in cells), new_replays=0, new_training=0))
    print('V178_READONLY_PREFLIGHT_PASSED', flush=True)


def run(args):
    reservation, _, _ = parents(args)
    cases = list(product(reservation['actors'], ('old', 'handoff')))
    require(0 <= args.start < args.end <= 24, 'invalid bucket range')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(bucket, args, a, m) for a, m in cases[args.start:args.end]]
        for job in as_completed(jobs):
            job.result()
    write(args.out_root/'receipts'/f'{args.start}-{args.end}'/'status.json',
        dict(status='v178_readonly_shard_complete', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
            start=args.start, end=args.end))


def aggregate(args):
    reservation, original, model = parents(args)
    actors, days = sorted(reservation['actors']), sorted(reservation['confirmation_days'])
    rows = []
    for actor, mode in product(actors, ('old', 'handoff')):
        bucket = read(args.out_root/'cells'/f'{actor}_{mode}.json')
        require(bucket['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'bucket source differs')
        rows.extend(bucket['cells'])
    results, details = [], []
    for mode, scope in product(('old', 'handoff'), SCOPES):
        expected = next(r for r in original['readouts'] if (r['mode'], r['scope']) == (mode, scope))
        result = summarize([r for r in rows if (r['mode'], r['scope']) == (mode, scope)], actors, days, expected)
        details.append(dict(mode=mode, scope=scope, **result))
        results.append(dict(mode=mode, scope=scope, **{k: v for k, v in result.items() if k != 'by_day'}))
    require(read(args.v176_root/'model.json') == model, 'sealed header changed')
    write(args.out_root/'details/results.json', details)
    write(args.out_root/'summary/status.json', dict(status='v178_readonly_lower_width_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], input_source='d1454cfc39', paths=1536,
        endpoints_and_crossed_intervals_reproduced=True, results=results, bootstrap_draws=5000,
        bootstrap_seed=20260803, interval_fit=False, map_updates=0, new_replays=0, new_training=0,
        main_policy_changed=False, interpretation='Consumed-cohort descriptive profiles, not fitted gates or causal effects.'))
    print('V178_READONLY_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'aggregate'))
    for name in ('out-root', 'v177-root', 'v176-root', 'v172-root', 'v167-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'aggregate': aggregate}[args.command](args)
