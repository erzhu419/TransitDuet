#!/usr/bin/env python3
"""Read consumed V185 archives: upper-tail day variation and lower-bound costs."""

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

import run_freqduet_v185_confirmation as saved
import audit_freqduet_v178_lower_width as profiles
import audit_freqduet_v174_upper_tails as upper


read, write, require, close = saved.read, saved.write, saved.require, saved.fit.close
SCOPES = ('hour2', 'clearance', 'hour13', 'hour13_moving_launch')
AXES = ('all', 'stage', 'horizon', 'remaining_stops', 'clearance_elapsed_s',
    'key_days', 'native_survivors', 'support_age_s', 'stage_horizon')
CHANNELS = ('running', 'service', 'residence')
METRICS = ('above', 'upper_excess_s', 'upper_headroom_s', 'point_bias_s',
    *(c+'_bias_s' for c in CHANNELS), 'below', 'reference_below', 'width_cost_s',
    'lower_saving_s', 'score_delta_s', 'widened', 'tightened', 'repaired_below',
    'new_below', 'widened_without_saving', 'widened_with_net_cost')
INTERVALS = ('above', 'below', 'width_cost_s', 'lower_saving_s', 'score_delta_s')


def parents(args):
    registration = read(args.v185_root/'registration.json')
    summary = read(args.v185_root/'summary/status.json')
    model = read(args.v184_root/'model.json')
    require(summary['status'] == 'v185_independent_confirmation_complete' and
        summary['execution_source'] == registration['execution_source'] == '9c36b3b3cf' and
        summary['map_source'] == model['source'] == saved.MAP_SOURCE and
        registration['sealed_model'] == model, 'requires completed frozen V185/V184')
    require(summary['paths'] == 768 and summary['shards'] == 48 and not summary['interval_fit'] and
        summary['map_updates'] == summary['new_training'] == 0 and not summary['main_policy_changed'] and
        all(not summary[k+'_changed'] for k in ('v184', 'v172', 'v151', 'v146')),
        'confirmation cohort incomplete or changed')
    r = registration['reservation']
    require(len(r['actors']) == 12 and r['confirmation_days'] == list(range(3452001, 3452033)) and
        model['base_model_root'] == str(args.v172_root), 'cohort or comparator differs')
    return r, summary, model


def diagnostics(q, p, residual):
    label, error = residual['label'], residual['residual_s']
    require(p['finite'] and p['reference_finite'] and label['status'] == 'resolved' and
        error is not None and all(error[c] is not None for c in CHANNELS), 'finite evidence missing')
    truth, point = label['full_eta_s'], q['point_s']
    close(sum(error[c] for c in CHANNELS), point-truth)
    require(label['available_time_s'] == label['target_ready_time_s'] >= q['query_time_s'], 'label clock differs')
    lo, hi = p['bounds']
    rl, ru = p['reference_bounds']
    require(hi == ru and (p['target'] or lo == rl), 'frozen upper/non-target bounds changed')
    cost = rl-lo
    saving = 10.*(max(rl-truth, 0.)-max(lo-truth, 0.))
    vector = (float(truth > hi), max(truth-hi, 0.), hi-point, point-truth,
        *(error[c] for c in CHANNELS), float(truth < lo), float(truth < rl), cost, saving,
        cost-saving, float(cost > 0.), float(cost < 0.), float(truth < rl and truth >= lo),
        float(truth < lo and truth >= rl), float(cost > 0. and saving == 0.),
        float(cost > 0. and cost-saving > 0.))
    result = np.asarray(vector, dtype=float)
    require(np.isfinite(result).all(), 'nonfinite diagnostic')
    close(result[METRICS.index('score_delta_s')], saved.values(p, label)[saved.METRICS.index('score_delta_s')])
    return result


def weighted_profiles(base, groups):
    neps, nfinite = len(base), sum(v[1] > 0 for v in base.values())
    require(neps > 0 and nfinite > 0, 'empty original cell; do not drop it')
    totals = defaultdict(lambda: np.zeros(3+len(METRICS)))
    for ep, entries in groups.items():
        ne, nf = base[ep]
        for identity, row in entries.items():
            totals[identity][:2] += row[:2]/(ne*neps)
            if nf:
                totals[identity][2] += row[1]/(nf*nfinite)
                totals[identity][3:] += row[2:]/(nf*nfinite)
    global_row = totals['all', 'all']
    for axis in {k[0] for k in totals}:
        require(np.allclose(sum((v for (a, _), v in totals.items() if a == axis),
            np.zeros_like(global_row)), global_row, rtol=1e-9, atol=1e-8), 'partition mass lost')
    close(global_row[0], 1.)
    close(global_row[2], 1.)
    return [dict(axis=a, value=v, values=row.tolist()) for (a, v), row in totals.items()]


def scan(args, actor, day, mode, tables):
    name = f'{actor}_{day}_{mode}'
    original = read(args.v185_root/'cells'/(name+'.json'))
    require(original['execution_source'] == '9c36b3b3cf' and original['map_source'] == saved.MAP_SOURCE and
        original['sampler_seed'] == saved.SAMPLERS[day] and original['strict_completed_prefix_checked'] and
        not original['interval_fit'] and original['map_updates'] == 0 and not original['main_policy_changed'],
        'original path freeze differs')
    base = {s: {} for s in SCOPES}
    groups = {s: defaultdict(dict) for s in SCOPES}
    sources = [saved.stream(args.v185_root/'confirmation'/d/(name+'.jsonl.gz'))
        for d in ('queries', 'interval_predictions', 'residuals')]
    def rows():
        for q, p, r in zip_longest(*sources):
            require(q is not None and p is not None and r is not None and
                q['query_key'] == p['query_key'] == r['query_key'] and
                q['query_time_s'] == p['query_time_s'] == r['query_time_s'] and
                (q['actor'], q['day'], q['mode']) == (actor, day, mode), 'saved streams misaligned')
            # This verifies the lookup from recorded prefix metadata, not a new simulation.
            n, latest, mean = p['prefix_n'], p['prefix_latest_s'], p['prefix_service_mean_s']
            require((n is None or n < 9) == (mean is None), 'recorded prefix support differs')
            require(latest is None or q['query_time_s']-3600. <= latest < q['query_time_s'], 'prefix clock differs')
            history = dict(n=n, latest=latest, mean=None if mean is None else np.array([0., mean, 0.]))
            require(saved.prediction(q, *tables, history) == p, 'sealed lookup differs')
            if q['denominator_eligible']:
                require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load used')
                bits = saved.fit.scope_bits(q)
                scopes = [s for i, s in enumerate(saved.fit.SCOPES) if bits & (1 << i)]
                if bits & 16 and q['native_stage']['kind'] == 'moving-launch':
                    scopes.append('hour13_moving_launch')
                selected = [s for s in SCOPES if s in scopes]
                if selected:
                    entry = None if p['candidate_key'] is None else tables[1].get(tuple(p['candidate_key']))
                    dims = profiles.features(q, entry)
                    dims = {a: dims[a] for a in AXES}
                    dims['service_sign'] = p['sign'] or 'unsupported'
                    vector = diagnostics(q, p, r) if p['finite'] else None
                    ep = tuple(q['episode'])
                    for scope in selected:
                        row = base[scope].setdefault(ep, np.zeros(2))
                        row += (1, int(p['finite']))
                        for identity in dims.items():
                            group = groups[scope][ep].setdefault(identity, np.zeros(2+len(METRICS)))
                            group[:2] += (1, int(p['finite']))
                            if vector is not None:
                                group[2:] += vector
            yield q, p, r['label']
    reproduced, counts = saved.reduce_rows(rows())
    require(counts == original['scoring_counts'], 'saved query/support cohort differs')
    for scope, cell in reproduced.items():
        for key, value in cell.items():
            expected = original['cells'][scope][key]
            require(expected is None, 'missing state changed') if value is None else close(value, expected)
    result = []
    for scope in SCOPES:
        state = weighted_profiles(base[scope], groups[scope])
        global_row = next(p['values'] for p in state if p['axis'] == 'all')
        close(global_row[1], reproduced[scope]['finite_fraction'])
        for m, ref in (('above', 'above'), ('below', 'below'), ('width_cost_s', 'width_delta_s'),
                ('score_delta_s', 'score_delta_s'), ('reference_below', 'reference_below')):
            close(global_row[3+METRICS.index(m)], reproduced[scope][ref])
        result.append(dict(actor=actor, day=day, mode=mode, scope=scope, profiles=state))
    return result


def bucket(args, actor, mode, days):
    tables = saved.tables(args.v184_root, args.v172_root, actor, mode)
    rows = []
    for i, day in enumerate(days, 1):
        rows.extend(scan(args, actor, day, mode, tables))
        print('V186_DAY_READ', actor, mode, i, '/32', flush=True)
    path = args.out_root/'cells'/f'{actor}_{mode}.json'
    write(path, dict(execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], cells=rows))
    return str(path)


def readout(matrix, weights):
    mean, draws = matrix.mean(axis=0), weights@matrix
    supported = bool((draws[:, 2] > 0.).all())
    def metric(i):
        return dict(mean=None if mean[2] == 0. else float(mean[i]/mean[2]),
            ci95=None if not supported else np.percentile(draws[:, i]/draws[:, 2], [2.5, 97.5]).tolist())
    return dict(eligible_mass=float(mean[0]), finite_mass=float(mean[1]), score_mass=float(mean[2]),
        observed_cells=int(np.count_nonzero(matrix[:, 0])), finite_cells=int(np.count_nonzero(matrix[:, 2])),
        means={m: None if mean[2] == 0. else float(mean[3+i]/mean[2]) for i, m in enumerate(METRICS)},
        conditional={m: metric(3+METRICS.index(m)) for m in INTERVALS},
        contributions={m: dict(mean=float(mean[3+METRICS.index(m)]),
            ci95=np.percentile(draws[:, 3+METRICS.index(m)], [2.5, 97.5]).tolist())
            for m in ('above', 'score_delta_s')})


def summarize(rows, actors, days, expected):
    index = {(r['actor'], r['day']): r for r in rows}
    require(len(index) == len(rows) == len(actors)*len(days) and set(index) == set(product(actors, days)),
        'actor/day grid incomplete or duplicated')
    matrices = {}
    for i, key in enumerate(product(actors, days)):
        for p in index[key]['profiles']:
            identity = p['axis'], p['value']
            matrices.setdefault(identity, np.zeros((len(rows), 3+len(METRICS))))[i] = p['values']
    weights = profiles.bootstrap_weights(len(actors), len(days))
    states = []
    for (axis, value), matrix in sorted(matrices.items(), key=lambda x: str(x[0])):
        support = matrix[:, 2].reshape(len(actors), len(days)) > 0.
        states.append(dict(axis=axis, value=value, finite_actors=int(support.any(axis=1).sum()),
            finite_days=int(support.any(axis=0).sum()), **readout(matrix, weights)))
    global_stats = next(p for p in states if p['axis'] == 'all')
    for metric in ('above', 'below', 'score_delta_s'):
        actual, old = global_stats['conditional'][metric], expected['estimates'][metric]
        close(actual['mean'], old['mean'])
        for a, b in zip(actual['ci95'], old['ci95']):
            close(a, b)
    cube = matrices['all', 'all'][:, 3:].reshape(len(actors), len(days), len(METRICS))
    margins = {axis: [dict(**{axis: key}, **dict(zip(METRICS, row.tolist()))) for key, row in zip(keys, cube.mean(axis=dim))]
        for axis, keys, dim in (('day', days, 0), ('actor', actors, 1))}
    daily = cube.mean(axis=0)
    centered = daily[:, METRICS.index('point_bias_s')]-daily[:, METRICS.index('point_bias_s')].mean()
    variance = float(np.mean(centered**2))
    cov = {c: float(np.mean(centered*(daily[:, METRICS.index(c+'_bias_s')]-
        daily[:, METRICS.index(c+'_bias_s')].mean()))) for c in CHANNELS}
    close(sum(cov.values()), variance)
    rate = daily[:, METRICS.index('above')]
    return dict(global_stats=global_stats, profiles=states, **{'by_'+k: v for k, v in margins.items()},
        descriptive_variance={m: upper.variation(cube[:, :, METRICS.index(m)])
            for m in ('above', 'upper_excess_s', 'point_bias_s', 'upper_headroom_s', 'score_delta_s')},
        day_component_covariance=dict(total_bias_variance=variance, covariance=cov,
            signed_fractions=None if variance == 0. else {c: v/variance for c, v in cov.items()},
            upper_rate_correlations={m: profiles.correlation(rate, daily[:, METRICS.index(m)]) for m in
                ('point_bias_s', 'running_bias_s', 'service_bias_s', 'residence_bias_s', 'upper_headroom_s')}),
        day_counts=dict(above_ten_percent=int((rate > .1).sum()),
            positive_score_delta=int((daily[:, METRICS.index('score_delta_s')] > 0.).sum())))


def compact(result):
    state_rows = []
    for p in result['profiles']:
        if p['axis'] in ('service_sign', 'stage', 'horizon', 'remaining_stops', 'clearance_elapsed_s'):
            state_rows.append(dict(axis=p['axis'], value=p['value'], score_mass=p['score_mass'],
                days=p['finite_days'], actors=p['finite_actors'], above=p['conditional']['above'],
                width_cost_s=p['means']['width_cost_s'], lower_saving_s=p['means']['lower_saving_s'],
                score_delta_s=p['conditional']['score_delta_s'],
                score_contribution=p['contributions']['score_delta_s']))
    return {**{k: v for k, v in result.items() if k not in ('profiles', 'by_day', 'by_actor')}, 'states': state_rows}


def run(args):
    require(args.out_root not in (args.v185_root, args.v184_root, args.v172_root), 'output must be separate')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v186_*.py', '-v'], check=True)
    reservation, original, model = parents(args)
    actors, days = sorted(reservation['actors']), sorted(reservation['confirmation_days'])
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(bucket, args, actor, mode, days) for actor, mode in product(actors, ('old', 'handoff'))]
        for i, job in enumerate(as_completed(jobs), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":24,"unit":"readonly_actor_mode"}' % i, flush=True)
    results = []
    for mode, scope in product(('old', 'handoff'), SCOPES):
        rows = []
        for actor in actors:
            b = read(args.out_root/'cells'/f'{actor}_{mode}.json')
            require(b['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'bucket source differs')
            rows.extend(r for r in b['cells'] if r['scope'] == scope)
        expected = next(r for r in original['readouts'] if (r['mode'], r['scope']) == (mode, scope))
        result = dict(mode=mode, scope=scope, **summarize(rows, actors, days, expected))
        write(args.out_root/'details'/f'{mode}_{scope}.json', result)
        results.append(compact(result))
    require(read(args.v184_root/'model.json') == model and read(args.v185_root/'summary/status.json') == original,
        'frozen header/confirmation changed')
    write(args.out_root/'summary/status.json', dict(status='v186_readonly_upper_cost_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], input_source='9c36b3b3cf', map_source=saved.MAP_SOURCE,
        paths=768, days=32, actors=12, endpoints_and_crossed_intervals_reproduced=True,
        results=results, bootstrap_draws=5000, bootstrap_seed=20260803, interval_fit=False, map_updates=0,
        new_replays=0, new_training=0, main_policy_changed=False,
        interpretation='Consumed-cohort descriptive profiles; component covariance is not causal attribution.'))
    print('V186_READONLY_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v185-root', 'v184-root', 'v172-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    run(parser.parse_args())
