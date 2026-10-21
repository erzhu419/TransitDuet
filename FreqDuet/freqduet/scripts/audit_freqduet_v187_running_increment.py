#!/usr/bin/env python3
"""Original64 calibration only: completed running-prefix information within states."""

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
from scipy.sparse import csr_matrix

import audit_freqduet_v179_calibration_prefix as previous
import audit_freqduet_v180_service_increment as states


read, write, require, close = previous.read, previous.write, previous.require, previous.close
temporal = previous.saved.saved.plan
DAYS = previous.DAYS
SCOPES = ('hour2', 'clearance', 'hour13', 'hour13_moving_launch')
PARTITIONS = dict(distance=('remaining_stops',), distance_stage=('remaining_stops', 'stage'),
    geometry_time=states.STATE)
TARGETS = ('underestimate_s', 'running_underestimate_s', 'above', 'upper_excess_s', 'below', 'score_s')
SIZE, CSIZE = 6+4*len(TARGETS), 3+3*len(TARGETS)


def parents(args):
    model = read(args.v172_root/'model.json')
    audit = read(args.v166_audit_root/'summary/status.json')
    require(model['status'] == 'sealed' and model['execution_source'] == 'a5b45df520' and
        model['calibration_days'] == list(DAYS) and model['capture_source'] == '20d914af03' and
        Path(model['capture_root']) == args.v166_root and len(model['actors']) == 12 and
        model['modes'] == ['old', 'handoff'], 'requires frozen V172 and original64 V166 captures')
    require(audit['status'] == 'v166_readback_passed' and audit['passed_paths'] == 1536 and
        audit['failed_paths'] == 0 and audit['query_metadata_complete'] and
        audit['query_support_and_label_clocks_checked'] and audit['calibration_days'] == list(DAYS),
        'original capture admission differs')
    frozen = model['frozen_protocol']['frozen']
    require(frozen['native_window_s'] == previous.WINDOW == 3600. and frozen['tails'] == [.1, .1],
        'window or tails differ')
    return model


def moment(x, y):
    r, s = x
    y = np.asarray(y)
    return np.r_[1., r, s, r*r, s*s, r*s, y, y*y, r*y, s*y]


def centered(v):
    mass, r, s = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    y, yy, ry, sy = np.split(v[..., 6:], 4, axis=-1)
    def divide(a):
        return np.divide(a, mass, out=np.zeros_like(a), where=mass > 0.)
    return np.concatenate((v[..., 3:4]-divide(r*r), v[..., 4:5]-divide(s*s),
        v[..., 5:6]-divide(r*s), yy-divide(y*y), ry-divide(r*y), sy-divide(s*y)), axis=-1)


def correlations(c):
    rr, ss, rs = c[..., :1], c[..., 1:2], c[..., 2:3]
    yy, ry, sy = np.split(c[..., 3:], 3, axis=-1)
    def ratio(a, b):
        return np.divide(a, b, out=np.full_like(a+b, np.nan), where=b > 1e-12)
    raw = ratio(ry, np.sqrt(np.maximum(rr, 0.)*np.maximum(yy, 0.)))
    service = ratio(sy, np.sqrt(np.maximum(ss, 0.)*np.maximum(yy, 0.)))
    adjusted = ry-ratio(rs*sy, ss)
    den = np.sqrt(np.maximum(rr-ratio(rs*rs, ss), 0.)*np.maximum(yy-ratio(sy*sy, ss), 0.))
    return np.stack((raw, service, ratio(adjusted, den)), axis=-1)


def outcomes(point, truth, running_error, bounds):
    lo, hi = bounds
    excess = max(truth-hi, 0.)
    return np.array([truth-point, -running_error, float(truth > hi), excess, float(truth < lo),
        hi-lo+10.*(max(lo-truth, 0.)+excess)])


def scan(args, actor, day, mode, table, archive):
    require(day in DAYS, 'not an original calibration day')
    root, name = args.v166_root/'calibration', f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require((receipt['execution_source'], receipt['seed'], receipt['day'], receipt['mode']) ==
        ('20d914af03', actor, day, mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'],
        'capture source or controller differs')
    requests, finite = archive['requests'], archive['finite_candidate']
    indices = np.flatnonzero(requests[:, 0] == day)
    lookup = {tuple(requests[i, 5:].astype(int)): i for i in indices}
    require(len(lookup) == len(indices), 'duplicate saved query identity')
    atoms, selected, count, moving = {}, [], 0, np.zeros(len(requests), dtype=bool)
    truth = {}
    entries = {tuple(e['key']): e for e in table['candidate']}
    for q, r in zip_longest(*(previous.stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals'))):
        require(q is not None and r is not None and q['query_key'] == r['query_key'] and
            q['episode'] == r['episode'] and q['query_time_s'] == r['query_time_s'] and
            (q['actor'], q['day'], q['mode']) == (actor, day, mode), 'query/residual join differs')
        count += 1
        previous.add_first(atoms, q, r)
        if not q['denominator_eligible']:
            continue
        i = lookup.pop(tuple(q['query_key']))
        require(requests[i, 3] == q['point_s'] and requests[i, 4] == q['query_time_s'] and
            archive['episode_keys'][int(requests[i, 1])].tolist() == [day, *q['episode']] and
            archive['native_supported'][i] == q['native_supported'], 'saved coordinates differ')
        bits = previous.saved.fit.scope_bits(q)
        require(bits == int(requests[i, 2]) and q['observable']['hour'] == int(q['query_time_s']//3600),
            'scope clock differs')
        moving[i] = bool(bits & 16) and q['native_stage']['kind'] == 'moving-launch'
        scopes = [s for s, yes in zip(SCOPES, (bits & 8, bits & 4, bits & 16, moving[i])) if yes]
        if not scopes:
            continue
        period = temporal.STRATA[int(temporal.strata(np.array([q['query_time_s']]))[0])]
        key = (*q['candidate_key'], period) if q['native_supported'] else None
        entry = entries.get(key)
        bounds = previous.saved.fit.interval(q['point_s'], entry, q['native_supported'])
        require(bool(finite[i]) == (bounds is not None), 'frozen V172 finite mask differs')
        require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load backfill')
        y = None
        if bounds is not None:
            require(list(bounds) == archive['bounds'][i, 0].tolist() and r['label']['status'] == 'resolved' and
                r['residual_s'] is not None and r['label']['available_time_s'] == r['label']['target_ready_time_s']
                >= q['query_time_s'], 'saved bounds or label clock differs')
            errors = np.array([r['residual_s'][c] for c in previous.prior.CHANNELS])
            require(np.isfinite(errors).all() and np.array_equal(errors, archive['errors'][i]), 'joint residual changed')
            truth[i] = r['label']['full_eta_s']
            close(errors.sum(), q['point_s']-truth[i])
            y = outcomes(q['point_s'], truth[i], errors[0], bounds)
        f = previous.prior.features(q, entry)
        selected.append((q['query_time_s'], q['clocks']['query_ready_order'], tuple(q['episode']),
            scopes, f, y))
    require(not lookup and count == receipt['query_record_count'], 'calibration path incomplete')
    prefix = previous.Prefix(atoms)
    base = {s: {} for s in SCOPES}
    groups = {s: defaultdict(dict) for s in SCOPES}
    for now, order, ep, scopes, f, y in sorted(selected, key=lambda v: v[:2]):
        h = prefix.query(now, None if f['remaining_stops'] is None else int(f['remaining_stops']))['global']
        # Positive signals mean prior underprediction; no unfinished label enters x.
        x = None if h['mean'] is None else -h['mean'][:2]
        for scope in scopes:
            row = base[scope].setdefault(ep, np.zeros(2+len(TARGETS)))
            row[:2] += (1., float(y is not None))
            if y is not None:
                row[2:] += y
            if y is None or x is None:
                continue
            for level, fields in PARTITIONS.items():
                key = level, tuple(f[k] for k in fields)
                groups[scope][ep].setdefault(key, np.zeros(SIZE))[:] += moment(x, y)
    rows = []
    days, episodes = requests[:, 0].astype(int), requests[:, 1].astype(int)
    bits = requests[:, 2].astype(int)
    masks = dict(hour2=(bits & 8) != 0, clearance=(bits & 4) != 0, hour13=(bits & 16) != 0,
        hour13_moving_launch=moving)
    for scope in SCOPES:
        supported = temporal.support_cell(day, scope, masks[scope], days, episodes,
            archive['native_supported'], finite, archive['finite_frozen_v167'])
        old = next(c for c in table['cells'] if (c['day'], c['scope']) == (day, scope))
        require(dict(actor=actor, mode=mode, **supported) == old, 'original V172 support endpoint differs')
        neps, nf = len(base[scope]), sum(v[1] > 0 for v in base[scope].values())
        require(neps == old['eligible_episodes'] and nf == old['finite_episodes'] and nf > 0, 'episode cohort differs')
        close(np.mean([v[1]/v[0] for v in base[scope].values()]), old['finite_fraction'])
        endpoint = np.mean([v[2:]/v[1] for v in base[scope].values() if v[1] > 0], axis=0)
        use = np.flatnonzero((days == day) & masks[scope] & finite)
        unique, inverse, counts = np.unique(episodes[use], return_inverse=True, return_counts=True)
        weights = 1./(len(unique)*counts[inverse])
        direct = weights@np.array([outcomes(requests[i, 3], truth[i], archive['errors'][i, 0],
            archive['bounds'][i, 0]) for i in use])
        require(np.allclose(endpoint, direct, rtol=0., atol=1e-8), 'original episode/query weights differ')
        totals = defaultdict(lambda: np.zeros(SIZE))
        for ep, g in groups[scope].items():
            for key, value in g.items():
                totals[key] += value/(base[scope][ep][1]*nf)
        rows.append(dict(actor=actor, day=day, mode=mode, scope=scope, endpoint=endpoint.tolist(),
            finite_fraction=old['finite_fraction'],
            groups=[dict(level=l, key=list(k), moments=v.tolist()) for (l, k), v in totals.items()]))
    return rows


def bucket(args, model, chunk):
    actor, mode = chunk['actor'], chunk['mode']
    table = read(args.v172_root/chunk['map_file'])
    require((table['execution_source'], table['actor'], table['mode']) == ('a5b45df520', actor, mode), 'map source differs')
    with np.load(args.v172_root/chunk['atom_file']) as z:
        archive = {k: z[k] for k in ('requests', 'episode_keys', 'native_supported', 'finite_candidate',
            'finite_frozen_v167', 'bounds')}
        archive['errors'] = np.full((len(archive['requests']), 3), np.nan)
        archive['errors'][z['atom_request']] = z['residual_rsd']
    require(set(archive['requests'][:, 0].astype(int)) == set(DAYS), 'not the original64 cohort')
    rows = []
    for i, day in enumerate(DAYS, 1):
        rows.extend(scan(args, actor, day, mode, table, archive))
        print('V187_CALIBRATION_DAY_READ', actor, mode, i, '/64', flush=True)
    require(read(args.v172_root/chunk['map_file']) == table, 'frozen map changed')
    path = args.out_root/'cells'/f'{actor}_{mode}.json'
    write(path, dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], rows=rows))
    return str(path)


def estimate(point, draws):
    return dict(mean=None if not np.isfinite(point) else float(point),
        ci95=None if not np.isfinite(draws).all() else np.percentile(draws, [2.5, 97.5]).tolist())


def summarize(rows, actors, days, level):
    index = {(r['actor'], r['day']): r for r in rows}
    require(len(index) == len(rows) and set(index) == set(product(actors, days)), 'crossed actor/day grid incomplete')
    keys = sorted({tuple(g['key']) for r in rows for g in r['groups'] if g['level'] == level}, key=str)
    if not keys:
        return dict(level=level, support_mass=0., estimates=None)
    positions = {k: i for i, k in enumerate(keys)}
    matrix = np.zeros((len(rows), len(keys), SIZE))
    for i, key in enumerate(product(actors, days)):
        for g in index[key]['groups']:
            if g['level'] == level:
                matrix[i, positions[tuple(g['key'])]] = g['moments']
    mean = matrix.mean(axis=0)
    c = centered(mean).sum(axis=0)
    local = centered(matrix).sum(axis=1)
    point = correlations(c)
    local_point = correlations(local.mean(axis=0))
    between_point = correlations(c-local.mean(axis=0))
    weights = previous.prior.bootstrap_weights(len(actors), len(days))
    flat = csr_matrix(matrix.reshape(len(rows), -1))
    boot, local_boot, between_boot, masses = [], [], [], []
    for start in range(0, len(weights), 64):
        w = weights[start:start+64]
        draws = (flat.T@w.T).T.reshape(-1, len(keys), SIZE)
        cov = centered(draws).sum(axis=1)
        local_cov = w@local
        boot.append(correlations(cov))
        local_boot.append(correlations(local_cov))
        between_boot.append(correlations(cov-local_cov))
        masses.extend(draws[:, :, 0].sum(axis=1))
    boot, local_boot, between_boot = map(np.concatenate, (boot, local_boot, between_boot))
    supported = matrix[:, :, 0].reshape(len(actors), len(days), len(keys)) > 0.
    support_days = supported.any(axis=0).sum(axis=0)
    result = {t: {name: estimate(point[i, j], boot[:, i, j]) for j, name in enumerate
        ('running_correlation', 'service_correlation', 'running_partial_service')} for i, t in enumerate(TARGETS)}
    for i, target in enumerate(TARGETS):
        for name, p, b in (('within_actor_day_state', local_point, local_boot),
                ('between_actor_day_state_means', between_point, between_boot)):
            result[target][name] = {metric: estimate(p[i, j], b[:, i, j]) for j, metric in
                ((0, 'running_correlation'), (2, 'running_partial_service'))}
    return dict(level=level, fields=list(PARTITIONS[level]), strata=len(keys),
        support_mass=estimate(mean[:, 0].sum(), np.asarray(masses)),
        support_mass_by_distinct_days={label: float(mean[mask, 0].sum()) for label, mask in
            (('1-18', support_days < 19), ('19-31', (support_days >= 19)&(support_days < 32)),
             ('32-47', (support_days >= 32)&(support_days < 48)), ('48-64', support_days >= 48))},
        cells_with_support=int(np.count_nonzero(matrix[:, :, 0].sum(axis=1))),
        retained_running_variance=None if centered(mean.sum(axis=0))[0] <= 0. else
            float(c[0]/centered(mean.sum(axis=0))[0]), estimates=result)


def reduce_scope(args, model, mode, scope):
    actors, rows = sorted(model['actors']), []
    for actor in actors:
        value = read(args.out_root/'cells'/f'{actor}_{mode}.json')
        require(value['source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'bucket source differs')
        rows.extend(r for r in value['rows'] if r['scope'] == scope)
    results = []
    for split, days in (('all64', DAYS), ('first32', DAYS[:32]), ('last32', DAYS[32:])):
        selected = [r for r in rows if r['day'] in days]
        for level in PARTITIONS:
            results.append(dict(mode=mode, scope=scope, split=split,
                **summarize(selected, actors, days, level)))
        print('V187_CROSSED_SUMMARY', mode, scope, split, flush=True)
    write(args.out_root/'details'/f'{mode}_{scope}.json', results)
    compact = []
    for r in results:
        targets = TARGETS if r['split'] == 'all64' else ('above', 'upper_excess_s', 'score_s')
        compact.append({**{k: v for k, v in r.items() if k != 'estimates'},
            'estimates': None if r['estimates'] is None else {t: {k: v for k, v in r['estimates'][t].items()
                if k != 'between_actor_day_state_means'} for t in targets}})
    return compact


def run(args):
    require(args.out_root not in (args.v166_root, args.v166_audit_root, args.v172_root), 'output must be separate')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v187_*.py', '-v'], check=True)
    model = parents(args)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for i, job in enumerate(as_completed([pool.submit(bucket, args, model, chunk) for chunk in model['chunks']]), 1):
            job.result()
            print('SCHEDULEURM_PROGRESS {"current":%d,"total":24,"unit":"readonly_actor_mode"}' % i, flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
        for job in as_completed([pool.submit(reduce_scope, args, model, m, s) for m, s in
                product(('old', 'handoff'), SCOPES)]):
            results.extend(job.result())
    require(parents(args) == model, 'frozen parent changed')
    write(args.out_root/'summary/status.json', dict(status='v187_calibration_running_increment_complete',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], paths=1536, calibration_days=list(DAYS), actors=model['actors'],
        v172_bounds_support_and_episode_weights_reproduced=True, window_s=3600., minimum_completed_episodes=9,
        bootstrap_draws=5000, bootstrap_seed=20260803, results=sorted(results, key=lambda r:(r['mode'],r['scope'],r['split'],r['level'])),
        interval_fit=False, predictor_fit=False, map_updates=0, confirmation_outcomes_read=False,
        confirmation_started=False, new_replays=0, new_training=0, main_policy_changed=False,
        interpretation='Calibration associations only; partial correlations are not a fitted predictor or independent confirmation.'))
    print('V187_CALIBRATION_RUNNING_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'v166-root', 'v166-audit-root', 'v172-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    run(p.parse_args())
