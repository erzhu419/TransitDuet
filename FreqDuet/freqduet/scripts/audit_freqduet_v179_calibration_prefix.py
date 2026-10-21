#!/usr/bin/env python3
"""Calibration-only replication and strictly completed-prefix risk diagnostics."""

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product, zip_longest
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v178_lower_width as prior
from decompose_freqduet_v178_state_mix import decomposition


saved = prior.saved
read, write, require, close, stream = prior.read, prior.write, prior.require, prior.close, prior.stream
DAYS = tuple(range(2951001, 2951065))
HISTORIES = ('global', 'same_remaining')
CHANNELS = ('total', 'running', 'service', 'residence')
TARGETS = ('total_residual', 'lower_miss')
WINDOW, MINIMUM = 3600., 9


def parents(args):
    model = read(args.v176_root/'model.json')
    audit = read(args.v166_audit_root/'summary/status.json')
    require(model['execution_source'] == '36108d4f49' and model['calibration_days'] == list(DAYS) and
        model['capture_source'] == '20d914af03' and Path(model['capture_root']) == args.v166_root and
        model['base_model_root'] == str(args.v172_root) and model['frozen_v167_root'] == str(args.v167_root),
        'requires unchanged V176 and original V166 calibration roots')
    require(audit['status'] == 'v166_readback_passed' and audit['passed_paths'] == 1536 and
        audit['failed_paths'] == 0 and audit['calibration_days'] == list(DAYS) and
        audit['query_metadata_complete'] and audit['query_support_and_label_clocks_checked'], 'calibration admission missing')
    require(model['frozen_protocol']['frozen']['native_window_s'] == WINDOW and
        model['frozen_protocol']['frozen']['tails'] == [.1, .1], 'frozen window or tails differ')
    return model


def add_first(atoms, q, r):
    if not q['denominator_eligible'] or q['prediction_coarse'] is None or r['residual_s'] is None:
        return
    require(r['label']['status'] == 'resolved', 'eligible prefix atom lacks a label')
    errors = np.array([r['residual_s'][c] for c in prior.CHANNELS])
    close(errors.sum(), q['point_s']-r['label']['full_eta_s'])
    now, available = q['query_time_s'], r['label']['available_time_s']
    require(available == r['label']['target_ready_time_s'] >= now, 'prefix label clock differs')
    ep = tuple(q['episode'])
    order = now, q['clocks']['query_ready_order']
    value = (order, available, q['observable']['position']['remaining_stops'], errors)
    if ep not in atoms or order < atoms[ep][0]:
        atoms[ep] = value


class Prefix:
    def __init__(self, atoms):
        self.events = sorted(atoms.values(), key=lambda v: (v[1], v[0]))
        self.index = 0
        self.pools = defaultdict(deque)
        self.sums = defaultdict(lambda: np.zeros(3))

    def query(self, now, remaining):
        while self.index < len(self.events) and self.events[self.index][1] < now:
            _, available, stops, errors = self.events[self.index]
            for k in ('global', stops):
                self.pools[k].append((available, errors))
                self.sums[k] += errors
            self.index += 1
        result = {}
        for name, key in (('global', 'global'), ('same_remaining', remaining)):
            pool = self.pools[key]
            while pool and pool[0][0] < now-WINDOW:
                self.sums[key] -= pool.popleft()[1]
            n = len(pool)
            require(not pool or now-WINDOW <= pool[0][0] <= pool[-1][0] < now, 'prefix includes unfinished/out-of-window labels')
            result[name] = dict(n=n, latest=None if not pool else pool[-1][0],
                mean=None if n < MINIMUM else self.sums[key].copy()/n)
        return result


def moment(x, errors, below):
    x = np.r_[sum(x), x]
    y = np.array([sum(errors), below])
    return np.r_[1., x, x*x, y, y*y, np.outer(x, y).ravel()]


def atom_archive(args, chunk):
    with np.load(args.v172_root/chunk['parent_atom_file']) as z:
        source = {k: z[k] for k in ('requests', 'episode_keys', 'bounds', 'frozen_v167_bounds')}
    with np.load(args.v176_root/chunk['audit_file']) as z:
        source.update(candidate=z['bounds'], finite=z['finite_candidate'])
    require(set(source['requests'][:, 0].astype(int)) == set(DAYS), 'atom cohort differs')
    return source


def scan(args, actor, day, mode, tables, model, archive):
    require(day in DAYS, 'only original calibration days allowed')
    root, name = args.v166_root/'calibration', f'{actor}_{day}_{mode}'
    receipt = read(root/'capture_cells'/(name+'.json'))
    require(receipt['execution_source'] == '20d914af03' and (receipt['seed'], receipt['day'], receipt['mode']) ==
        (actor, day, mode) and not receipt['interval_fit'] and not receipt['main_policy_changed'], 'capture differs')
    requests = archive['requests']
    indices = np.flatnonzero(requests[:, 0] == day)
    lookup = {tuple(requests[i, 5:].astype(int)): i for i in indices}
    require(len(lookup) == len(indices), 'duplicate saved query identity')
    atoms, selected, count = {}, [], 0
    sources = [stream(root/d/(name+'.jsonl.gz')) for d in ('queries', 'residuals')]
    for q, r in zip_longest(*sources):
        require(q is not None and r is not None and q['query_key'] == r['query_key'] and
            q['query_time_s'] == r['query_time_s'] and q['episode'] == r['episode'] and
            (q['actor'], q['day'], q['mode']) == (actor, day, mode), 'calibration stream join differs')
        count += 1
        add_first(atoms, q, r)
        if not q['denominator_eligible']:
            continue
        i = lookup.pop(tuple(q['query_key']))
        require(requests[i, 3] == q['point_s'] and requests[i, 4] == q['query_time_s'] and
            archive['episode_keys'][int(requests[i, 1])].tolist() == [day, *q['episode']], 'frozen atom coordinates differ')
        scopes = prior.prior.scopes(q)
        if not scopes:
            continue
        p = saved.prediction(q, *tables, model['frozen_protocol'])
        require(p['finite'] == bool(archive['finite'][i]), 'frozen finite mask differs')
        entry = None if p['candidate_key'] is None else tables[0].get(tuple(p['candidate_key']))
        if p['finite']:
            require(p['bounds'] == archive['candidate'][i].tolist() and
                p['temporal_bounds'] == archive['bounds'][i, 0].tolist() and
                p['reference_bounds'] == archive['bounds'][i, 1].tolist() and
                p['secondary_bounds'] == archive['frozen_v167_bounds'][i].tolist(), 'sealed calibration bounds differ')
        require(q['clocks']['query_load'] is None and not q['clocks']['query_load_available'], 'late load was backfilled')
        selected.append(dict(time=q['query_time_s'], order=q['clocks']['query_ready_order'],
            episode=tuple(q['episode']), remaining=None if q['observable']['position'] is None else q['observable']['position']['remaining_stops'],
            scopes=scopes, finite=p['finite'], features=prior.features(q, entry),
            scored=saved.values(p, r['label']),
            diagnostic=prior.diagnostics(q, p, r, entry) if p['finite'] else None,
            errors=None if r['residual_s'] is None else np.array([r['residual_s'][c] for c in prior.CHANNELS])))
    require(not lookup and count == receipt['query_record_count'], 'calibration query cohort incomplete')
    prefix = Prefix(atoms)
    base = {s: {} for s in prior.SCOPES}
    groups = {s: defaultdict(dict) for s in prior.SCOPES}
    signals = {s: defaultdict(dict) for s in prior.SCOPES}
    for q in sorted(selected, key=lambda v: (v['time'], v['order'])):
        histories = prefix.query(q['time'], q['remaining'])
        dimensions = dict(q['features'])
        for h, value in histories.items():
            mean = value['mean']
            dimensions['prefix_'+h+'_support'] = 'none' if not value['n'] else '1-8' if mean is None else '>=9'
            dimensions['prefix_'+h+'_sign'] = 'unavailable' if mean is None else 'negative' if mean.sum() < 0 else 'positive' if mean.sum() > 0 else 'zero'
        for scope in q['scopes']:
            ep = q['episode']
            row = base[scope].setdefault(ep, np.zeros(3+len(saved.METRICS)))
            row[:3] += (1, int(q['finite']), int(q['scored'] is not None))
            require(not q['finite'] or q['scored'] is not None, 'finite calibration label/reference missing')
            if q['scored'] is not None:
                row[3:] += q['scored']
            for axis, value in dimensions.items():
                group = groups[scope][ep].setdefault((axis, value), np.zeros(2+len(prior.METRICS)))
                group[:2] += (1, int(q['finite']))
                if q['finite']:
                    group[2:] += q['diagnostic']
            if q['finite']:
                for h, value in histories.items():
                    if value['mean'] is not None:
                        key = h, q['remaining']
                        target = signals[scope][ep].setdefault(key, np.zeros(21))
                        target += moment(value['mean'], q['errors'], q['diagnostic'][prior.METRICS.index('below')])
    rows = []
    for scope in prior.SCOPES:
        endpoint = saved.collapse(base[scope], 0, 0)
        profiles = prior.weighted_profiles(base[scope], groups[scope])
        sums = defaultdict(lambda: np.zeros(21))
        nf = sum(v[1] > 0 for v in base[scope].values())
        for ep, entries in signals[scope].items():
            for key, values in entries.items():
                sums[key] += values/(base[scope][ep][1]*nf)
        rows.append(dict(actor=actor, day=day, mode=mode, scope=scope, endpoint=endpoint,
            endpoint_reproduced=True, profiles=profiles,
            signals=[dict(history=h, remaining=k, moments=v.tolist()) for (h, k), v in sums.items()]))
    return rows


def bucket(args, actor, mode):
    model = parents(args)
    chunk = next(c for c in model['chunks'] if (c['actor'], c['mode']) == (actor, mode))
    archive = atom_archive(args, chunk)
    tables = saved.map_tables(args.v176_root, args.v172_root, args.v167_root, actor, mode)
    rows = []
    for i, day in enumerate(DAYS, 1):
        rows.extend(scan(args, actor, day, mode, tables, model, archive))
        print('V179_CALIBRATION_DAY_READ', actor, mode, i, '/64', flush=True)
    write(args.out_root/'cells'/f'{actor}_{mode}.json', dict(execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], cells=rows))


def within_correlations(values):
    mass = values[..., :1]
    x, xx, y, yy = values[..., 1:5], values[..., 5:9], values[..., 9:11], values[..., 11:13]
    xy = values[..., 13:].reshape(*values.shape[:-1], 4, 2)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0.)
    vx = (xx-divide(x*x, mass)).sum(axis=-2)
    vy = (yy-divide(y*y, mass)).sum(axis=-2)
    cov = (xy-divide(x[..., :, None]*y[..., None, :], mass[..., None])).sum(axis=-3)
    denominator = np.sqrt(np.maximum(vx[..., :, None], 0.)*np.maximum(vy[..., None, :], 0.))
    return np.divide(cov, denominator, out=np.full_like(cov, np.nan), where=denominator > 0.)


def signal_summary(rows, actors, days, history):
    index = {(r['actor'], r['day']): r for r in rows}
    require(len(index) == len(rows) == len(actors)*len(days), 'signal grid incomplete')
    remaining = sorted({s['remaining'] for r in rows for s in r['signals'] if s['history'] == history})
    if not remaining:
        return dict(history=history, support_mass=0., correlations=None)
    matrix = np.zeros((len(rows), len(remaining), 21))
    for i, key in enumerate(product(actors, days)):
        for s in index[key]['signals']:
            if s['history'] == history:
                matrix[i, remaining.index(s['remaining'])] = s['moments']
    weights = prior.bootstrap_weights(len(actors), len(days))
    draws = (weights@matrix.reshape(len(rows), -1)).reshape(5000, len(remaining), 21)
    point = within_correlations(matrix.mean(axis=0))
    boot = within_correlations(draws)
    support = matrix[:, :, 0].sum(axis=1)
    output = []
    for i, c in enumerate(CHANNELS):
        for j, target in enumerate(TARGETS):
            output.append(dict(signal=c, target=target, correlation=None if not np.isfinite(point[i, j]) else float(point[i, j]),
                ci95=None if not np.isfinite(boot[:, i, j]).all() else np.percentile(boot[:, i, j], [2.5, 97.5]).tolist()))
    return dict(history=history, support_mass=float(support.mean()),
        support_ci95=np.percentile(draws[:, :, 0].sum(axis=1), [2.5, 97.5]).tolist(),
        cells_with_support=int(np.count_nonzero(support)), correlations=output)


def state_mix(rows, actors, days):
    index = {(r['actor'], r['day']): r for r in rows}
    output = {}
    for axis in ('remaining_stops', 'stage', 'key_days'):
        categories = sorted({str(p['value']) for r in rows for p in r['profiles'] if p['axis'] == axis})
        mass, lower = np.zeros((len(days), len(categories))), np.zeros((len(days), len(categories)))
        for i, day in enumerate(days):
            for actor in actors:
                for p in index[actor, day]['profiles']:
                    if p['axis'] == axis:
                        k = categories.index(str(p['value']))
                        mass[i, k] += p['values'][2]/len(actors)
                        lower[i, k] += p['values'][3+prior.METRICS.index('below')]/len(actors)
        output[axis] = decomposition(mass, lower)
    return output


def aggregate(args):
    model = parents(args)
    actors, days = sorted(model['actors']), list(DAYS)
    rows = []
    for actor, mode in product(actors, ('old', 'handoff')):
        value = read(args.out_root/'cells'/f'{actor}_{mode}.json')
        require(value['execution_source'] == os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], 'analysis source differs')
        rows.extend(value['cells'])
    results, signals = [], []
    for mode, scope in product(('old', 'handoff'), prior.SCOPES):
        selected = [r for r in rows if (r['mode'], r['scope']) == (mode, scope)]
        endpoint = saved.crossed([dict(actor=r['actor'], day=r['day'], **r['endpoint']) for r in selected], actors, days)
        replicated = prior.summarize(selected, actors, days, endpoint)
        results.append(dict(mode=mode, scope=scope, **{k: v for k, v in replicated.items() if k != 'by_day'},
            state_mix=state_mix(selected, actors, days)))
        for split, cohort in (('all64', days), ('first32', days[:32]), ('last32', days[32:])):
            subset = [r for r in selected if r['day'] in cohort]
            for h in HISTORIES:
                signals.append(dict(mode=mode, scope=scope, split=split,
                    **signal_summary(subset, actors, cohort, h)))
    require(read(args.v176_root/'model.json') == model, 'sealed header changed')
    write(args.out_root/'summary/status.json', dict(status='v179_calibration_prefix_readonly_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], paths=1536, calibration_days=days,
        prefix_window_s=WINDOW, minimum_distinct_completed_episodes=MINIMUM,
        results=results, signals=signals, interval_fit=False, map_updates=0, new_replays=0,
        new_training=0, confirmation_outcomes_read=False, confirmation_started=False, main_policy_changed=False,
        interpretation='Calibration-only descriptive replication; frozen bands are in-sample. No predictor, gate or new interval fitted.'))
    print('V179_CALIBRATION_PREFIX_COMPLETE', flush=True)


def prepare(args):
    model = parents(args)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v179_*.py', '-v'], check=True)
    actor, day = model['actors'][0], DAYS[0]
    chunk = next(c for c in model['chunks'] if (c['actor'], c['mode']) == (actor, 'old'))
    tables = saved.map_tables(args.v176_root, args.v172_root, args.v167_root, actor, 'old')
    cells = scan(args, actor, day, 'old', tables, model, atom_archive(args, chunk))
    write(args.out_root/'prepare/summary/status.json', dict(status='v179_readonly_preflight_passed',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], calibration_day=day,
        frozen_bounds_and_masks_reproduced=True, target_cells=len(cells), new_replays=0, new_training=0))
    print('V179_READONLY_PREFLIGHT_PASSED', flush=True)


def run(args):
    model = parents(args)
    cases = list(product(model['actors'], ('old', 'handoff')))
    require(0 <= args.start < args.end <= 24, 'invalid actor/mode shard')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(bucket, args, a, m) for a, m in cases[args.start:args.end]]
        for job in as_completed(jobs):
            job.result()
    write(args.out_root/'receipts'/f'{args.start}-{args.end}'/'status.json',
        dict(status='v179_readonly_shard_complete', execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],
            start=args.start, end=args.end))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'aggregate'))
    for name in ('out-root', 'v166-root', 'v166-audit-root', 'v176-root', 'v172-root', 'v167-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'aggregate': aggregate}[args.command](args)
