#!/usr/bin/env python3
"""Split consumed actual dispatch intervals into eligibility and readiness."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v125_timing_propagation as timing

frozen = timing.frozen


def launch_rows(launches, directions):
    points = {}
    for row in launches:
        trip, actual, lag = int(row['trip_id']), row['time_s'], row['readiness_s']
        points[trip, 0, directions[trip]] = dict(time_s=actual,
            coordinates=np.asarray([actual-lag, lag]))
    return timing.edge_rows(points), points


def effect(base, candidate):
    delta = (candidate['covariance']-base['covariance'])/base['mean']**2
    normalization = candidate['variance']*(1/candidate['mean']**2-1/base['mean']**2)
    result = dict(cv_squared=candidate['cv_squared']-base['cv_squared'],
        raw_variance_s2=candidate['variance']-base['variance'],
        eligibility_allocated=float(delta[0].sum()), readiness_allocated=float(delta[1].sum()),
        eligibility_diagonal=float(delta[0, 0]), readiness_diagonal=float(delta[1, 1]),
        eligibility_readiness_cross=float(2*delta[0, 1]), mean_normalization=normalization)
    np.testing.assert_allclose(result['eligibility_allocated']+result['readiness_allocated']+normalization,
        result['cv_squared'], atol=1e-10, rtol=0)
    return result


def partition(base, candidate, grouping):
    tables = []
    for rows in (base, candidate):
        table = {}
        for row in rows:
            table.setdefault(grouping(row), []).append(row)
        tables.append(table)
    totals = timing.moment(base), timing.moment(candidate)
    rows = []
    for group in sorted(tables[0].keys() | tables[1].keys()):
        b, c = tables[0].get(group, []), tables[1].get(group, [])
        bm, cm = timing.moment(b), timing.moment(c)
        rows.append(dict(group=group, base_n=len(b), candidate_n=len(c),
            effect=effect(bm, cm) if bm is not None and cm is not None else None,
            pooled_cv2_delta=timing.contribution(c, totals[1])-timing.contribution(b, totals[0])))
    np.testing.assert_allclose(sum(r['pooled_cv2_delta'] for r in rows),
        totals[1]['cv_squared']-totals[0]['cv_squared'], atol=1e-10, rtol=0)
    return rows


def read(root, seed, day, method):
    pair = json.loads((root/'pairs'/f'{seed}_{day}_{method}.json').read_text())
    with np.load(root/'records'/f'{seed}_{day}_{method}_0.npz') as record:
        fields = record['event_fields'].tolist()
        directions = {int(row[fields.index('trip_id')]): bool(row[fields.index('direction')]) for row in record['arrivals']}
        launches = [dict(zip(record['launch_fields'].tolist(), row.tolist())) for row in record['launches']]
    rows, points = launch_rows(launches, directions)
    moments = timing.moment(rows)
    np.testing.assert_allclose([moments['cv_squared'], moments['mean'], moments['variance']],
        [pair['results'][k] for k in ('dispatch_cv_squared', 'dispatch_gap_mean_s',
         'dispatch_gap_variance_s2')], atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.mean([r['readiness_s'] for r in launches]),
        pair['results']['readiness_delay_s'], atol=1e-6, rtol=0)
    return rows, points, moments


def analyze_cell(v123, v124, index):
    seed, day = frozen.ITEMS[index]
    paths = {m: read(v124 if m == frozen.PRIMARY else v123, seed, day, m) for m in frozen.METHODS}
    reference = paths[frozen.prior.BILATERAL][1]
    effects, profiles, first = {}, {}, {}
    for name, (a, b) in frozen.CONTRASTS.items():
        br, bp, bm = paths[b]
        cr, cp, cm = paths[a]
        effects[name] = effect(bm, cm)
        profiles[name] = partition(br, cr, lambda r: (int(r['key'][2]), int(reference[r['key']]['time_s']//14400)))
        first[name] = {}
        for label, coordinate in (('actual', None), ('eligibility', 0), ('readiness', 1)):
            value = lambda p: p['time_s'] if coordinate is None else p['coordinates'][coordinate]
            keys = [k for k in bp if value(bp[k]) != value(cp[k])]
            key = min(keys, key=lambda k: min(bp[k]['time_s'], cp[k]['time_s'])) if keys else None
            first[name][label] = None if key is None else dict(trip_id=key[0], direction=key[2],
                base_actual_s=bp[key]['time_s'], candidate_actual_s=cp[key]['time_s'],
                base_value_s=float(value(bp[key])), candidate_value_s=float(value(cp[key])))
    return dict(seed=seed, day=day, effects=effects, direction_cohorts=profiles, first_changes=first)


def run(v123, v124, v125, out, workers):
    parent = json.loads((v125/'summary/status.json').read_text())
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        cells = list(pool.map(analyze_cell, [v123]*len(frozen.ITEMS), [v124]*len(frozen.ITEMS), range(len(frozen.ITEMS))))
    for cell, old in zip(cells, parent['actor_day']):
        if (cell['seed'], cell['day']) != (old['seed'], old['day']):
            raise AssertionError('consumed actor/day order differs from V125')
        for name, values in cell['effects'].items():
            np.testing.assert_allclose(values['cv_squared'], old['effects'][name]['launch_cv_squared'], atol=1e-10, rtol=0)
    frozen.pilot.write_json(out/'summary/status.json', dict(status='v126_consumed_dispatch_readiness_complete',
        effects=frozen.prior.inference.describe(cells, 'effects'), actor_day=cells,
        axes=frozen.prior.axes.axis_summary(cells, 'effects', list(cells[0]['effects'][timing.FOCUS[0]])),
        consumed_paths=32, original_dispatch_and_readiness_exact=True, v125_dispatch_effects_exact=True,
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Read-only covariance accounting on consumed paths. First eligibility is actual launch '
            'minus recorded waiting for fleet readiness, not the original timetable or a pure upper-policy action. '
            'It already depends on prior actual releases and sticky eligibility. '
            'Readiness differences are not terminal-return timestamps. No causal mediation claim.'))
    print('V126_DISPATCH_READINESS_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v123-root', 'v124-root', 'v125-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v126_*.py', '-v'], check=True)
    run(a.v123_root, a.v124_root, a.v125_root, a.out_root, a.workers)
