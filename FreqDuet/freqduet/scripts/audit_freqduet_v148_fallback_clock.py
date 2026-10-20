#!/usr/bin/env python3
"""Split V148's unallocated source-switch error using saved raw-clock fields."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v147_full_eta as prior


PARTS = ('raw_arrival_clock', 'raw_target_service_proxy', 'raw_gap_adjustment')
read, write, key, close = prior.read, prior.write, prior.key, prior.close


def opaque(row, saved):
    return not (row['online']['component_used'] or saved['v140_status'] == 'passive_stop_component'
        or saved['v141_status'] == 'passive_unissued_ready')


def raw_parts(row, saved, arrival_s, ready_s):
    if not opaque(row, saved):
        return dict.fromkeys(PARTS, 0.)
    e, now = row['evidence'], row['prediction_time_s']
    parts = dict(raw_arrival_clock=e['eta_prediction_s']-(arrival_s-now),
        raw_target_service_proxy=e['service_dwell_proxy_s']-(ready_s-arrival_s),
        raw_gap_adjustment=saved['v141_s']-e['eta_prediction_s']-e['service_dwell_proxy_s'])
    close(sum(parts.values()), saved['v141_s']-(ready_s-now))
    return parts


def record(k, parts, full_delta_error):
    sign = float(np.sign(full_delta_error))
    return dict(episode=k, unallocated_bias_s=sum(parts.values()),
        unallocated_mae_allocation_s=sign*sum(parts.values()),
        **{p+'_bias_s': v for p, v in parts.items()},
        **{p+'_mae_allocation_s': sign*v for p, v in parts.items()})


def load(root, name):
    path = {kind: {key(r): r for r in prior.trace(root/folder/(name+'.jsonl.gz'))}
        for kind, folder in (('trace', 'traces'), ('ready', 'ready_labels'))}
    path['saved'] = {tuple(r['query']): r for r in prior.trace(root/'shadows'/(name+'.jsonl.gz'))}
    with np.load(root/'records'/(name+'.npz')) as z:
        fields = z['event_fields'].tolist()
        path['arrivals'] = {key(r): r for r in (dict(zip(fields, v.tolist())) for v in z['arrivals'])}
    return path


def case(root, parent, out, seed, day):
    paths = {m: load(root, f'{seed}_{day}_{m}') for m in ('old', 'handoff')}
    expected = read(parent/'cells'/f'{seed}_{day}.json')
    rows, counts = [], Counter()
    for k in paths['old']['trace'].keys() & paths['handoff']['trace'].keys():
        b, c = [paths[m]['trace'][k] for m in ('old', 'handoff')]
        sb, sc = [paths[m]['saved'][k] for m in ('old', 'handoff')]
        if prior.source(b, sb) == prior.source(c, sc):
            continue
        labels = [prior.target_label(paths[m]['trace'][k], paths[m]['ready']) for m in paths]
        if any(s != 'resolved' for s, _ in labels) or any(
                saved[v+'_s'] is None for saved in (sb, sc) for v in prior.VERSIONS):
            continue
        pieces = {}
        for mode, row, saved in (('old', b, sb), ('handoff', c, sc)):
            target = (prior.follower(row)[0], *k[1:])
            pieces[mode] = raw_parts(row, saved, paths[mode]['arrivals'][target]['time_s'],
                paths[mode]['ready'][target]['ready_time_s'])
            if opaque(row, saved):
                counts[mode+'/'+row['evidence']['follower_source']] += 1
        delta = {p: pieces['handoff'][p]-pieces['old'][p] for p in PARTS}
        full_error_delta = (sc['v141_s']-labels[1][1])-(sb['v141_s']-labels[0][1])
        rows.append(record(k, delta, full_error_delta))
    previous = expected['paired'].get('switch_conditional')
    if len(rows) != (previous['queries'] if previous else 0):
        raise AssertionError('raw-clock audit differs from V148 source-switch cohort')
    denominator = expected['paired']['switch_burden']['queries']
    fields = set(record([0], dict.fromkeys(PARTS, 0.), 0.))-{'episode'}
    stats = dict(switch_burden=dict(queries=denominator, episodes=denominator,
        metrics={f: sum(r[f] for r in rows)/denominator for f in fields}))
    if rows:
        stats['switch_conditional'] = prior.describe(rows)
    for group in stats:
        for field in ('unallocated_bias_s', 'unallocated_mae_allocation_s'):
            close(stats[group]['metrics'][field], expected['paired'][group]['metrics'][field])
    write(out/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, stats=stats, counts=dict(counts)))


def run(args):
    parent = read(args.v148_root/'summary/status.json')
    if parent['status'] != 'v148_readonly_eta_components_complete':
        raise AssertionError('requires completed V148 clock accounting')
    v147 = read(args.v147_root/'summary/status.json')
    cases = list(itertools.product(v147['actors'], v147['days']))
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v148_fallback_clock.py', '-v'], check=True)
    root = args.v146_root/'confirmation'
    case(root, args.v148_root, args.out_root, *cases[0])
    print('V148_RAW_CLOCK_SAVED_PAIR_PREFLIGHT_COMPLETE', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, root, args.v148_root, args.out_root, s, d) for s, d in cases[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 64 == 0:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=n, total=len(cases), unit='saved_pair')), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in cases]
    write(args.out_root/'summary/status.json', dict(status='v148_readonly_raw_clock_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], paths=2*len(cases),
        results=prior.aggregate(cells, lambda c: c['stats']),
        opaque_endpoint_sources=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        original_unallocated_budget_reproduced=True, point_parameters_changed=False,
        saved_point_files_changed=False, calibration_refit=False, new_rollouts=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Post-hoc read-only split of V148 unallocated error; no fitting or new admission gate. '
            'Raw arrival-clock error includes upstream stops and can reference an already-past arrival; '
            'it is not pure running-time error. Target service proxy uses full arrival-to-ready service. '
            'MAE allocations use the original full delta-error sign, not a new absolute-error metric. '
            'Conditional sparse groups keep no CI; no-event burden zeros are observed event absence.'))
    print('V148_READONLY_RAW_CLOCK_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v146-root', 'v147-root', 'v148-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
