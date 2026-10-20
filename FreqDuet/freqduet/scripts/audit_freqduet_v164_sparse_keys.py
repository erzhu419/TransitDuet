#!/usr/bin/env python3
"""Read-only hour13 support decomposition of sealed V163 maps and indices."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import seal_freqduet_v163_joint_intervals as original


read, write, require, close = original.read, original.write, original.require, original.close


def support_weights(days, episodes, actors, calibration_days):
    contributing, denominator = original.hierarchy(days, episodes)
    require(set(contributing.tolist()) == set(calibration_days), 'hour13 actor/day grid incomplete')
    weights = 1./(actors*len(calibration_days)*denominator)
    close(float(weights.sum()), 1./actors)
    return weights


def partition(weights, finite, native):
    require(not np.any(finite & ~native), 'finite estimate without native support')
    return dict(finite_mass=float(weights[finite].sum()),
        native_abstention_mass=float(weights[~native].sum()),
        key_abstention_mass=float(weights[native & ~finite].sum()))


def bucket(args, model, chunk):
    actor, mode = chunk['actor'], chunk['mode']
    saved = read(args.v163_root/chunk['map_file'])
    require(saved['execution_source'] == '35ae15b185' and (saved['actor'], saved['mode']) == (actor, mode),
        'sealed map identity changed')
    with np.load(args.v163_root/chunk['atom_file']) as archive:
        requests = archive['requests']
        selected = np.flatnonzero((requests[:, 2].astype(np.int64) & 16) != 0)
        key_index = np.full(len(requests), -1, dtype=np.int64)
        key_index[archive['atom_request']] = archive['candidate_key']
        native = archive['native_supported'][selected]
        finite = archive['finite_candidate'][selected]
        keys = key_index[selected]
        require(np.array_equal(native, keys >= 0), 'sealed key/request join differs')
        query = requests[selected]
    days, episodes = query[:, 0].astype(np.int64), query[:, 1].astype(np.int64)
    cal_days, actor_n = model['calibration_days'], len(model['actors'])
    weights = support_weights(days, episodes, actor_n, cal_days)
    masses = partition(weights, finite, native)
    row = dict(actor=actor, mode=mode, eligible_queries=len(query), finite_queries=int(finite.sum()),
        native_abstentions=int((~native).sum()), key_abstentions=int((native & ~finite).sum()), **masses)
    fields = model['frozen_protocol']['conditioning']['candidate_key']
    key_rows, day_rows = [], []
    for index in np.unique(keys[native]):
        entry = saved['candidate'][int(index)]
        mask = keys == index
        require(entry['day_count'] == len(set(entry['days'])), 'stored day count differs')
        supported = entry['status'] == 'finite'
        require(supported == (entry['day_count'] >= 19) and np.all(finite[mask] == supported),
            'sealed abstention rule changed')
        key_rows.append(dict(actor=actor, mode=mode, key_id=int(index),
            values=dict(zip(fields, entry['key'])), day_count=entry['day_count'],
            day_shortfall=max(0, 19-entry['day_count']), hour13_days=len(np.unique(days[mask])),
            queries=int(mask.sum()), mass=float(weights[mask].sum()), finite=supported))
    for day in cal_days:
        mask = days == day
        m = partition(weights[mask], finite[mask], native[mask])
        day_rows.append(dict(actor=actor, day=day, mode=mode,
            finite_fraction=m['finite_mass']*actor_n*len(cal_days),
            native_abstention_fraction=m['native_abstention_mass']*actor_n*len(cal_days),
            key_abstention_fraction=m['key_abstention_mass']*actor_n*len(cal_days)))
    return dict(counts=row, keys=key_rows, days=day_rows)


def group_keys(rows, names, partition_name):
    groups = defaultdict(list)
    for row in rows:
        values = tuple(row['day_count'] if f == 'day_count' else row['values'][f] for f in names)
        groups[(row['mode'], values)].append(row)
    output = []
    for (mode, values), selected in sorted(groups.items()):
        low = [r for r in selected if not r['finite']]
        good = [r for r in selected if r['finite']]
        output.append(dict(mode=mode, partition=partition_name, value=json.dumps(values, separators=(',', ':')),
            keys=len(selected), low_support_keys=len(low), actors=len({r['actor'] for r in selected}),
            finite_queries=sum(r['queries'] for r in good), key_abstentions=sum(r['queries'] for r in low),
            finite_mass=sum(r['mass'] for r in good), key_abstention_mass=sum(r['mass'] for r in low),
            minimum_low_key_days=min((r['day_count'] for r in low), default=None),
            maximum_low_key_days=max((r['day_count'] for r in low), default=None)))
    return output


def diagnose(args):
    require(args.out_root != args.v163_root, 'analysis output must not replace sealed maps')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v164_*.py', '-v'], check=True)
    model = read(args.v163_root/'model.json')
    require(model['status'] == 'sealed' and model['execution_source'] == '35ae15b185' and
        not model['confirmation_started'] and not model['confirmation_release_eligible'],
        'requires sealed V163 with blocked, unopened confirmation')
    expected = {(a, m) for a in model['actors'] for m in original.protocol.MODES}
    require(len(model['chunks']) == len(expected) and
        {(r['actor'], r['mode']) for r in model['chunks']} == expected, 'sealed chunk grid incomplete')
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(bucket, args, model, chunk) for chunk in model['chunks']]
        for done, job in enumerate(as_completed(jobs), 1):
            results.append(job.result())
            print(f'V164_PROGRESS {done}/{len(jobs)}', flush=True)
    results.sort(key=lambda r: (r['counts']['actor'], r['counts']['mode']))
    key_rows = [r for result in results for r in result['keys']]
    day_rows = [r for result in results for r in result['days']]
    modes = []
    for mode in original.protocol.MODES:
        selected = [r['counts'] for r in results if r['counts']['mode'] == mode]
        entry = dict(mode=mode, **{f: sum(r[f] for r in selected) for f in selected[0] if f not in ('actor', 'mode')})
        expected = next(r for r in model['support_admission']['mode_scopes']
            if (r['mode'], r['scope']) == (mode, 'hour13'))
        close(entry['finite_mass'], expected['finite_fraction'])
        close(entry['finite_mass']+entry['native_abstention_mass']+entry['key_abstention_mass'], 1.)
        for fresh, old in (('eligible_queries', 'eligible_queries'), ('finite_queries', 'finite_queries'),
                ('native_abstentions', 'native_abstentions'), ('key_abstentions', 'key_abstentions')):
            require(entry[fresh] == expected[old], 'hour13 endpoint counts do not reproduce V163')
        entry['support_gap_to_80'] = .8-entry['finite_mass']
        entry['one_day_shortfall_mass'] = sum(r['mass'] for r in key_rows
            if r['mode'] == mode and not r['finite'] and r['day_shortfall'] == 1)
        entry['two_day_shortfall_mass'] = sum(r['mass'] for r in key_rows
            if r['mode'] == mode and not r['finite'] and r['day_shortfall'] == 2)
        entry['three_day_shortfall_mass'] = sum(r['mass'] for r in key_rows
            if r['mode'] == mode and not r['finite'] and r['day_shortfall'] == 3)
        modes.append(entry)
    fields = [f for f in model['frozen_protocol']['conditioning']['candidate_key'] if f not in ('actor', 'mode')]
    field_rows = [r for field in fields for r in group_keys(key_rows, [field], field)]
    patterns = group_keys(key_rows, fields, 'structural_pattern')
    days = group_keys(key_rows, ['day_count'], 'independent_calibration_days')
    actor_rows = [dict(actor=r['counts']['actor'], mode=r['counts']['mode'],
        finite_fraction=r['counts']['finite_mass']*len(model['actors']),
        native_abstention_fraction=r['counts']['native_abstention_mass']*len(model['actors']),
        key_abstention_fraction=r['counts']['key_abstention_mass']*len(model['actors'])) for r in results]
    day_marginals = []
    for mode in original.protocol.MODES:
        for day in model['calibration_days']:
            selected = [r for r in day_rows if (r['mode'], r['day']) == (mode, day)]
            require(len(selected) == len(model['actors']), 'day marginal incomplete')
            day_marginals.append(dict(mode=mode, day=day,
                **{f: float(np.mean([r[f] for r in selected])) for f in
                    ('finite_fraction', 'native_abstention_fraction', 'key_abstention_fraction')}))
    summary = dict(status='v164_readonly_sparse_key_audit_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], sealed_source='35ae15b185',
        v163_counts_and_weighted_support_reproduced=True, modes=modes, pattern_fields=fields,
        top_patterns={m: sorted((r for r in patterns if r['mode'] == m),
            key=lambda r: r['key_abstention_mass'], reverse=True)[:8] for m in original.protocol.MODES},
        independent_days='reported separately for each sealed actor-specific key, never pooled across actors',
        interval_fit=False, new_rollouts=0, new_training=0, regrouped_keys=False,
        sealed_maps_changed=False, confirmation_started=False, main_policy_changed=False,
        evidence='descriptive calibration support decomposition; not causal field attribution or future-day success')
    write(args.out_root/'details/key_support.json', key_rows)
    write(args.out_root/'details/structural_patterns.json', patterns)
    (args.out_root/'summary').mkdir(parents=True, exist_ok=True)
    for filename, rows in (('by_field.csv', field_rows), ('by_support_days.csv', days),
            ('by_actor.csv', actor_rows), ('by_day.csv', day_marginals)):
        original.csv_rows(args.out_root/'summary'/filename, rows)
    write(args.out_root/'summary/status.json', summary)
    print('V164_READONLY_SPARSE_KEY_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v163-root', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    diagnose(parser.parse_args())
