#!/usr/bin/env python3
"""Read saved V145 intervals; retain the original independent-validation gate."""

import argparse
from collections import Counter, defaultdict
import gzip
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from freqduet_interval_validation import mean
from scripts.run_freqduet_protocol_v2_matrix import hierarchical_bootstrap


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def margins(cells, kind, scope, axis):
    groups = defaultdict(list)
    for c in cells:
        groups[c[axis]].append(c['strata'][kind][scope])
    rows = []
    for identity, stats in sorted(groups.items()):
        rows.append(dict(identity=identity, cells=len(stats),
            contributing_cells=sum(s['episode_coverage'] is not None for s in stats),
            episode_coverage=mean([s['episode_coverage'] for s in stats if s['episode_coverage'] is not None]),
            finite_fraction=mean([s['finite_fraction'] for s in stats if s['finite_fraction'] is not None])))
    complete = [r for r in rows if r['contributing_cells'] == r['cells']]
    values = [r['episode_coverage'] for r in complete]
    return rows, dict(groups=len(rows), complete_groups=len(complete),
        complete_group_mean=mean(values), minimum=None if not values else min(values),
        maximum=None if not values else max(values),
        below_target_n=sum(v < .8 for v in values),
        below_target_ids=[r['identity'] for r in complete if r['episode_coverage'] < .8])


def hour_contrast(cells, kind):
    rows = []
    for c in cells:
        a, b = [c['strata'][kind][s]['episode_coverage'] for s in ('hour13', 'rest')]
        if a is not None and b is not None:
            rows.append(dict(train_seed=c['seed'], eval_seed=c['day'], difference=a-b))
    result = dict(paired_cells=len(rows), registered_cells=len(cells), mean=None, ci95=None,
        decision='inconclusive_support')
    if len(rows) == len(cells):
        draws = hierarchical_bootstrap(pd.DataFrame(rows), 'difference', draws=5000)
        result.update(mean=mean([r['difference'] for r in rows]),
            ci95=np.percentile(draws, [2.5, 97.5]).tolist(), decision='complete_crossed_contrast')
    return result


def add(bucket, row):
    bucket['requests'] += 1
    bucket['pool_sum'] += row['pool_n']
    bucket['survivor_sum'] += row['survivor_n']
    bucket['pool_depleted_n'] += row['pool_n'] > row['survivor_n']
    finite = row['rank']['finite']
    bucket['finite_n'] += finite
    if not finite or row['label_status'] != 'resolved':
        return
    bucket['finite_resolved_n'] += 1
    for k in ('covered', 'below', 'above'):
        bucket[k+'_n'] += row[k]
    bucket['truth_sum_s'] += row['truth_s']
    bucket['lower_sum_s'] += row['rank']['lower_s']
    bucket['width_sum_s'] += row['rank']['upper_s']-row['rank']['lower_s']
    if row['below']:
        bucket['lower_shortfall_sum_s'] += row['rank']['lower_s']-row['truth_s']
    if row['above']:
        bucket['upper_excess_sum_s'] += row['truth_s']-row['rank']['upper_s']


def pool_summary(bucket):
    n, finite, resolved = [bucket[k] for k in ('requests', 'finite_n', 'finite_resolved_n')]
    return dict(requests=n, finite_n=finite, finite_resolved_n=resolved,
        finite_fraction=None if not n else finite/n,
        mean_pool_n=None if not n else bucket['pool_sum']/n,
        mean_survivor_n=None if not n else bucket['survivor_sum']/n,
        pool_depleted_n=bucket['pool_depleted_n'],
        query_coverage=None if not resolved else bucket['covered_n']/resolved,
        below_n=bucket['below_n'], above_n=bucket['above_n'],
        mean_truth_residual_s=None if not resolved else bucket['truth_sum_s']/resolved,
        mean_lower_s=None if not resolved else bucket['lower_sum_s']/resolved,
        mean_width_s=None if not resolved else bucket['width_sum_s']/resolved,
        mean_lower_shortfall_s=None if not bucket['below_n'] else bucket['lower_shortfall_sum_s']/bucket['below_n'],
        mean_upper_excess_s=None if not bucket['above_n'] else bucket['upper_excess_sum_s']/bucket['above_n'])


def labels(row):
    result = ['all', 'reason/'+row['support_reason'], 'hour13' if row['hour'] == 13 else 'rest']
    if row['hour'] == 13:
        result.append('hour13/direction/'+str(int(row['episode'][2])))
        n = row['rank']['n']
        result.append('hour13/rank_support/'+('0-8' if n < 9 else '9-18' if n < 19 else '19+'))
        if row['rank']['finite'] and row['label_status'] == 'resolved':
            result.append('hour13/'+('below' if row['below'] else 'above' if row['above'] else 'covered'))
    return result


def run(args):
    parent = read(args.v145_root/'summary/status.json')
    if parent['status'] != 'v145_independent_interval_validation_complete' or parent['main_policy_changed']:
        raise AssertionError('requires completed unchanged V145')
    reg = parent['registration']
    actors, days, modes = [reg[k] for k in ('archived_actors', 'demand_days', 'modes')]
    cells, buckets = [], defaultdict(Counter)
    cases = list(itertools.product(actors, days, modes))
    for i, (seed, day, mode) in enumerate(cases, 1):
        name = f'{seed}_{day}_{mode}'
        c = read(args.v145_root/'validation_cells'/(name+'.json'))
        if (c['seed'], c['day'], c['mode']) != (seed, day, mode):
            raise AssertionError('saved cell differs from registered grid')
        counts = Counter()
        with gzip.open(args.v145_root/'intervals'/(name+'.jsonl.gz'), 'rt') as f:
            for line in f:
                row = json.loads(line)
                counts[row['kind'], 'requests'] += 1
                counts[row['kind'], 'finite_n'] += row['rank']['finite']
                for label in labels(row):
                    add(buckets[mode, row['kind'], label], row)
        for kind in ('service', 'ready'):
            for field in ('requests', 'finite_n'):
                if counts[kind, field] != c['strata'][kind]['all'][field]:
                    raise AssertionError('saved interval cohort differs from V145 cell')
        cells.append(c)
        if i % 32 == 0:
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i, total=len(cases), unit='saved_path')), flush=True)
    results, details = {}, {}
    for mode in modes:
        selected = [c for c in cells if c['mode'] == mode]
        results[mode], details[mode] = {}, {}
        for kind in ('service', 'ready'):
            entry, detail = {}, {}
            for scope in ('all', 'hour13'):
                entry[scope], detail[scope] = {}, {}
                for axis in ('seed', 'day'):
                    rows, summary = margins(selected, kind, scope, axis)
                    entry[scope][axis], detail[scope][axis] = summary, rows
            entry['hour13_minus_rest'] = hour_contrast(selected, kind)
            pooled = {label: pool_summary(b) for (m, k, label), b in sorted(buckets.items()) if (m, k) == (mode, kind)}
            total = pooled['all']['requests']
            unsupported = total-pooled['all']['finite_n']
            for label, v in pooled.items():
                if label.startswith('reason/') and label != 'reason/finite':
                    v['fraction_of_all_requests'] = v['requests']/total if total else None
                    v['fraction_of_unsupported'] = v['requests']/unsupported if unsupported else None
            entry['query_diagnostics'] = pooled
            results[mode][kind], details[mode][kind] = entry, detail
    write(args.out_root/'details/marginals.json', details)
    write(args.out_root/'summary/status.json', dict(status='v145_readonly_coverage_audit_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], paths=len(cases), results=results,
        primary_finite_coverage_confirmed=parent['primary_finite_coverage_confirmed'],
        original_gate_replaced=False, interval_bounds_recomputed=False, point_forecasts_recomputed=False,
        new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Original V145 episode/actor/day gate remains authoritative. Axis margins are descriptive '
            'and use complete groups only; incomplete groups remain in remote details. Pooled query diagnostics '
            'do not replace episode-weighted inference. Hour13-rest contrasts use the complete paired crossed '
            'grid or remain inconclusive. These validation days are consumed, not new calibration data. '
            'No window, tail, point, actor or main-policy changes; no full-ETA or control-efficacy claim.'))
    print('V145_READONLY_COVERAGE_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v145-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v145_coverage_audit.py', '-v'], check=True)
    run(args)
