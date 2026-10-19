#!/usr/bin/env python3
"""Read saved V142 intervals; diagnose support and late-hour misses only."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import audit_freqduet_v142_rank_intervals as frozen


read, write, close = frozen.read, frozen.write, frozen.close


def support_context(history, direction, station, now, elapsed):
    samples = history.groups.get((direction, station), [])
    times = history.times.get((direction, station), [])
    end = bisect_left(times, now)
    recent = bisect_left(times, now-frozen.reference.model.WINDOW_S)
    start = 0 if recent == end else recent
    pool = samples[start:end]
    survivors = [(t, d) for t, d in pool if d >= elapsed]
    n = len(survivors)
    if n >= 9:
        reason = 'finite'
    elif not end:
        reason = 'no_completed_history'
    elif end < 9:
        reason = 'prefix_less_than_nine'
    elif len(pool) < 9:
        reason = 'window_less_than_nine'
    else:
        reason = 'elapsed_survivor_depletion'
    return dict(reason=reason, past_n=end, recent_n=end-recent, pool_n=len(pool),
        survivor_n=n, elapsed_s=elapsed, historical_fallback=start == 0 and recent == end,
        pool_mean_total_s=None if not pool else float(np.mean([d for _, d in pool])),
        survivor_mean_total_s=None if not survivors else float(np.mean([d for _, d in survivors])),
        latest_survivor_lag_s=None if not survivors else now-max(t for t, _ in survivors)), survivors


def start_cohort(events, now, elapsed):
    # Retrospective durations explain completion selection, never change bounds.
    window = frozen.reference.model.WINDOW_S
    started = [(a, b) for a, b in events if now-window <= a < now]
    completed = [(a, b) for a, b in started if b < now]
    pending = [(a, b) for a, b in started if b >= now]
    return dict(started_n=len(started), completed_started_n=len(completed), pending_n=len(pending),
        pending_already_older_than_query_n=sum(now-a >= elapsed for a, _ in pending),
        completed_mean_total_s=None if not completed else float(np.mean([b-a for a, b in completed])),
        pending_mean_total_s=None if not pending else float(np.mean([b-a for a, b in pending])))


def describe(rows):
    finite = [r for r in rows if r['rank']['finite']]
    def avg(values):
        return float(np.mean(values)) if values else None
    def mean_field(name):
        return avg([r[name] for r in rows if r[name] is not None])
    episodes = defaultdict(list)
    for r in finite:
        episodes[tuple(r['episode'])].append(r)
    return dict(n=len(rows), finite_n=len(finite),
        reasons=dict(Counter(r['reason'] for r in rows)),
        finite_fraction=None if not rows else len(finite)/len(rows),
        coverage=avg([r['covered'] for r in finite]),
        below_n=sum(r['below'] for r in finite), above_n=sum(r['above'] for r in finite),
        finite_episodes=len(episodes),
        earliest_query_coverage=avg([min(v, key=lambda r: r['time_s'])['covered'] for v in episodes.values()]),
        mean_pool_n=mean_field('pool_n'), mean_survivor_n=mean_field('survivor_n'),
        mean_elapsed_s=mean_field('elapsed_s'), mean_truth_residual_s=mean_field('truth_s'),
        mean_truth_total_s=mean_field('truth_total_s'),
        mean_pool_total_s=mean_field('pool_mean_total_s'),
        mean_survivor_total_s=mean_field('survivor_mean_total_s'),
        mean_pending_n=avg([r['retrospective_start_cohort']['pending_n'] for r in rows]),
        mean_pending_total_s=avg([r['retrospective_start_cohort']['pending_mean_total_s'] for r in rows
            if r['retrospective_start_cohort']['pending_mean_total_s'] is not None]),
        lower_shortfall_mean_s=avg([r['rank']['lower_s']-r['truth_s'] for r in finite if r['below']]),
        upper_excess_mean_s=avg([r['truth_s']-r['rank']['upper_s'] for r in finite if r['above']]))


def grouped(rows, field):
    return {str(v): describe([r for r in rows if r[field] == v]) for v in sorted({r[field] for r in rows})}


def path_audit(args, p):
    seed, day, mode = p['seed'], p['day'], p['mode']
    path = frozen.prior.load_path(args.v137_root, args.v135_root, seed, day, mode)
    events = {kind: defaultdict(list) for kind in ('service', 'ready')}
    for k, r in path['arrivals'].items():
        events['service'][k[2], k[1]].append((r['time_s'], path['ready'][k]['ready_time_s']))
    for k, r in path['departures'].items():
        if k in path['ready']:
            events['ready'][k[2], k[1]].append((path['ready'][k]['ready_time_s'], r['time_s']))
    histories = {kind: frozen.reference.ServiceResiduals([(d, s, b, b-a)
        for (d, s), values in groups.items() for a, b in values]) for kind, groups in events.items()}
    rows = read(args.v142_root/'details'/f'{seed}_{day}_{mode}.json')['intervals']
    for r in rows:
        kind, current, now = r['kind'], tuple(r['episode'][-3:]), r['time_s']
        start = path['arrivals'][current]['time_s'] if kind == 'service' else path['ready'][current]['ready_time_s']
        context, survivors = support_context(histories[kind], current[2], current[1], now, now-start)
        if context['survivor_n'] != r['rank']['n']:
            raise AssertionError('diagnostic support differs from saved V142')
        if r['rank']['finite']:
            values = sorted(d-(now-start) for _, d in survivors)
            close(values[r['rank']['lower_rank']-1], r['rank']['lower_s'], 'frozen lower bound')
            close(values[r['rank']['upper_rank']-1], r['rank']['upper_s'], 'frozen upper bound')
        finish = path['ready'][current]['ready_time_s'] if kind == 'service' else path['departures'][current]['time_s']
        close(finish-now, r['truth_s'], 'same saved duration outcome')
        r.update(context, truth_total_s=r['truth_s']+now-start,
            station=str(current[1]), mode=mode, actor=seed, day=day,
            path=f'{seed}_{day}_{mode}',
            retrospective_start_cohort=start_cohort(events[kind][current[2], current[1]], now, now-start))
    write(args.out_root/'details'/f'{seed}_{day}_{mode}.json', rows)
    return rows


def run(args):
    parent = read(args.v142_root/'summary/status.json')
    if parent['status'] != 'v142_fixed_rank_intervals_complete' or parent['main_policy_changed']:
        raise AssertionError('requires completed frozen V142')
    rows, started = [], time.monotonic()
    for i, p in enumerate(parent['paths']):
        rows.extend(path_audit(args, p))
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(current=i+1, total=len(parent['paths']),
            unit='saved_path', elapsed_seconds=time.monotonic()-started)), flush=True)
    for kind in ('service', 'ready'):
        same = frozen.summarize([r for r in rows if r['kind'] == kind])
        if same != parent['by_kind'][kind]:
            raise AssertionError('diagnostic changed frozen interval cohort or coverage')
    service = [r for r in rows if r['kind'] == 'service']
    low = [r for r in service if not r['rank']['finite']]
    tail = [r for r in service if r['hour'] == 13]
    write(args.out_root/'summary/status.json', dict(status='v143_readonly_support_complete',
        frozen_intervals_and_coverage_exact=True,
        by_kind={k: describe([r for r in rows if r['kind'] == k]) for k in ('service', 'ready')},
        service_low_support=grouped(low, 'reason'), service_by_hour=grouped(service, 'hour'),
        service_by_support=grouped(service, 'support_group'),
        hour13_by_path=grouped(tail, 'path'), hour13_by_station=grouped(tail, 'station'),
        hour13_by_direction=grouped(tail, 'direction'), hour13_by_mode=grouped(tail, 'mode'),
        hour13_by_outcome=grouped([dict(r, outcome='unsupported' if not r['rank']['finite'] else
            'below' if r['below'] else 'above' if r['above'] else 'covered') for r in tail], 'outcome'),
        service_low_by_hour=grouped(low, 'hour'),
        actor_updates=0, critic_updates=0, dual_updates=0, new_rollouts=0, new_demand_days=0,
        action_selection_calls=0, point_forecasts_recomputed=False, interval_updates=0,
        main_policy_changed=False,
        limitations='Read-only consumed selected V137/V142 paths. Finite coverage has its own denominator; '
            'missing support is not a failure or success. Retrospective pending durations are outcome '
            'diagnostics, never available forecast inputs. Correlated overlapping history and selected '
            'queries do not identify causal failure mechanisms or independent calibration. No window, '
            'tail, point or interval tuning; no replay, training or control execution.'))
    print('V143_READONLY_SUPPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v137-root', 'v135-root', 'v142-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v14[23]_*.py', '-v'], check=True)
    run(args)
