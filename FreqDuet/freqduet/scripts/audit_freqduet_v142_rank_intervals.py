#!/usr/bin/env python3
"""Repair only duration intervals; read frozen V140/V141 point forecasts."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v141_ready_eta as reference
from freqduet_rank_interval import completed_values, rank_interval


prior, frozen = reference.prior, reference.frozen
read, write, close, key = reference.read, reference.write, reference.close, reference.key


def record_interval(interval, linear, truth, episode, now, points):
    finite = interval['finite']
    return dict(points, rank=interval, truth_s=truth, episode=episode, time_s=now,
        linear_lower_s=None if linear is None else linear['p10'],
        linear_upper_s=None if linear is None else linear['p90'],
        conditional_point_s=None if linear is None else linear['mean'],
        covered=None if not finite else interval['lower_s'] <= truth <= interval['upper_s'],
        below=None if not finite else truth < interval['lower_s'],
        above=None if not finite else truth > interval['upper_s'])


def summarize(rows):
    finite = [r for r in rows if r['rank']['finite']]
    linear = [r for r in rows if r['linear_lower_s'] is not None]
    paired = [r for r in finite if r['linear_lower_s'] is not None]
    episodes = defaultdict(list)
    for r in finite:
        episodes[tuple(r['episode'])].append(r)
    def avg(values):
        return float(np.mean(values)) if values else None
    return dict(requests=len(rows), status_counts=dict(Counter(r['rank']['status'] for r in rows)),
        finite_n=len(finite), finite_fraction=None if not rows else len(finite)/len(rows),
        linear_n=len(linear), linear_coverage_all_available=avg([
            r['linear_lower_s'] <= r['truth_s'] <= r['linear_upper_s'] for r in linear]),
        paired_finite_n=len(paired), paired_linear_coverage=avg([
            r['linear_lower_s'] <= r['truth_s'] <= r['linear_upper_s'] for r in paired]),
        paired_rank_coverage=avg([r['covered'] for r in paired]),
        rank_below_rate=avg([r['below'] for r in finite]), rank_above_rate=avg([r['above'] for r in finite]),
        linear_mean_width_s=avg([r['linear_upper_s']-r['linear_lower_s'] for r in paired]),
        rank_mean_width_s=avg([r['rank']['upper_s']-r['rank']['lower_s'] for r in paired]),
        finite_episodes=len(episodes), episode_weighted_rank_coverage=avg([
            np.mean([r['covered'] for r in group]) for group in episodes.values()]))


def interval_record(kind, old, new, path, histories, prefix):
    forecast = old['forecast']
    current = tuple(forecast['current_stop_key'])
    now = path['trace'][key(old)]['evidence']['ready_time_s']
    if kind == 'service':
        start = path['arrivals'][current]['time_s']
        linear = forecast.get('service_residual')
    else:
        start = new['forecast']['ready_s']
        linear = new['forecast']['ready_residual']
    values, latest = completed_values(histories[kind], current[2], current[1], now, now-start)
    if latest is not None and latest >= now:
        raise AssertionError('rank interval used simultaneous or unfinished labels')
    if linear is not None:
        if len(values) != linear['count']:
            raise AssertionError('rank support differs from frozen conditional distribution')
        close(float(np.mean(values)), linear['mean'], 'frozen conditional point mean')
    elif len(values):
        raise AssertionError('unexpected conditional support for frozen missing interval')
    bounds = rank_interval(values)
    # Future outcomes enter scoring only after bounds and support are fixed.
    truth = ((path['ready'][current]['ready_time_s'] if kind == 'service'
        else path['departures'][current]['time_s'])-now)
    row = record_interval(bounds, linear, truth, [*prefix, kind, *current], now,
        dict(v140_gap_s=old['shadow_s'], v141_gap_s=new['shadow_s']))
    row.update(kind=kind, sample_latest_completion_s=latest,
        support_group='0' if not len(values) else '1-8' if len(values) < 9 else '9' if len(values) == 9 else '10+',
        direction=int(current[2]), hour=int(now//3600))
    return row


def path_audit(args, seed, day, mode):
    path = prior.load_path(args.v137_root, args.v135_root, seed, day, mode)
    v140 = read(args.v140_root/'details'/f'{seed}_{day}_{mode}.json')['rows']
    v141 = read(args.v141_root/'details'/f'{seed}_{day}_{mode}.json')['rows']
    lookup = {key(r): r for r in v141}
    if {key(r) for r in v140} != lookup.keys():
        raise AssertionError('frozen point-forecast support differs')
    histories = dict(service=reference.ServiceResiduals([(k[2], k[1], path['ready'][k]['ready_time_s'],
        path['ready'][k]['ready_time_s']-r['time_s']) for k, r in path['arrivals'].items()]),
        ready=reference.ServiceResiduals([(k[2], k[1], r['time_s'],
            r['time_s']-path['ready'][k]['ready_time_s']) for k, r in path['departures'].items() if k in path['ready']]))
    intervals, points = [], []
    for old in v140:
        new = lookup[key(old)]
        if old['original_fallback']:
            points.append(dict(baseline_s=old['shadow_s'], shadow_s=new['shadow_s'], truth_s=old['truth_s']))
        if old['forecast'].get('service_residual') is not None or old['forecast']['status'] == 'no_completed_service_survivor':
            intervals.append(interval_record('service', old, new, path, histories, [seed, day, mode]))
        if new['forecast']['status'] == 'passive_unissued_ready':
            intervals.append(interval_record('ready', old, new, path, histories, [seed, day, mode]))
    summary = dict(seed=seed, day=day, mode=mode, frozen_points=frozen.metrics(points),
        by_kind={kind: summarize([r for r in intervals if r['kind'] == kind]) for kind in ('service', 'ready')})
    write(args.out_root/'details'/f'{seed}_{day}_{mode}.json', dict(summary=summary, intervals=intervals))
    print('V142_RANK_PATH_DONE', seed, day, mode, flush=True)
    return intervals, points, summary


def run(args):
    parent = read(args.v141_root/'summary/status.json')
    if parent['status'] != 'v141_passive_ready_eta_complete' or parent['main_policy_changed']:
        raise AssertionError('requires completed frozen V141')
    rows, points, paths = [], [], []
    for p in parent['paths']:
        r, v, summary = path_audit(args, p['seed'], p['day'], p['mode'])
        rows.extend(r)
        points.extend(v)
        paths.append(summary)
    fixed = frozen.metrics(points)
    expected = parent['fallback_vs_v140']
    if fixed['n'] != expected['n']:
        raise AssertionError('frozen point cohort differs')
    for k in ('baseline_mae_s', 'shadow_mae_s', 'baseline_bias_s', 'shadow_bias_s'):
        close(fixed[k], expected[k], 'unchanged V140/V141 point forecasts')
    write(args.out_root/'summary/status.json', dict(status='v142_fixed_rank_intervals_complete', paths=paths,
        point_forecasts_recomputed=False, frozen_points=fixed,
        by_kind={kind: summarize([r for r in rows if r['kind'] == kind]) for kind in ('service', 'ready')},
        by_kind_and_support={kind: {g: summarize([r for r in rows if r['kind'] == kind and r['support_group'] == g])
            for g in sorted({r['support_group'] for r in rows if r['kind'] == kind})} for kind in ('service', 'ready')},
        service_by_hour={str(h): summarize([r for r in rows if r['kind'] == 'service' and r['hour'] == h])
            for h in sorted({r['hour'] for r in rows if r['kind'] == 'service'})},
        consumed_paths=12, new_rollouts=0, new_demand_days=0, actor_updates=0, critic_updates=0,
        dual_updates=0, action_selection_calls=0, main_policy_changed=False,
        limitations='Fixed 10% tails and original one-hour completed-prefix window; only rank construction '
            'changes. n<9 has no finite upper rank, encoded as upper_s=null with explicit status; '
            'unbounded intervals are neither successes nor failures in finite coverage. No-survivor '
            'service queries remain explicit, no mean/interval imputation. Nonnegative lower support '
            'is physical, not an empirical zero estimate. IID/exchangeable rank theory does not establish '
            'coverage for adaptive dependent completed-window bus data. Report finite availability and '
            'linear-vs-rank coverage on identical finite support. Current-stage intervals are not full '
            'ETA uncertainty. Same consumed selected paths, no independent calibration or control gain.'))
    print('V142_FIXED_RANK_INTERVALS_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v137-root', 'v135-root', 'v140-root', 'v141-root', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v142_*.py', '-v'], check=True)
    run(args)
