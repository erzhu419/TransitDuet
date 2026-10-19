#!/usr/bin/env python3
"""Add an unissued-ready shadow to frozen V140; diagnose its service intervals."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys

import numpy as np

import audit_freqduet_v140_stop_eta as frozen
from freqduet_stop_phase_eta import ServiceResiduals


prior, model = frozen.prior, frozen.model
read, write, close, key = frozen.read, frozen.write, frozen.close, frozen.key


def predict_ready(history, durations, route, direction, origin, target, now, ready_s):
    if ready_s >= now or origin >= target:
        return dict(applied=False, status='ready_not_observed_or_target_already_ready')
    residual = durations.estimate(direction, route['ids'][origin], now, now-ready_s)
    if residual is None:
        return dict(applied=False, status='no_completed_ready_residence_survivor')
    tail = model.component_prediction(history, route, direction, origin, target,
        float(route['positions'][origin]), now)
    if tail is None:
        return dict(applied=False, status='missing_downstream_components')
    latest = max(residual['latest'], tail['latest_completion_s'])
    if latest >= now:
        raise AssertionError('ready forecast used simultaneous or unfinished labels')
    return dict(applied=True, status='passive_unissued_ready', gap_s=residual['mean']+
        sum(tail[k] for k in ('running_s', 'service_s', 'holding_s', 'ticks_s', 'target_service_s')),
        ready_residual=residual, ready_s=ready_s, latest_label_s=latest)


def interval_row(values, lower, upper, truth, episode, now):
    return dict(episode=episode, time_s=now, covered=bool(lower <= truth <= upper),
        below=bool(truth < lower), above=bool(truth > upper),
        outside_empirical=bool(truth < min(values) or truth > max(values)),
        empirical_containment=float(np.mean((values >= lower) & (values <= upper))),
        support=len(values), width_s=upper-lower, mean_bias_s=float(np.mean(values)-truth))


def interval_metrics(rows):
    if not rows:
        return dict(n=0, coverage=None)
    episodes = defaultdict(list)
    for r in rows:
        episodes[tuple(r['episode'])].append(r)
    return dict(n=len(rows), episodes=len(episodes), coverage=float(np.mean([r['covered'] for r in rows])),
        episode_weighted_coverage=float(np.mean([np.mean([r['covered'] for r in group]) for group in episodes.values()])),
        earliest_query_coverage=float(np.mean([min(group, key=lambda r: r['time_s'])['covered'] for group in episodes.values()])),
        below_rate=float(np.mean([r['below'] for r in rows])), above_rate=float(np.mean([r['above'] for r in rows])),
        outside_empirical_rate=float(np.mean([r['outside_empirical'] for r in rows])),
        mean_past_empirical_containment=float(np.mean([r['empirical_containment'] for r in rows])),
        median_support=float(np.median([r['support'] for r in rows])),
        median_width_s=float(np.median([r['width_s'] for r in rows])),
        mean_bias_s=float(np.mean([r['mean_bias_s'] for r in rows])))


def service_interval(saved, path, services, episode_prefix):
    forecast = saved['forecast']
    current = tuple(forecast['current_stop_key'])
    now = path['trace'][key(saved)]['evidence']['ready_time_s']
    age = now-path['arrivals'][current]['time_s']
    k = current[2], current[1]
    times = services.times[k]
    end = bisect_left(times, now)
    start = bisect_left(times, now-model.WINDOW_S)
    if start == end:
        start = 0
    values = np.asarray([duration-age for _, duration in services.groups[k][start:end] if duration >= age])
    residual = forecast['service_residual']
    if len(values) != residual['count']:
        raise AssertionError('service diagnosis does not recover frozen V140 conditional support')
    row = interval_row(values, residual['p10'], residual['p90'], saved['service_remaining_truth_s'],
        [*episode_prefix, *current], now)
    n = len(values)
    row.update(support_group='1' if n == 1 else '2-4' if n < 5 else '5-9' if n < 10 else '10-19' if n < 20 else '20+',
        station_id=current[1], direction=int(current[2]), hour=int(now//3600), historical_fallback=residual['historical_fallback'])
    return row


def path_audit(args, seed, day, mode, geometry):
    path = prior.load_path(args.v137_root, args.v135_root, seed, day, mode)
    saved = read(args.v140_root/'details'/f'{seed}_{day}_{mode}.json')
    launches = {trip: dict(time_s=r['actual_s']) for trip, r in path['dispatch'].items()}
    history = model.build_history(path['arrivals'], path['departures'], path['ready'], launches, geometry)
    services = ServiceResiduals([(k[2], k[1], path['ready'][k]['ready_time_s'],
        path['ready'][k]['ready_time_s']-r['time_s']) for k, r in path['arrivals'].items()])
    durations = ServiceResiduals([(k[2], k[1], r['time_s'],
        r['time_s']-path['ready'][k]['ready_time_s']) for k, r in path['departures'].items() if k in path['ready']])
    rows, intervals, statuses, ready_intervals = [], [], Counter(), []
    for old in saved['rows']:
        reference = old['forecast']
        result = dict(applied=False, status='unchanged_v140')
        if reference['status'] == 'ready_without_prior_command':
            current = tuple(reference['current_stop_key'])
            r = path['trace'][key(old)]
            now = r['evidence']['ready_time_s']
            route = geometry[current[2]]
            result = predict_ready(history, durations, route, current[2], route['index'][current[1]],
                route['index'][old['station_id']], now, path['ready'][current]['ready_time_s'])
            if result['applied']:
                truth = path['departures'][current]['time_s']-now
                residual = result['ready_residual']
                ready_intervals.append(dict(covered=residual['p10'] <= truth <= residual['p90'],
                    mean_error_s=residual['mean']-truth))
        statuses[result['status']] += 1
        rows.append(dict(trip_id=old['trip_id'], station_id=old['station_id'], direction=old['direction'],
            baseline_s=old['shadow_s'], shadow_s=result['gap_s'] if result['applied'] else old['shadow_s'],
            native_s=old['baseline_s'], truth_s=old['truth_s'], new_branch=result['applied'],
            original_fallback=old['original_fallback'], forecast=result))
        if 'service_remaining_truth_s' in old:
            intervals.append(service_interval(old, path, services, [seed, day, mode]))
    fallback = [r for r in rows if r['original_fallback']]
    summary = dict(seed=seed, day=day, mode=mode, vs_v140=frozen.metrics(fallback),
        vs_native=frozen.metrics([dict(r, baseline_s=r['native_s']) for r in fallback]),
        new_ready_support=frozen.metrics([r for r in rows if r['new_branch']]), statuses=dict(statuses),
        current_ready_interval=dict(n=len(ready_intervals), coverage=None if not ready_intervals else
            float(np.mean([r['covered'] for r in ready_intervals])), mae_s=None if not ready_intervals else
            float(np.mean([abs(r['mean_error_s']) for r in ready_intervals]))),
        frozen_service_interval=interval_metrics(intervals))
    close(summary['vs_v140']['baseline_mae_s'], saved['summary']['original_fallback_support']['shadow_mae_s'],
        'same fallback support as frozen V140')
    write(args.out_root/'details'/f'{seed}_{day}_{mode}.json', dict(summary=summary, rows=rows, service_interval_rows=intervals))
    print('V141_PASSIVE_PATH_DONE', seed, day, mode, flush=True)
    return rows, intervals, summary


def pair_audit(args, seed, day, levels):
    source = read(args.v139_root/'details'/f'{seed}_{day}.json')['changed_source_actions']
    lookup = {m: {key(r): r for r in levels[m][0]} for m in levels}
    rows = []
    for r in source:
        b, c = [lookup[m][key(r)] for m in ('old', 'handoff')]
        rows.append(dict(baseline_s=c['baseline_s']-b['baseline_s'], shadow_s=c['shadow_s']-b['shadow_s'],
            native_s=c['native_s']-b['native_s'], truth_s=c['truth_s']-b['truth_s']))
    return rows, dict(seed=seed, day=day, vs_v140=frozen.metrics(rows),
        vs_native=frozen.metrics([dict(r, baseline_s=r['native_s']) for r in rows]))


def run(args):
    parent = read(args.v140_root/'summary/status.json')
    if parent['status'] != 'v140_passive_stop_eta_complete' or parent['main_policy_changed']:
        raise AssertionError('requires frozen V140 shadow results')
    geometry = model.load_geometry(args.runtime)
    rows, intervals, paths, pair_rows, pairs = [], [], [], [], []
    for p in parent['pairs']:
        seed, day = p['seed'], p['day']
        levels = {m: path_audit(args, seed, day, m, geometry) for m in ('old', 'handoff')}
        for mode in levels:
            rows.extend(levels[mode][0])
            intervals.extend(levels[mode][1])
            paths.append(levels[mode][2])
        r, result = pair_audit(args, seed, day, levels)
        pair_rows.extend(r)
        pairs.append(result)
    fallback = [r for r in rows if r['original_fallback']]
    write(args.out_root/'summary/status.json', dict(status='v141_passive_ready_eta_complete', paths=paths, pairs=pairs,
        fallback_vs_v140=frozen.metrics(fallback), fallback_vs_native=frozen.metrics([dict(r, baseline_s=r['native_s']) for r in fallback]),
        new_ready_support=frozen.metrics([r for r in rows if r['new_branch']]),
        source_switch_vs_v140=frozen.metrics(pair_rows), source_switch_vs_native=frozen.metrics([dict(r, baseline_s=r['native_s']) for r in pair_rows]),
        statuses=dict(Counter(r['forecast']['status'] for r in rows)), service_interval=interval_metrics(intervals),
        service_by_support={s: interval_metrics([r for r in intervals if r['support_group'] == s]) for s in sorted({r['support_group'] for r in intervals})},
        service_by_hour={str(h): interval_metrics([r for r in intervals if r['hour'] == h]) for h in sorted({r['hour'] for r in intervals})},
        service_by_direction={str(d): interval_metrics([r for r in intervals if r['direction'] == d]) for d in (0, 1)},
        consumed_paths=12, new_rollouts=0, new_demand_days=0, action_selection_calls=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Same consumed selected development paths; no independent confirmation or control gains. '
            'V140 is unchanged outside its ready-without-prior-command branch. Ready residence includes '
            'future holding uncertainty but uses only completed past departures, never current future '
            'commands. Original one-hour window and p10/p90 remain fixed. Service interval diagnosis '
            'uses V140 saved forecasts; empirical self-containment is not held-out coverage. Query, '
            'episode-weighted and earliest-query coverage use the same retrospective labels. Unsupported '
            'service cases retain their original missing-interval status, not zero coverage.'))
    print('V141_PASSIVE_READY_ETA_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v137-root', 'v135-root', 'v139-root', 'v140-root', 'runtime', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v141_*.py', '-v'], check=True)
    run(args)
