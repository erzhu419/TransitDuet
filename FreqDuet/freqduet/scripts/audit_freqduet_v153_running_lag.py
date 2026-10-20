#!/usr/bin/env python3
"""Localize frozen V151 running errors using saved causal motion prefixes."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import fmean
import subprocess
import sys

import audit_freqduet_v152_component_truth as prior


frozen, candidate, parts, stats = prior.frozen, prior.candidate, prior.parts, prior.stats
read, write, close = prior.read, prior.write, prior.close


class RunningPrefix:
    def __init__(self, path, geometry):
        self.rows = defaultdict(list)
        for (trip, station, direction), arrival in path['arrivals'].items():
            route = geometry[direction]
            i = route['index'][station]
            previous = path['departures'].get((trip, route['ids'][i-1], direction))
            start = path['dispatch'][trip]['actual_s'] if previous is None else previous['time_s']
            end = arrival['time_s']
            self.rows[direction, station].append((end, start, end-start))
        self.rows = {k: sorted(v) for k, v in self.rows.items()}
        self.times = {k: [r[0] for r in v] for k, v in self.rows.items()}

    def selected(self, direction, station, now):
        k = direction, station
        times = self.times.get(k, [])
        end = bisect_left(times, now)
        start = bisect_left(times, now-3600.)
        fallback = start == end
        if fallback:
            start = 0
        rows = self.rows.get(k, [])[start:end]
        if not rows:
            return None
        boundary = 3600.*int(now//3600)
        current = [r for r in rows if r[1] >= boundary]
        return dict(mean_s=fmean(r[2] for r in rows), count=len(rows),
            latest_age_s=now-rows[-1][0], mean_completion_age_s=fmean(now-r[0] for r in rows),
            before_hour_fraction=fmean(r[1] < boundary for r in rows),
            current_hour_n=len(current), fallback=float(fallback),
            current_hour_mean_s=fmean(r[2] for r in current) if current else None)


def observed_motion(path, row, route):
    """Use only arrivals/departures strictly before the query, not future labels."""
    now, trip = row['prediction_time_s'], stats.follower(row)[0]
    launch = path['dispatch'][trip]['actual_s']
    progress = float(route['positions'][route['index'][row['station_id']]])-row['evidence']['spatial_gap_m']
    residence, previous, segment_speed = 0., launch, None
    for station in route['ids'][1:]:
        k = trip, station, bool(row['direction'])
        arrival = path['arrivals'].get(k)
        if arrival is None or arrival['time_s'] >= now:
            i = route['index'][station]-1
            distance, elapsed = progress-float(route['positions'][i]), now-previous
            if distance > 0. and elapsed > 0.:
                segment_speed = distance/elapsed
            break
        departure = path['departures'].get(k)
        if departure is None or departure['time_s'] >= now:
            residence += now-arrival['time_s']
            break
        residence += departure['time_s']-arrival['time_s']
        previous = departure['time_s']
    elapsed = now-launch-residence
    return dict(observed_progress_m=progress, observed_running_s=elapsed,
        running_average_mps=progress/elapsed if progress > 0. and elapsed > 0. else None,
        current_segment_average_mps=segment_speed, evidence_source=row['evidence']['follower_source'],
        evidence_speed_mps=row['evidence']['speed_mps'])


def legs(path, row, p, prefix, history):
    route, now, direction = p['route'], p['now'], p['direction']
    trip = stats.follower(row)[0]
    result = []
    for i in range(p['origin']+1, p['target']+1):
        station = route['ids'][i]
        length = float(route['positions'][i]-route['positions'][i-1])
        distance = float(route['positions'][i]-p['progress']) if i == p['origin']+1 else length
        fraction = distance/length
        estimate = history.estimate(('running', direction, station), now)
        selected = prefix.selected(direction, station, now)
        close(selected['mean_s'], estimate['mean'])
        if selected['latest_age_s'] <= 0.:
            raise AssertionError('running prefix includes simultaneous/future completion')
        k = trip, station, direction
        before = path['departures'].get((trip, route['ids'][i-1], direction))
        start = path['dispatch'][trip]['actual_s'] if before is None else before['time_s']
        end = path['arrivals'][k]['time_s']
        truth = max(0., end-max(start, now))
        predicted = estimate['mean']*fraction
        result.append(dict(segment=f'{int(direction)}/{route["ids"][i-1]}->{station}',
            predicted_s=predicted, truth_s=truth, bias_s=predicted-truth, distance_m=distance,
            actual_entry_hour=int(start//3600), query_hour=int(now//3600),
            **selected, current_hour_clock_s=None if selected['current_hour_mean_s'] is None else
                fraction*selected['current_hour_mean_s']))
    return result


def budget(rows, segment_names):
    episodes = defaultdict(list)
    for row in rows:
        episodes[tuple(row['episode'])].append(row['segments'])
    values = {name: [] for name in segment_names}
    for queries in episodes.values():
        for name in segment_names:
            values[name].append(fmean(q.get(name, 0.) for q in queries))
    return {name: dict(queries=len(rows), episodes=len(episodes),
        metrics=dict(bias_contribution_s=fmean(v))) for name, v in values.items()}


def case(root, v152, out, runtime, seed, day, model):
    geometry = candidate.frozen.model.load_geometry(runtime)
    expected = read(v152/'cells'/f'{seed}_{day}.json')
    modes, launches, counts = {}, [], Counter()
    for mode in frozen.MODES:
        path, objects = parts.prepare(root/'confirmation', seed, day, mode, geometry)
        path['_observed_ready'] = candidate.ObservedReady(path['ready'])
        law = candidate.ServiceLaw(model['actors'][str(seed)], path['arrivals'], path['ready'], path['cutoff'])
        prefix = RunningPrefix(path, geometry)
        groups, distributions, allocations, reconciled = (defaultdict(list), defaultdict(list),
            defaultdict(list), defaultdict(list))
        saved = {tuple(r['query']): r for r in stats.trace(root/'confirmation/candidate_shadows'/
            f'{seed}_{day}_{mode}.jsonl.gz')}
        for k, row in path['trace'].items():
            label, truth = stats.target_label(row, path['ready'])
            s = saved[k]
            if label != 'resolved' or s['points']['v141'] is None:
                continue
            reference, _, _ = candidate.position(path, row, objects[-1], geometry, objects)
            if reference is None:
                packet, _, _ = parts.component_packet(path, row, path['shadows'][k], geometry, *objects)
                has_reference = packet is not None
            else:
                has_reference = True
            p, status, _ = candidate.position(path, row, objects[-1], geometry, objects, law, True)
            if status != s['details']['joint']['status']:
                raise AssertionError('frozen V151 running support changed')
            if p is None:
                p = reference
            if p is None:
                counts[mode+'/no_structured_running_clock'] += 1
                continue
            routes = geometry[bool(row['direction'])]
            episode = (*stats.follower(row), *k[1:])
            segments = legs(path, row, p, prefix, objects[0])
            predicted, actual = sum(r['predicted_s'] for r in segments), sum(r['truth_s'] for r in segments)
            exact = parts.actual_parts(path, row, routes)
            close(actual, exact['running'])
            scopes = ['all', 'hour/'+str(int(row['prediction_time_s']//3600))]
            if row['prediction_time_s'] >= path['cutoff']:
                scopes.append('clearance')
            if int(row['prediction_time_s']//3600) == 2:
                scopes.append('hour2')
            values = dict(episode=episode, running_bias_s=predicted-actual,
                running_mae_s=abs(predicted-actual), predicted_running_s=predicted, actual_running_s=actual)
            if has_reference:
                for scope in scopes:
                    if scope in ('all', 'clearance', 'hour2'):
                        reconciled[scope].append(values)
            if segments:
                values.update(mean_prefix_completion_age_s=fmean(r['mean_completion_age_s'] for r in segments),
                    mean_latest_completion_age_s=fmean(r['latest_age_s'] for r in segments),
                    before_hour_fraction=fmean(r['before_hour_fraction'] for r in segments),
                    no_current_hour_fraction=fmean(r['current_hour_n'] == 0 for r in segments))
            motion = observed_motion(path, row, routes)
            for scope in scopes:
                groups[scope].append(values)
                if segments and all(r['current_hour_clock_s'] is not None for r in segments):
                    clock = sum(r['current_hour_clock_s'] for r in segments)
                    groups[scope+'/same_hour_completed'].append(dict(episode=episode,
                        native_bias_s=predicted-actual, native_mae_s=abs(predicted-actual),
                        prefix_mix_gap_s=predicted-clock, same_hour_bias_s=clock-actual,
                        same_hour_mae_delta_s=abs(clock-actual)-abs(predicted-actual)))
                if scope in ('hour2', 'clearance'):
                    allocations[scope].append(dict(episode=episode,
                        segments={r['segment']: r['bias_s'] for r in segments}))
                    for r in segments:
                        distributions[scope+'/'+r['segment']].append(dict(episode=episode,
                            **{f: r[f] for f in ('predicted_s', 'truth_s', 'bias_s', 'mean_completion_age_s',
                                'latest_age_s', 'before_hour_fraction', 'count')}))
            distance = sum(r['distance_m'] for r in segments)
            for field in ('running_average_mps', 'current_segment_average_mps'):
                speed = motion[field]
                if speed is not None and distance > 0.:
                    clock = distance/speed
                    for scope in ('all', 'hour2', 'clearance'):
                        if scope in scopes:
                            groups[scope+'/'+field].append(dict(episode=episode,
                                observed_speed_mps=speed, implied_prefix_speed_mps=distance/predicted,
                                native_bias_s=predicted-actual, observable_clock_gap_s=predicted-clock,
                                observable_clock_residual_s=clock-actual,
                                observable_clock_mae_delta_s=abs(clock-actual)-abs(predicted-actual)))
            if 'clearance' in scopes and status == 'observed_same_clock_launch':
                launches.append(dict(seed=seed, day=day, mode=mode, query=list(k),
                    time_s=row['prediction_time_s'], motion=motion, segments=segments,
                    running_bias_s=predicted-actual))
        for scope, rows in reconciled.items():
            actual = stats.describe(rows)
            before = expected['modes'][mode]['components'][scope]
            if actual['queries'] != before['queries']:
                raise AssertionError('V152 paired running support changed')
            for f in ('bias', 'mae'):
                close(actual['metrics']['running_'+f+'_s'], before['metrics']['joint_running_'+f+'_s'])
        names = [f'{int(d)}/{a}->{b}' for d, r in geometry.items() for a, b in zip(r['ids'], r['ids'][1:])]
        budgets = {scope+'/'+name: v for scope, rows in allocations.items() for name, v in budget(rows, names).items()}
        for scope in allocations:
            close(sum(v['metrics']['bias_contribution_s'] for name, v in budgets.items() if name.startswith(scope+'/')),
                stats.describe(groups[scope])['metrics']['running_bias_s'])
        modes[mode] = dict(groups={g: stats.describe(r) for g, r in groups.items()}, budgets=budgets,
            segments={g: stats.describe(r) for g, r in distributions.items()})
    write(out/'cells'/f'{seed}_{day}.json', dict(seed=seed, day=day, modes=modes,
        counts=dict(counts), launches=launches, v152_running_reconciled=True))


def run(args):
    parent = read(args.v152_root/'summary/status.json')
    model = read(args.v151_root/'model.json')
    if parent['status'] != 'v152_readonly_component_truth_complete' or model['status'] != 'sealed':
        raise AssertionError('requires completed V152 and sealed V151')
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v153_running_lag.py', '-v'], check=True)
    case(args.v151_root, args.v152_root, args.out_root, args.runtime, *frozen.CASES[0], model)
    print('V153_FIRST_SAVED_PAIR_RECONCILED', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        jobs = [pool.submit(case, args.v151_root, args.v152_root, args.out_root, args.runtime, s, d, model)
            for s, d in frozen.CASES[1:]]
        for n, job in enumerate(as_completed(jobs), 2):
            job.result()
            if n % 32 == 0:
                print('V153_PROGRESS', n, len(frozen.CASES), flush=True)
    cells = [read(args.out_root/'cells'/f'{s}_{d}.json') for s, d in frozen.CASES]
    results = {}
    for mode in frozen.MODES:
        groups = stats.aggregate(cells, lambda c: c['modes'][mode]['groups'])
        budgets = stats.aggregate(cells, lambda c: c['modes'][mode]['budgets'])
        top = {scope: sorted((g for g in budgets if g.startswith(scope+'/')),
            key=lambda g: budgets[g]['metrics']['bias_contribution_s']['mean'])[:6]
            for scope in ('hour2', 'clearance')}
        selected = {g for values in top.values() for g in values}
        segments = stats.aggregate(cells, lambda c: {g: v for g, v in c['modes'][mode]['segments'].items() if g in selected})
        results[mode] = dict(groups=groups, most_negative_segments=top,
            budgets={g: budgets[g] for g in selected}, segments=segments)
    write(args.out_root/'summary/status.json', dict(status='v153_readonly_running_lag_complete',
        execution_source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], cases=len(cells), paths=2*len(cells),
        results=results, counts=dict(sum((Counter(c['counts']) for c in cells), Counter())),
        launches=[r for c in cells for r in c['launches']], v152_running_reconciled=True,
        segment_budgets_exact=True, new_rollouts=0, actor_updates=0, critic_updates=0, dual_updates=0,
        calibration_refit=False, main_policy_changed=False, v151_changed=False,
        v146_extensions_s=[3., 0.], window_s=3600., tails=[.1, .1],
        limitations='Consumed-path diagnosis, not new forecast/control validation. Top six negative '
            'segment contributions are descriptive selections, not multiplicity-adjusted gates. '
            'Zero budget for an unvisited segment is observed no contribution, not a missing outcome. '
            'Saved evidence speed is usually journey-average including stops, not instantaneous. '
            'Derived moving-time averages remove only already observed residence; current-segment '
            'average requires positive observed distance/time and is not instantaneous velocity. '
            'Same-hour completed subsets and speed clocks are diagnostic comparisons with their '
            'own support, not deployed forecasts. Future segments can change speed after the query; '
            'remaining residual cannot be uniquely assigned to stochasticity or lag. No new replay/fitting.'))
    print('V153_READONLY_RUNNING_LAG_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v151-root', 'v152-root', 'out-root', 'runtime'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    run(p.parse_args())
