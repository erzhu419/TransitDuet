#!/usr/bin/env python3
"""Prequential component ETA on existing V118 paths; no control changes."""

import argparse
from bisect import bisect_left
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import audit_freqduet_v119_eta_clock as clock


prior = clock.prior
MODES = ('raw', 'running', 'stops', 'split')
WINDOW_S = 3600.


class CompletedMeans:
    def __init__(self, rows):
        groups = defaultdict(list)
        for key, end, value in rows:
            if value < 0:
                raise AssertionError('negative completed component duration')
            groups[key].append((end, value))
        self.groups = {}
        for key, values in groups.items():
            values.sort()
            times, numbers = zip(*values)
            self.groups[key] = (times, np.r_[0., np.cumsum(numbers)])

    def estimate(self, key, now):
        if key not in self.groups:
            return None
        times, sums = self.groups[key]
        end = bisect_left(times, now)
        start = bisect_left(times, now-WINDOW_S)
        historical = start == end
        if historical:
            start = 0
        if start == end:
            return None
        return dict(mean=float((sums[end]-sums[start])/(end-start)), count=end-start,
            latest=times[end-1], historical_fallback=historical)


def load_geometry(runtime):
    stops = pd.read_excel(runtime/'env/data/stop_news.xlsx')
    routes = pd.read_excel(runtime/'env/data/route_news.xlsx')
    lengths = {(r.start_stop, r.end_stop): float(r.distance) for r in routes.itertuples()}
    result = {}
    for direction in (True, False):
        order = list(stops.itertuples())
        if not direction:
            order.reverse()
        ids = [int(s.stop_id) for s in order]
        distances = [lengths[(a.stop_name, b.stop_name)] for a, b in zip(order, order[1:])]
        positions = np.r_[0., np.cumsum(distances)]
        result[direction] = dict(ids=ids, positions=positions, index={s: i for i, s in enumerate(ids)})
    return result


class MovingPrefix:
    def __init__(self, arrivals, departures):
        self.arrivals = sorted(arrivals, key=lambda r: r['time_s'])
        self.departures = sorted(departures, key=lambda r: r['time_s'])
        self.at = [r['time_s'] for r in self.arrivals]
        self.dt = [r['time_s'] for r in self.departures]

    def origin(self, now, progress, route):
        i, j = bisect_left(self.dt, now), bisect_left(self.at, now)
        if not i or (j and self.at[j-1] > self.dt[i-1]):
            return None
        origin = route['index'][int(self.departures[i-1]['station_id'])]
        if origin+1 == len(route['ids']):
            return None
        left, right = route['positions'][origin:origin+2]
        # At an arrival boundary the strictly earlier departure is insufficient.
        if not left-1e-8 <= progress < right-1e-8:
            return None
        return origin


def component_prediction(history, route, direction, origin, target, progress, now):
    totals = np.zeros(4)  # running, upstream service, holding, execution ticks
    estimates = []
    for i in range(origin+1, target+1):
        station = route['ids'][i]
        value = history.estimate(('running', direction, station), now)
        if value is None:
            return None
        fraction = (route['positions'][i]-progress)/(route['positions'][i]-route['positions'][i-1]) if i == origin+1 else 1.
        totals[0] += fraction*value['mean']
        estimates.append(value)
        if i < target:
            for part, channel in enumerate(('service', 'holding', 'ticks'), start=1):
                value = history.estimate((channel, direction, station), now)
                if value is None:
                    return None
                totals[part] += value['mean']
                estimates.append(value)
    service = history.estimate(('service', direction, route['ids'][target]), now)
    if service is None:
        return None
    estimates.append(service)
    if not all(e['latest'] < now for e in estimates):
        raise AssertionError('component forecast used an unfinished or simultaneous label')
    return dict(running_s=float(totals[0]), service_s=float(totals[1]), holding_s=float(totals[2]),
        ticks_s=float(totals[3]), target_service_s=service['mean'],
        minimum_support=min(v['count'] for v in estimates),
        historical_fallback=any(v['historical_fallback'] for v in estimates),
        latest_completion_s=max(v['latest'] for v in estimates))


def build_history(arrivals, departures, ready, geometry):
    samples = []
    for key, arrival in arrivals.items():
        trip, station, direction = key
        route = geometry[direction]
        i = route['index'][station]
        preceding = (trip, route['ids'][i-1], direction)
        if i == 0 or preceding not in departures:
            raise AssertionError('completed arrival lacks its adjacent preceding departure')
        samples.append((('running', direction, station), arrival['time_s'],
            arrival['time_s']-departures[preceding]['time_s']))
        end, start = departures[key], ready[key]['ready_time_s']
        held = end['realized_holding']-arrival['realized_holding']
        for channel, completion, value in (
                ('service', start, start-arrival['time_s']),
                ('holding', end['time_s'], held),
                ('ticks', end['time_s'], end['time_s']-start-held)):
            samples.append(((channel, direction, station), completion, value))
    return CompletedMeans(samples)


def read_cell(root, index, geometry):
    seed, day = prior.ITEMS[index]
    name = f'{seed}_{day}_{prior.frozen.PRIMARY}_0'
    trace = prior.read_trace(root/'traces'/(name+'.jsonl.gz'))
    ready = {prior.audit.key(r): r for r in prior.read_trace(root/'ready'/(name+'.jsonl.gz'))}
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        events = {k: [dict(zip(fields, r.tolist())) for r in record[k]] for k in ('arrivals', 'departures')}
        launches = {int(r[0]): dict(zip(record['launch_fields'].tolist(), r.tolist())) for r in record['launches']}
    arrivals = {prior.audit.key(r): r for r in events['arrivals']}
    departures = {prior.audit.key(r): r for r in events['departures']}
    by_a, by_d = defaultdict(dict), defaultdict(list)
    for key, row in arrivals.items():
        by_a[key[0]][key] = row
    for row in events['departures']:
        by_d[int(row['trip_id'])].append(row)
    prefixes = {trip: MovingPrefix(by_a[trip].values(), rows) for trip, rows in by_d.items()}
    clocks = {trip: clock.CompletedPrefix(rows, by_a[trip], ready) for trip, rows in by_d.items()}
    history = build_history(arrivals, departures, ready, geometry)
    groups = defaultdict(list)
    for row in trace:
        e, own = row['evidence'], prior.audit.key(row)
        base = dict(follower_eligible=bool(row['follower_valid']), follower_source=e['follower_source'])
        parts = prior.audit.action_parts(row)
        cuts = dict(negative=parts['negative_cut_s'], quota=-parts['quota'])
        if not row['follower_valid']:
            groups['unlabelled'].append(base)
            for cause, cut in cuts.items():
                if cut > 0:
                    groups[cause+'_unlabelled'].append(dict(base, cut_s=cut))
            continue
        now, direction = e['ready_time_s'], own[2]
        target = int(e['predicted_follower_trip_id']), own[1], direction
        truth = ready[target]['ready_time_s']-now
        departure_truth = departures[target]['time_s']-now
        if ready[target]['bus_id'] != e['predicted_follower_bus_id'] or truth < 0:
            raise AssertionError('same-object labels do not match frozen V118')
        prediction, projection = None, None
        status = 'other_eta_source'
        if e['follower_source'] == clock.JOURNEY:
            elapsed = now-launches[target[0]]['time_s']
            progress = e['speed_mps']*elapsed
            route = geometry[direction]
            target_index = route['index'][own[1]]
            np.testing.assert_allclose(progress+e['spatial_gap_m'], route['positions'][target_index], atol=1e-6, rtol=0)
            origin = prefixes[target[0]].origin(now, progress, route)
            status = 'not_causally_moving'
            if origin is not None:
                prediction = component_prediction(history, route, direction, origin, target_index, progress, now)
                status = 'missing_completed_component' if prediction is None else 'predicted'
            if prediction is not None:
                projection = clock.clock_projection(e, launches[target[0]]['time_s'], clocks[target[0]].before(now))
                base.update(prediction)
                future = clocks[target[0]].before(arrivals[target]['time_s'])-clocks[target[0]].before(now)
                np.testing.assert_allclose(future[0], arrivals[target]['realized_holding']-
                    clocks[target[0]].before(now)[0], atol=1e-8, rtol=0)
                actual_eta = arrivals[target]['time_s']-now
                truths = dict(running_s=actual_eta-float(future.sum()), service_s=float(future[1]),
                    holding_s=float(future[0]), ticks_s=float(future[2]),
                    target_service_s=ready[target]['ready_time_s']-arrivals[target]['time_s'])
                for field, actual in truths.items():
                    base[field+'_error'] = prediction[field]-actual
                    base[field+'_mae'] = abs(prediction[field]-actual)
        base['component_used'] = prediction is not None
        gaps = {mode: e['follower_gap_prediction_s'] for mode in MODES}
        if prediction is not None:
            raw_running = e['eta_prediction_s']-float(projection.sum())
            new_stops = prediction['service_s']+prediction['holding_s']+prediction['ticks_s']
            gaps.update(running=prediction['running_s']+float(projection.sum())+e['service_dwell_proxy_s'],
                stops=raw_running+new_stops+prediction['target_service_s'],
                split=prediction['running_s']+new_stops+prediction['target_service_s'])
        for mode, gap in gaps.items():
            base[mode+'_ready_error_s'] = gap-truth
            base[mode+'_ready_mae_s'] = abs(gap-truth)
            base[mode+'_ready_mae_delta_s'] = abs(gap-truth)-abs(gaps['raw']-truth)
        groups['forecasts'].append(base)
        groups[status].append(base)
        for cause, cut in cuts.items():
            if cut <= 0:
                continue
            executed = departures[own]['time_s']-now
            signs = clock.sign_metrics(gaps, truth, e['forward_gap_s'], executed, cut)
            values = dict(base, cut_s=cut, **signs)
            for mode, gap in gaps.items():
                values[mode+'_sign_delta'] = int(signs[mode+'_sign_disagrees'])-int(signs['raw_sign_disagrees'])
                values[mode+'_departure_sign_disagrees'] = bool(
                    (prior.prior.pair_restore_change(e['forward_gap_s'], gap, executed, cut) < 0) !=
                    (prior.prior.pair_restore_change(e['forward_gap_s'], departure_truth, executed, cut) < 0))
            for mode in MODES:
                values[mode+'_departure_sign_delta'] = int(values[mode+'_departure_sign_disagrees'])-int(values['raw_departure_sign_disagrees'])
            groups[cause].append(values)
            if prediction is not None:
                groups[cause+'_predicted'].append(values)
    return groups


def aggregate(root, out, runtime):
    parent = json.loads((root/'summary/status.json').read_text())
    if parent['status'] != 'v115_passive_identity_replay_complete':
        raise AssertionError('V120 requires the completed exact V118 replay')
    geometry = load_geometry(runtime)
    combined, cells = defaultdict(list), []
    fields = ['component_used', 'minimum_support', 'historical_fallback']
    fields += [p+'_'+m for p in ('running_s', 'service_s', 'holding_s', 'ticks_s', 'target_service_s') for m in ('error', 'mae')]
    fields += [v+'_'+m for v in MODES for m in ('ready_mae_s', 'ready_mae_delta_s',
        'sign_disagrees', 'sign_delta', 'false_positive', 'false_negative', 'departure_sign_delta')]
    for index, (seed, day) in enumerate(prior.ITEMS):
        groups = read_cell(root, index, geometry)
        means = {}
        for group, rows in groups.items():
            combined[group].extend(rows)
            if group in ('forecasts', 'predicted', 'negative', 'quota', 'negative_predicted', 'quota_predicted'):
                metrics = clock.summarize(rows)['metrics']
                means[group] = {m: metrics[m]['mean'] for m in fields if m in metrics}
        cells.append(dict(seed=seed, day=day, means=means))
        print('V120_COMPONENT_CELL_DONE', seed, day, flush=True)
    for cause in ('negative', 'quota'):
        if len(combined[cause]) != parent['cohorts'][cause]['metrics']['same_ready_error_s']['n']:
            raise AssertionError('V120 labelled cut denominator differs from V118')
    common = {g: sorted(set.intersection(*(set(c['means'][g]) for c in cells)))
        for g in cells[0]['means'] if all(g in c['means'] for c in cells)}
    inference_rows = [dict(seed=c['seed'], day=c['day'], values={g: {m: c['means'][g][m] for m in names}
        for g, names in common.items()}) for c in cells]
    intervals = prior.audit.inference.describe(inference_rows, 'values')
    advance = intervals['forecasts']['split_ready_mae_delta_s']['ci95'][1] < 0 and all(
        intervals[g]['split_sign_delta']['ci95'][1] < 0 and intervals[g]['split_departure_sign_delta']['mean'] <= 0
        for g in ('negative', 'quota'))
    prior.frozen.pilot.write_json(out/'summary/status.json', dict(status='v120_prequential_component_eta_complete',
        cohorts={g: clock.summarize(rows) for g, rows in combined.items()}, actor_day=cells,
        cell_mean_intervals=intervals, eligible_for_independent_predictor_confirmation=advance,
        window_s=WINDOW_S, source_paths=len(prior.ITEMS), replayed_paths=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, quota_changed=False, main_policy_changed=False,
        limitations='Consumed development days, not independent controller validation. Same-path '
            'rolling completed-label component means, not network training. Only causally confirmed '
            'moving followers use components; missing support, pending/boundary and other ETA sources '
            'retain raw estimates. Uniform motion within a remaining road segment; future holding '
            'is a same-stop completed mean, not a future actor rollout. Future events are labels only. '
            'Local restoration signs are not closed-loop system returns. Two actors/four days.'))
    print('V120_COMPONENT_COMPLETE independent_predictor_confirmation=', advance, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v118-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--runtime-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v120_*.py', '-v'], check=True)
    aggregate(a.v118_root, a.out_root, a.runtime_root)
