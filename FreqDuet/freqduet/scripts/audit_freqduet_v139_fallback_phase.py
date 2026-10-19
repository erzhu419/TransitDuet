#!/usr/bin/env python3
"""Classify frozen ETA fallback gates from saved V137/V138 events only."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
import subprocess
import sys

import audit_freqduet_v138_request_feedback as prior
import audit_freqduet_v120_component_eta as model


read, write, close, key = prior.read, prior.write, prior.close, prior.key


class FollowerTimeline:
    def __init__(self, arrivals, departures, ready, launch, route):
        departures = list(departures)
        if not any(int(r['station_id']) == route['ids'][0] for r in departures):
            departures.append(dict(station_id=route['ids'][0], time_s=launch))
        self.motion = model.MovingPrefix(arrivals, departures)
        self.ready = ready
        self.ready_times = sorted(r['ready_time_s'] for r in ready.values())
        self.route = route

    def inspect(self, now, progress):
        m, route = self.motion, self.route
        i, j = bisect_left(m.dt, now), bisect_left(m.at, now)
        boundaries = [name for name, times in (('arrival', m.at), ('departure', m.dt),
            ('ready', self.ready_times)) if bisect_left(times, now) < len(times)
            and times[bisect_left(times, now)] == now]
        previous_departure = m.departures[i-1] if i else None
        pending = bool(j and (not i or m.at[j-1] > m.dt[i-1]))
        phase = 'unlaunched' if not i else 'in_transit'
        if pending:
            arrival = m.arrivals[j-1]
            ready = self.ready[key(arrival)]['ready_time_s']
            phase = ('mandatory_service' if ready > now else
                'ready_boundary' if ready == now else 'ready_stop')
        origin, left, right = None, None, None
        if not i:
            reason = 'no_completed_departure'
        elif pending:
            reason = 'pending_stop'
        else:
            candidate = route['index'][int(previous_departure['station_id'])]
            if candidate+1 == len(route['ids']):
                reason = 'route_end'
            else:
                left, right = [float(v) for v in route['positions'][candidate:candidate+2]]
                if progress < left-1e-8:
                    reason = 'progress_before_completed_segment'
                elif progress >= right-1e-8:
                    reason = 'progress_at_or_beyond_next_stop'
                else:
                    reason, origin = 'moving_segment', candidate
        if origin != m.origin(now, progress, route):
            raise AssertionError('fallback explanation differs from original strict-prefix gate')
        return dict(reason=reason, strict_phase=phase, same_clock_events=boundaries,
            origin=origin, progress_m=float(progress), segment_left_m=left, segment_right_m=right,
            completed_departures=i, completed_arrivals=j,
            previous_departure_s=None if previous_departure is None else previous_departure['time_s'],
            previous_departure_station=None if previous_departure is None else int(previous_departure['station_id']))


def timelines(path, geometry):
    arrivals, departures, ready = defaultdict(list), defaultdict(list), defaultdict(dict)
    for k, r in path['arrivals'].items():
        arrivals[k[0]].append(r)
    for k, r in path['departures'].items():
        departures[k[0]].append(r)
    for k, r in path['ready'].items():
        ready[k[0]][k] = r
    return {trip: FollowerTimeline(arrivals[trip], departures[trip], ready[trip],
        row['actual_s'], geometry[bool(row['direction'])]) for trip, row in path['dispatch'].items()}


def inspect_row(row, path, indexed, geometry):
    e, base = row['evidence'], row['original_prediction']
    status = base['status']
    raw = e['follower_gap_prediction_s']
    if raw is None:
        if status != 'unavailable':
            raise AssertionError('unavailable raw evidence has a different original status')
        return dict(reason='unavailable', original_status=status)
    if e['follower_source'] != model.clock.JOURNEY:
        if status != 'other_eta_source':
            raise AssertionError('non-journey original status is inconsistent')
        return dict(reason='other_eta_source', original_status=status)
    trip, now, direction = e['predicted_follower_trip_id'], e['ready_time_s'], bool(row['direction'])
    progress = e['speed_mps']*(now-path['dispatch'][trip]['actual_s'])
    route = geometry[direction]
    close(progress+e['spatial_gap_m'], float(route['positions'][route['index'][row['station_id']]]),
        'saved journey clock and route position')
    result = indexed[trip].inspect(now, progress)
    blocked = result['origin'] is None
    if blocked != (status == 'not_causally_moving'):
        raise AssertionError(f'original fallback status not explained: {status}, {result}')
    if not blocked and status not in ('predicted', 'missing_completed_component'):
        raise AssertionError('unexpected completed-prefix component status')
    return dict(result, original_status=status)


def summarize(rows):
    return dict(decisions=len(rows), holding_delta_s=sum(r['delta_s'] for r in rows),
        holding_person_delta_s=sum(r['person_delta_s'] for r in rows),
        positive_holding_s=sum(max(0., r['delta_s']) for r in rows),
        negative_holding_s=sum(min(0., r['delta_s']) for r in rows),
        median_absolute_own_query_shift_s=median(abs(r['candidate_time_s']-r['base_time_s']) for r in rows),
        mean_absolute_realized_follower_gap_change_s=fmean(abs(r['realized_follower_delta_s']) for r in rows),
        mean_absolute_prediction_change_s=fmean(abs(r['prediction_delta_s']) for r in rows),
        same_clock_fallback_events=sum(bool(r['fallback']['same_clock_events']) for r in rows),
        fallback_phases=dict(Counter(r['fallback']['strict_phase'] for r in rows)))


def group_summaries(rows, field):
    return {name: summarize([r for r in rows if r[field] == name])
        for name in sorted({r[field] for r in rows})}


def case(args, seed, day, geometry, parent):
    paths = {m: prior.load_path(args.v137_root, args.v135_root, seed, day, m) for m in ('old', 'handoff')}
    inspections, gates = {}, {}
    for mode, path in paths.items():
        indexed = timelines(path, geometry)
        inspections[mode] = {k: inspect_row(r, path, indexed, geometry) for k, r in path['trace'].items()}
        values = list(inspections[mode].values())
        fallback = [r for r in values if r['original_status'] == 'not_causally_moving']
        gates[mode] = dict(decisions=len(values), original_status=dict(Counter(r['original_status'] for r in values)),
            fallback_reasons=dict(Counter(r['reason'] for r in fallback)),
            fallback_phases=dict(Counter(r['strict_phase'] for r in fallback)),
            fallback_same_clock_events=sum(bool(r['same_clock_events']) for r in fallback))
    detail = read(args.v138_root/'details'/f'{seed}_{day}.json')
    rows = []
    for r in detail['changed_commands']:
        if r['direct'] or r['category'] != 'source_change':
            continue
        k = key(r)
        b, c = [paths[m]['trace'][k] for m in ('old', 'handoff')]
        bi, ci = [inspections[m][k] for m in ('old', 'handoff')]
        if {bi['original_status'], ci['original_status']} != {'predicted', 'not_causally_moving'}:
            raise AssertionError('new source-switch regime outside admitted V138 comparison')
        fallback = ci if ci['original_status'] == 'not_causally_moving' else bi
        bp, cp = prior.request(b)[2], prior.request(c)[2]
        rows.append(dict(r, base_inspection=bi, candidate_inspection=ci, fallback=fallback,
            switch=bi['original_status']+' -> '+ci['original_status'],
            fallback_reason=fallback['reason'], fallback_phase=fallback['strict_phase'],
            prediction_delta_s=cp-bp, base_prediction_s=bp, candidate_prediction_s=cp,
            base_forward_s=b['context']['forward_headway_s'], candidate_forward_s=c['context']['forward_headway_s']))
    expected = parent['downstream_by_category']['source_change']
    observed = summarize(rows)
    close(observed['holding_delta_s'], expected['holding_delta_s'], 'changed source actions recover V138 source holding')
    if observed['decisions'] != expected['changed_actions']:
        raise AssertionError('V138 changed source-action support differs')
    bands = []
    for band in sorted(detail['timeline'], key=lambda r: -abs(r['waiting_person_s']))[:4]:
        selected = [r for r in rows if int(r['candidate_time_s']//900)*900 == band['start_s']]
        bands.append(dict(passenger_and_turnover=band,
            candidate_query_anchored_source_actions=group_summaries(selected, 'switch')))
    result = dict(seed=seed, day=day, effect=parent['effect'], original_gate_by_mode=gates,
        changed_source_actions=observed,
        unchanged_source_action_load_exposure_person_s=expected['holding_person_delta_s']-observed['holding_person_delta_s'],
        by_switch=group_summaries(rows, 'switch'), by_fallback_reason=group_summaries(rows, 'fallback_reason'),
        by_fallback_phase=group_summaries(rows, 'fallback_phase'),
        by_switch_and_reason=group_summaries([dict(r, group=r['switch']+' / '+r['fallback_reason']) for r in rows], 'group'),
        largest_source_exposure_changes=sorted(rows, key=lambda r: -abs(r['person_delta_s']))[:4],
        largest_waiting_change_bands=bands, delay_accounting=parent['delay_accounting'])
    write(args.out_root/'details'/f'{seed}_{day}.json', dict(summary=result, changed_source_actions=rows))
    print('V139_READONLY_PHASE_CASE_DONE', seed, day, flush=True)
    return result


def run(args):
    parent = read(args.v138_root/'summary/status.json')
    if parent['status'] != 'v138_readonly_request_feedback_complete' or parent['main_policy_changed']:
        raise AssertionError('requires completed frozen V138 accounting')
    geometry = model.load_geometry(args.runtime)
    cases = [case(args, r['seed'], r['day'], geometry, r) for r in parent['cases']]
    write(args.out_root/'summary/status.json', dict(status='v139_readonly_fallback_phase_complete', cases=cases,
        consumed_paths=12, new_rollouts=0, new_demand_days=0, parameter_changes=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Same six outcome-selected consumed pairs, no new policy confirmation or fitting. '
            'Original strict-prefix gate is unchanged; same-clock event flags and ready-stage labels '
            'are retrospective explanations, not new predictor inputs. Source-change action groups '
            'contain changed actions only; unchanged-action load exposure is separate. Candidate-query '
            'anchored action sums and passenger exposure bands are descriptive time alignment, not '
            'per-action mediation. No stop-level passenger demand/queue trace was saved.'))
    print('V139_READONLY_FALLBACK_PHASE_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('v137-root', 'v135-root', 'v138-root', 'runtime', 'out-root'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v139_*.py', '-v'], check=True)
    run(args)
