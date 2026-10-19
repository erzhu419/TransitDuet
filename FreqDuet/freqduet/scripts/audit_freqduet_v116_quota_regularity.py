#!/usr/bin/env python3
"""Read V115 paths: removed compensation, local geometry and exact CV2 accounting."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from freqduet_incremental_compensation import ACTIONS, bounded_action
import run_freqduet_v115_funded_quota as frozen
import run_freqduet_v94_compensation_confirmation as inference


ARRIVAL_PARTS = ('release', 'quarter_shadow', 'negative_pulse', 'positive_pulse', 'quota', 'other_elapsed')
DEPARTURE_PARTS = ('arrival_time', 'service_dwell', 'current_action')


def key(row):
    return int(row['trip_id']), int(row['station_id']), bool(row['direction'])


def action_parts(row):
    shadow, chosen = row['shadow_s'], row['chosen_s']
    pulse = ACTIONS[bounded_action(row['raw_request_s'], row['feedback_error_s'], [True]*7)]
    if chosen > pulse or chosen != ACTIONS[bounded_action(row['raw_request_s'],
            min(max(row['feedback_error_s'], 0.), row['available_s']), [True]*7)]:
        raise AssertionError('completed command differs from frozen pulse/quota arithmetic')
    negative_cut = max(shadow-pulse, 0.)
    quota_cut = max(min(shadow, pulse)-chosen, 0.)
    if negative_cut and row['increment_s'] >= 0:
        raise AssertionError('negative pulse attribution has no negative current response')
    if negative_cut+quota_cut != max(shadow-chosen, 0.):
        raise AssertionError('baseline cuts double-count the pulse and funding stages')
    return dict(quarter_shadow=shadow, negative_pulse=min(pulse-shadow, 0.),
        positive_pulse=max(pulse-shadow, 0.), quota=chosen-pulse,
        negative_cut_s=negative_cut, quota_cut_s=quota_cut,
        extra_quota_withheld_s=pulse-chosen-quota_cut, added_s=max(chosen-shadow, 0.))


def series(events, names):
    groups, gaps, parts = defaultdict(list), [], []
    for row in events:
        group = groups[(int(row['station_id']), bool(row['direction']))]
        if group:
            previous = group[-1]
            gaps.append(row['time_s']-previous['time_s'])
            parts.append([row[n]-previous[n] for n in names])
        group.append(row)
    h, c = np.asarray(gaps), np.asarray(parts)
    if np.any(h < 0):
        raise AssertionError('event order is not the completed chronological headway order')
    np.testing.assert_allclose(c.sum(axis=1), h, atol=1e-8, rtol=0)
    mean, variance = float(h.mean()), float(h.var())
    covariance = ((c-c.mean(axis=0))*(h-mean)[:, None]).mean(axis=0)
    np.testing.assert_allclose(covariance.sum(), variance, atol=1e-7, rtol=1e-12)
    return dict(samples=len(h), mean_s=mean, variance_s2=variance, cv_squared=variance/mean**2,
        covariance_s2=dict(zip(names, covariance.tolist())),
        first_sum_s=sum(g[0]['time_s'] for g in groups.values()),
        last_sum_s=sum(g[-1]['time_s'] for g in groups.values()),
        orders={str(k): [int(r['trip_id']) for r in v] for k, v in groups.items()})


def effect(base, candidate, names):
    mb, mc = base['mean_s'], candidate['mean_s']
    result = {n: (candidate['covariance_s2'][n]-base['covariance_s2'][n])/mb**2 for n in names}
    result['mean_normalization'] = candidate['variance_s2']*(1/mc**2-1/mb**2)
    np.testing.assert_allclose(sum(result.values()), candidate['cv_squared']-base['cv_squared'], atol=1e-10, rtol=0)
    result.update(cv_squared=candidate['cv_squared']-base['cv_squared'], mean_headway_s=mc-mb,
        raw_variance_s2=candidate['variance_s2']-base['variance_s2'])
    nb, nc = base['samples'], candidate['samples']
    result['first_boundary_mean_s'] = -(candidate['first_sum_s']-base['first_sum_s'])/nb
    result['last_boundary_mean_s'] = (candidate['last_sum_s']-base['last_sum_s'])/nb
    result['count_normalization_mean_s'] = mc*nc*(1/nc-1/nb)
    np.testing.assert_allclose(sum(result[n] for n in ('first_boundary_mean_s', 'last_boundary_mean_s',
        'count_normalization_mean_s')), mc-mb, atol=1e-8, rtol=0)
    return result


def coordinates(arrivals, departures, decisions):
    dep = {key(r): r for r in departures}
    totals = defaultdict(lambda: defaultdict(float))
    result_arr, result_dep, actions = [], {}, {}
    for row in arrivals:
        identity = key(row)
        end = dep[identity]
        actual = end['realized_holding']-row['realized_holding']
        trace = decisions.get(identity) if decisions is not None else None
        if decisions is not None and trace is None and actual != 0:
            raise AssertionError('uncontrolled initial service executed untraced holding')
        parts = action_parts(trace) if trace is not None else dict(
            quarter_shadow=actual, negative_pulse=0., positive_pulse=0., quota=0.)
        cumulative = totals[identity[0]]
        np.testing.assert_allclose(sum(cumulative.values()), row['realized_holding'], atol=1e-8, rtol=0)
        if trace is not None:
            if (trace['executed_s'] != actual or trace['chosen_s'] != actual or
                    trace['executed_before_s'] != row['realized_holding'] or
                    not row['time_s'] <= trace['policy_time_s'] <= end['time_s']):
                raise AssertionError('decision trace is not aligned to completed execution')
        result_arr.append(dict(row, **{n: cumulative[n] for n in ARRIVAL_PARTS[1:-1]}))
        result_dep[identity] = dict(end, arrival_time=row['time_s'],
            service_dwell=end['time_s']-actual-row['time_s'], current_action=actual)
        actions[identity] = actual
        for n in ARRIVAL_PARTS[1:-1]:
            cumulative[n] += parts[n]
    observed = {key(r) for r in arrivals}
    if decisions is not None and not decisions.keys() <= observed:
        raise AssertionError('a completed decision is missing from observed stop support')
    return result_arr, [result_dep[key(r)] for r in departures if key(r) in observed], actions


def geometry_change(times, shifted, moment):
    old, new = np.diff(times), np.diff(np.sort(shifted, kind='stable'))
    dq, ds, n = float(new@new-old@old), float(new.sum()-old.sum()), moment['samples']
    m = moment['mean_s']+ds/n
    q = moment['variance_s2']+moment['mean_s']**2+dq/n
    return dict(squared_gap_change_s2=dq, mean_headway_change_s=ds/n,
        cv_squared_change=q/m**2-1-moment['cv_squared'])


def local_geometry(departures, decisions, moment):
    groups = defaultdict(list)
    for r in departures:
        groups[(int(r['station_id']), bool(r['direction']))].append(r)
    cohorts = defaultdict(Counter)
    all_changes = Counter()
    for events in groups.values():
        times = np.asarray([r['time_s'] for r in events])
        shifts = {name: np.zeros(len(events)) for name in ('negative', 'quota', 'all_cuts', 'all_shadows')}
        for i, event in enumerate(events):
            if key(event) not in decisions:
                continue
            row, parts = decisions[key(event)], action_parts(decisions[key(event)])
            for cause in ('negative', 'quota'):
                cut = parts[cause+'_cut_s']
                shifts[cause][i] = cut
                if not cut:
                    continue
                modified = times.copy()
                modified[i] += cut
                delta = geometry_change(times, modified, moment)
                useful = delta['cv_squared_change'] < -1e-12
                harmful = delta['cv_squared_change'] > 1e-12
                order_changed = not np.array_equal(np.argsort(modified, kind='stable'), np.arange(len(times)))
                load = row['onboard_passengers']
                names = (cause, cause+('_follower_valid' if row['follower_valid'] else '_follower_missing'),
                    cause+('_empty' if load == 0 else '_load1_20' if load <= 20 else '_load_over20'),
                    cause+'_station_'+str(int(row['station_id'])))
                for name in names:
                    c = cohorts[name]
                    c['decisions'] += 1
                    c['cut_s'] += cut
                    c['useful_restores'] += int(useful)
                    c['harmful_restores'] += int(harmful)
                    c['useful_cut_s'] += cut*useful
                    c['harmful_cut_s'] += cut*harmful
                    c['raw_squared_gap_useful_restores'] += int(delta['squared_gap_change_s2'] < 0)
                    c['order_changes'] += int(order_changed)
                    c['boundary_events'] += int(i in (0, len(times)-1))
                    c['onboard_exposure_if_restored_person_s'] += cut*load
                    c['single_event_squared_gap_change_s2_sum'] += delta['squared_gap_change_s2']
            shifts['all_cuts'][i] = parts['negative_cut_s']+parts['quota_cut_s']
            shifts['all_shadows'][i] = row['shadow_s']-row['chosen_s']
        for name, delta in shifts.items():
            modified = times+delta
            all_changes[name+'_squared_gap_change_s2'] += geometry_change(times, modified, moment)['squared_gap_change_s2']
            all_changes[name+'_sum_headway_change_s'] += float(np.diff(np.sort(modified)).sum()-np.diff(times).sum())
    simultaneous = {}
    n, m, v = moment['samples'], moment['mean_s'], moment['variance_s2']
    for name in shifts:
        dm = all_changes[name+'_sum_headway_change_s']/n
        dq = all_changes[name+'_squared_gap_change_s2']/n
        simultaneous[name] = dict(cv_squared_change=(v+m*m+dq)/(m+dm)**2-1-moment['cv_squared'],
            mean_headway_change_s=dm, squared_gap_change_s2=all_changes[name+'_squared_gap_change_s2'])
    return {k: dict(v) for k, v in cohorts.items()}, simultaneous


def read_path(root, seed, day, method):
    item = seed, day, method, 0
    pair = json.loads((root/'pairs'/f'p{frozen.ITEMS.index(item)}.json').read_text())
    if tuple(pair[n] for n in ('seed', 'day', 'method', 'rep')) != item:
        raise AssertionError('V115 path identity differs from the consumed grid')
    name = f'{seed}_{day}_{method}_0'
    with np.load(root/'records'/(name+'.npz')) as record:
        fields = record['event_fields'].tolist()
        arrays = {kind: [dict(zip(fields, r.tolist())) for r in record[kind]]
                  for kind in ('arrivals', 'departures')}
        trips = len(record['launches'])
    decisions = None
    if method == frozen.PRIMARY:
        with gzip.open(root/'traces'/(name+'.jsonl.gz'), 'rt', encoding='ascii') as stream:
            trace = [json.loads(line) for line in stream]
        decisions = {key(r): r for r in trace}
        if len(trace) != len(decisions) or len(trace) != pair['accounting']['decisions']:
            raise AssertionError('duplicate or missing completed decisions')
    arrivals, departures, actions = coordinates(arrays['arrivals'], arrays['departures'], decisions)
    a, d = series(arrivals, ARRIVAL_PARTS), series(departures, DEPARTURE_PARTS)
    if not math.isclose(a['cv_squared'], pair['results']['cv_squared'], abs_tol=1e-10):
        raise AssertionError('actual arrival CV2 does not reproduce the V115 endpoint')
    if not math.isclose(sum(actions.values())/trips, pair['results']['holding_s_per_trip'], abs_tol=1e-6):
        raise AssertionError('completed action totals do not reproduce V115 holding')
    return dict(arrival=a, departure=d, actions=actions, decisions=decisions, departures=departures,
        uncontrolled_events=len(actions)-len(decisions) if decisions is not None else None,
        results=pair['results'])


def changed_predecessors(base, candidate):
    def neighbours(orders):
        return {(g, t): trips[i-1] if i else None for g, trips in orders.items() for i, t in enumerate(trips)}
    b, c = neighbours(base['orders']), neighbours(candidate['orders'])
    if b.keys() != c.keys():
        raise AssertionError('paired cleared paths have different headway event support')
    return sum(b[k] != c[k] for k in b)


def review(root, out):
    parent = json.loads((root/'summary/status.json').read_text())
    if parent['status'] != 'funded_quota_independent_pilot_complete':
        raise AssertionError('V116 requires the completed V115 pilot')
    cells, combined = [], defaultdict(Counter)
    for seed in frozen.SEEDS:
        for day in frozen.DAYS:
            b = read_path(root, seed, day, 'bilateral025')
            f = read_path(root, seed, day, frozen.PRIMARY)
            if b['actions'].keys() != f['actions'].keys():
                raise AssertionError('paired action support differs')
            cohorts, geometry = local_geometry(f['departures'], f['decisions'], f['departure'])
            for name, values in cohorts.items():
                combined[name].update(values)
            parts = Counter()
            for identity, row in f['decisions'].items():
                p = action_parts(row)
                parts.update(p)
                parts['between_path_shadow_s'] += row['shadow_s']-b['actions'][identity]
            actual = sum(f['actions'].values())-sum(b['actions'].values())
            np.testing.assert_allclose(sum(parts[n] for n in ('negative_pulse', 'positive_pulse', 'quota',
                'between_path_shadow_s')), actual, atol=1e-8, rtol=0)
            effects = dict(arrival=effect(b['arrival'], f['arrival'], ARRIVAL_PARTS),
                departure=effect(b['departure'], f['departure'], DEPARTURE_PARTS),
                commands=dict(parts, actual_holding_delta_s=actual),
                actual={m: f['results'][m]-b['results'][m] for m in
                    ('journey_min', 'cv_squared', 'readiness_delay_s', 'holding_s_per_trip')})
            effects.update({'local_restore_'+n: v for n, v in geometry.items()})
            cells.append(dict(seed=seed, day=day, effects=effects, cohorts=cohorts,
                uncontrolled_zero_holding_events=f['uncontrolled_events'],
                changed_arrival_predecessors=changed_predecessors(b['arrival'], f['arrival']),
                changed_departure_predecessors=changed_predecessors(b['departure'], f['departure'])))
            print('V116_CONSUMED_CELL_DONE', seed, day, flush=True)
    command_count = sum(c['cohorts']['negative']['decisions'] for c in cells)
    if sum(c['effects']['commands']['quarter_shadow'] for c in cells) != parent['accounting']['issued_s']:
        raise AssertionError('consumed trace coverage differs from all V115 authorization')
    frozen.pilot.write_json(out/'summary/status.json', dict(status='v115_readonly_quota_regularity_audit_complete',
        effects=inference.describe(cells, 'effects'), cohorts={k: dict(v) for k, v in combined.items()},
        actor_day=cells, actor_means={str(s): {kind: {m: float(np.mean([r['effects'][kind][m]
            for r in cells if r['seed'] == s])) for m in cells[0]['effects'][kind]} for kind in ('arrival', 'departure', 'actual')}
            for s in frozen.SEEDS}, day_means={str(d): {kind: {m: float(np.mean([r['effects'][kind][m]
            for r in cells if r['day'] == d])) for m in cells[0]['effects'][kind]} for kind in ('arrival', 'departure', 'actual')}
            for d in frozen.DAYS}, consumed_paths=2*len(cells), funded_decisions=parent['accounting']['decisions'],
        negative_response_cuts=command_count, new_rollouts=0, new_demand_days=0,
        actor_updates=0, critic_updates=0, dual_updates=0, main_policy_changed=False,
        limitations='Completed-neighbour geometry holds all other departure times fixed and is not an executed '
            'counterfactual. Joint shadow shifts include reordering and boundary normalization. Actual CV2 '
            'components are exact covariance accounting, not causal mediation. Pulse/quota components use '
            'quarter shadows on funded visited prefixes, not the separate bilateral states. Negative means '
            'the response increment -0.75*delta_R is negative. Two actors/four consumed days; no new validation.'))
    print('V116_READONLY_QUOTA_REGULARITY_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v115-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    a = p.parse_args()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v116_*.py', '-v'], check=True)
    review(a.v115_root, a.out_root)
