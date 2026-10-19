#!/usr/bin/env python3
"""Passive same-vehicle turnover labels on eight consumed bilateral paths."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_freqduet_v124_eta_factorial as frozen
import audit_freqduet_v125_timing_propagation as timing


DAYS = (2231001, 2231004)
ITEMS = list(itertools.product(frozen.prior.SEEDS, DAYS))
METHODS = (frozen.prior.BILATERAL, frozen.PRIMARY)
SMOKE_INDEX = 1
PARTS = ('eligibility', 'selected_vehicle_wait', 'post_available_wait')


def dispatch_parts(eligible, actual, visible):
    wait = 0. if visible is None else max(visible-eligible, 0.)
    residual = actual-eligible-wait
    if residual < 0:
        raise AssertionError('selected vehicle became dispatch-visible after its launch')
    return [eligible, wait, residual]


class TurnoverAudit(frozen.BilateralAudit):
    def __init__(self, seed, geometry, arm):
        super().__init__(seed, geometry, arm)
        self.returns, self.pending_visible, self.attempts, self.dispatches = {}, [], {}, []

    def attach(self, runner):
        super().attach(runner)
        self.turn_step, self.turn_launch = self.env.step, self.env.launch_bus
        self.turn_drive = self.bus_type.drive

        def step(*args, **kwargs):
            for event in self.pending_visible:
                event['first_dispatch_visible_s'] = float(self.env.current_time)
            self.pending_visible.clear()
            return self.turn_step(*args, **kwargs)

        def drive(bus, *args, **kwargs):
            active = bool(bus.on_route)
            result = self.turn_drive(bus, *args, **kwargs)
            if active and not bus.on_route:
                trip = int(bus.trip_id)
                launch = self.launches[trip]
                held = float(bus.episode_hold_vehicle_seconds)-launch['holding_counter_at_launch']
                event = dict(trip_id=trip, bus_id=int(bus.bus_id),
                    available_direction=bool(bus.direction), return_s=float(bus.back_to_terminal_time),
                    first_dispatch_visible_s=None, launch_s=launch['time_s'], holding_s=held,
                    nonholding_elapsed_s=float(bus.back_to_terminal_time)-launch['time_s']-held)
                if trip in self.returns or bus.last_completed_trip_id != trip:
                    raise AssertionError('terminal return is not once-only on its completed vehicle trip')
                self.returns[trip] = event
                self.pending_visible.append(event)
            return result

        def launch(trip, actual_launch=None):
            ready = {int(bus.bus_id): dict(prior_trip_id=int(bus.trip_id),
                return_s=None if not np.isfinite(bus.back_to_terminal_time) else float(bus.back_to_terminal_time))
                for bus in self.env.bus_in_terminal if bool(bus.direction) == bool(trip.direction)}
            identity = int(trip.launch_turn)
            attempt = self.attempts.setdefault(identity, dict(first_attempt_s=float(self.env.current_time), denied=0))
            bus = self.turn_launch(trip, actual_launch=actual_launch)
            if bus is None:
                attempt['denied'] += 1
                return bus
            old = ready[int(bus.bus_id)]
            returned = None if old['prior_trip_id'] < 0 else self.returns[old['prior_trip_id']]
            visible = None if returned is None else returned['first_dispatch_visible_s']
            if returned is not None and (visible is None or returned['bus_id'] != int(bus.bus_id)
                    or returned['return_s'] != old['return_s']):
                raise AssertionError('launch is not linked to the selected vehicle actual return')
            eligible, actual = float(trip._freqduet_first_eligible_time), float(self.env.current_time)
            self.dispatches.append(dict(trip_id=identity, bus_id=int(bus.bus_id),
                direction=bool(trip.direction), prior_trip_id=old['prior_trip_id'], return_s=old['return_s'],
                first_dispatch_visible_s=visible, eligible_s=eligible, actual_s=actual,
                coordinates=dispatch_parts(eligible, actual, visible),
                ready_before=len(ready), denied_attempts=attempt['denied'],
                first_attempt_s=attempt['first_attempt_s']))
            return bus

        self.env.step, self.env.launch_bus, self.bus_type.drive = step, launch, drive

    def detach(self):
        self.env.step, self.env.launch_bus, self.bus_type.drive = self.turn_step, self.turn_launch, self.turn_drive
        super().detach()

    def finish(self, runner, records, holds):
        super().finish(runner, records, holds)
        if len(self.dispatches) != len(self.launches) or len(self.returns) != len(self.launches):
            raise AssertionError('cleared-service turnover labels do not cover every actual trip')
        for row in self.dispatches:
            np.testing.assert_allclose(sum(row['coordinates']), row['actual_s'], atol=0, rtol=0)
            np.testing.assert_allclose(row['actual_s']-row['eligible_s'],
                self.launches[row['trip_id']]['readiness_s'], atol=0, rtol=0)
            if row['first_attempt_s'] != row['eligible_s']:
                raise AssertionError('first fleet attempt differs from sticky eligibility')
        self.turn_step = self.turn_launch = self.turn_drive = None


def run_path(root, cached, runtime, seed, day, arm):
    rule = TurnoverAudit(seed, frozen.prior.model.load_geometry(runtime), arm)
    credit = frozen.pilot.CreditObserver()
    result, _, _, _ = frozen.pilot.rollout(root, seed, day, 'stochastic', 187000000+seed+100*day,
        [0.]*4, f'v127_{seed}_{day}_{arm}', probability_rule=rule, trace_observer=rule, credit_observer=credit)
    result.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s, **rule.dispatch)
    actual = dict(seed=seed, day=day, method=arm, rep=0, results=result, contract=rule.contract,
        coverage=rule.coverage, feedforward=rule.feedforward, accounting=rule.accounting,
        prediction_use=rule.prediction_use)
    name = f'{seed}_{day}_{arm}'
    # Open consumed future records only after the live replay has completed.
    old = json.loads((cached/'pairs'/(name+'.json')).read_text())
    frozen.pilot.joint.assert_same_state(old, {k: actual[k] for k in old}, 'V127_all_consumed_endpoints')
    current = root/'records'/(name+'_0.npz')
    rule.save(current)
    with np.load(cached/'records'/(name+'_0.npz')) as before, np.load(current) as after:
        if before.files != after.files:
            raise AssertionError('turnover observation changed timing-record fields')
        for field in before.files:
            np.testing.assert_array_equal(before[field], after[field])
        fields = before['event_fields'].tolist()
        ends = {timing.elapsed.key(dict(zip(fields, r.tolist()))): r[fields.index('realized_holding')]
            for r in before['departures']}
        for row in before['arrivals']:
            key = timing.elapsed.key(dict(zip(fields, row.tolist())))
            ends[key] -= row[fields.index('realized_holding')]
    for decision in rule.decisions:
        if decision['chosen_s'] != ends[timing.elapsed.key(decision)]:
            raise AssertionError('requested holding differs from the original executed stop action')
    if arm == frozen.PRIMARY:
        old_trace = frozen.prior.prior.read_trace(cached/'traces'/(name+'_0.jsonl.gz'))
        frozen.prior.prior.assert_decisions(old_trace, [dict(r, evidence=rule.evidence[timing.elapsed.key(r)])
            for r in rule.decisions])
    frozen.pilot.write_json(root/'turnover'/(name+'.json'), dict(dispatches=rule.dispatches,
        returns=list(rule.returns.values())))
    return dict(method=arm, endpoints_exact=True, full_timing_arrays_exact=True,
        holding_commands_exact=True, dispatches=len(rule.dispatches), returns=len(rule.returns),
        last_returns_without_next_dispatch=sum(r['first_dispatch_visible_s'] is None for r in rule.returns.values()),
        values=dict({k: result[k] for k in frozen.prior.METRICS}, **rule.dispatch))


def series(rows):
    points = {(r['trip_id'], 0, r['direction']): dict(time_s=r['actual_s'],
        coordinates=np.asarray(r['coordinates'])) for r in rows}
    edges = timing.edge_rows(points)
    return edges, timing.moment(edges)


def paired_details(base, candidate):
    bd, cd = [{r['trip_id']: r for r in path['dispatches']} for path in (base, candidate)]
    br, cr = [{r['trip_id']: r for r in path['returns']} for path in (base, candidate)]
    matched, unmatched = {}, []
    for trip, c in cd.items():
        b = bd[trip]
        if (b['bus_id'], b['prior_trip_id']) != (c['bus_id'], c['prior_trip_id']):
            unmatched.append(trip)
            continue
        entry = dict(trip_id=trip, direction=c['direction'], prior_trip_id=c['prior_trip_id'],
            base=b, candidate=c, delta_actual_s=c['actual_s']-b['actual_s'])
        prior = c['prior_trip_id']
        if prior >= 0:
            before, after = br[prior], cr[prior]
            deltas = {k: after[k]-before[k] for k in ('return_s', 'launch_s', 'holding_s', 'nonholding_elapsed_s')}
            np.testing.assert_allclose(deltas['return_s'], sum(deltas[k] for k in
                ('launch_s', 'holding_s', 'nonholding_elapsed_s')), atol=0, rtol=0)
            entry['prior_return_delta'] = deltas
            entry['availability_state'] = ('bound' if b['coordinates'][1] else 'free')+'_to_'+(
                'bound' if c['coordinates'][1] else 'free')
        else:
            entry['availability_state'] = 'initial_pool'
        matched[trip] = entry
    be, bm = series(base['dispatches'])
    ce, cm = series(candidate['dispatches'])
    matrix = (cm['covariance']-bm['covariance'])/bm['mean']**2
    parts = {name+'_allocated': float(matrix[i].sum()) for i, name in enumerate(PARTS)}
    parts.update(cv_squared=cm['cv_squared']-bm['cv_squared'],
        mean_normalization=cm['variance']*(1/cm['mean']**2-1/bm['mean']**2))
    np.testing.assert_allclose(sum(parts[n+'_allocated'] for n in PARTS)+parts['mean_normalization'],
        parts['cv_squared'], atol=1e-10, rtol=0)
    old_edges = {r['key']: r for r in be}
    changes = []
    for row in ce:
        old = old_edges[row['key']]
        if old['predecessor'] != row['predecessor']:
            raise AssertionError('consumed bilateral pair changed its actual dispatch predecessor')
        contribution = timing.contribution([row], cm)-timing.contribution([old], bm)
        changes.append(dict(trip_id=row['key'][0], predecessor=old['predecessor'][0],
            base_gap_s=old['headway'], candidate_gap_s=row['headway'], cv2_contribution=contribution))
    np.testing.assert_allclose(sum(r['cv2_contribution'] for r in changes), parts['cv_squared'], atol=1e-10, rtol=0)
    ordered = sorted(changes, key=lambda r: r['cv2_contribution'])
    selected = ordered[:3]+ordered[-3:]
    groups = {}
    for label in ('bound_to_bound', 'bound_to_free', 'free_to_bound', 'free_to_free', 'initial_pool'):
        chosen = [r for r in matched.values() if r['availability_state'] == label]
        groups[label] = dict(n=len(chosen), mean_launch_delta_s=None if not chosen else
            float(np.mean([r['delta_actual_s'] for r in chosen])))
    return dict(dispatch_effects=parts, availability_groups=groups, unmatched_vehicle_or_prior_trip=unmatched,
        trip16=matched.get(16), localized_intervals=[dict(r, current=matched.get(r['trip_id']),
            preceding=matched.get(r['predecessor'])) for r in selected])


def run_cell(root, v123, v124, runtime, index):
    receipt = root/'pairs'/f'p{index}.json'
    if receipt.exists():
        return
    seed, day = ITEMS[index]
    paths = [run_path(root, v124 if arm == frozen.PRIMARY else v123, runtime, seed, day, arm) for arm in METHODS]
    raw = [json.loads((root/'turnover'/f'{seed}_{day}_{arm}.json').read_text()) for arm in METHODS]
    details = paired_details(*raw)
    np.testing.assert_allclose(details['dispatch_effects']['cv_squared'],
        paths[1]['values']['dispatch_cv_squared']-paths[0]['values']['dispatch_cv_squared'], atol=1e-10, rtol=0)
    frozen.pilot.write_json(receipt, dict(seed=seed, day=day, index=index, paths=paths, **details))
    print('V127_EXACT_TURNOVER_CELL_DONE', index, seed, day, flush=True)


def aggregate(root):
    cells = [json.loads((root/'pairs'/f'p{i}.json').read_text()) for i in range(len(ITEMS))]
    if [(r['seed'], r['day']) for r in cells] != ITEMS:
        raise AssertionError('turnover replay requires the four registered consumed actor/day cells')
    frozen.pilot.write_json(root/'summary/status.json', dict(status='v127_passive_turnover_replay_complete',
        actor_day=cells, replayed_paths=8, new_demand_days=0, actor_updates=0, critic_updates=0,
        dual_updates=0, main_policy_changed=False, all_endpoints_and_commands_exact=True,
        limitations='Consumed success/failing days, not independent confirmation or causal mediation. '
            'Selected-vehicle availability does not identify the first possible fleet-wide release. '
            'Post-availability residual includes release checks and timetable iteration, not a pure upper action. '
            'Initial stock has no observed prior return; final returns without a next dispatch phase remain null.'))
    print('V127_TURNOVER_REPLAY_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('smoke', 'run', 'aggregate'))
    for name in ('out-root', 'v123-root', 'v124-root', 'runtime-root'):
        p.add_argument('--'+name, type=Path, required=name == 'out-root')
    p.add_argument('--index', type=int)
    a = p.parse_args()
    if a.command == 'smoke':
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v127_*.py', '-v'], check=True)
        run_cell(a.out_root, a.v123_root, a.v124_root, a.runtime_root, SMOKE_INDEX)
        frozen.pilot.write_json(a.out_root/'smoke/done.json', dict(status='v127_failed_day_exact_pair_passed',
            index=SMOKE_INDEX, replayed_paths=2))
    elif a.command == 'run':
        if not (a.out_root/'smoke/done.json').exists():
            raise AssertionError('remaining consumed paths require exact failed-day smoke')
        run_cell(a.out_root, a.v123_root, a.v124_root, a.runtime_root, a.index)
        frozen.pilot.write_json(a.out_root/'shards'/str(a.index)/'done.json', dict(status='complete', index=a.index))
    else:
        aggregate(a.out_root)
