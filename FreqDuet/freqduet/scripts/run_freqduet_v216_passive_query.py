#!/usr/bin/env python3
"""V215 passive query capture/equivalence only; no interventions or future forks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import inspect
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import freqduet_v216_observation as obs
import preregister_freqduet_v215_positive_event as protocol
import run_freqduet_v211_joint_resources as prior


BASE = prior.prior.TradeoffAudit


def write(path, value):
    prior.pilot.write_json(path, value)


def exact(a, b, label):
    prior.pilot.joint.assert_same_state(a, b, 'V216_'+label)


def trace(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as f:
        json.dump(data, f, allow_nan=False, separators=(',', ':'))


class QueryAudit(BASE):
    def __init__(self, seed, capture):
        super().__init__(seed, 'bilateral025')
        self.capture = capture
        self.clock, self.observed_ready = obs.EventClock(), {}
        self.queries, self.fresh_queries = [], []
        self.policy_inputs = []
        self.commands, self.callbacks, self.dispatches, self.returns = [], [], [], []
        self.anchor, self.followup, self.anchor_query = None, None, None
        self.current_trip = None

    def attach(self, runner):
        super().attach(runner)
        env, planner = runner.env, runner.timetable_planner
        assert env.fleet_inventory_mode == 'fixed_pool' and runner.cfg['upper']['N_fleet'] == 12
        assert prior.pilot.step.ACTIONS.tolist() == list(protocol.ACTIONS)
        assert (planner is not None and runner.timetable_terminal_dispatch and planner.shared_directions
            and planner.terminal_schedule_mode == 'exact_headway_curve'
            and planner.headway_budget_mode == 'rolling_zero_sum_delta_v6'
            and planner.coefficient_parameterization == 'antisymmetric_linear_v5'
            and runner.upper_action_dim == 1 and np.all(runner.upper_action_low == -60.)
            and np.all(runner.upper_action_high == 60.))
        self.scalars = (float(runner.lower_trainer.alpha), float(runner.lower_trainer.lambda_param),
                        float(runner.upper_trainer.alpha))
        self.native_upper, self.native_action = runner._upper_callback_v2, runner.upper_trainer.policy_net.get_action
        self.native_launch = env.launch_bus
        from env.bus import Bus
        self.bus_type = Bus
        self.native_arrive, self.native_prepare = Bus.arrive_station, Bus._prepare_for_action

        def prepare(bus, current_time, *args, **kwargs):
            value = self.native_prepare(bus, current_time, *args, **kwargs)
            key = (int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction))
            stamp = self.clock.stamp('prepare', current_time, bus_id=key[0], trip_id=key[1],
                                     station_id=key[2], direction=key[3])
            self.observed_ready[key] = stamp
            return value

        def action(state, *args, **kwargs):
            value = self.native_action(state, *args, **kwargs)
            trip = self.current_trip
            assert trip is not None
            self.commands.append(dict(time_s=float(env.current_time), trip_id=int(trip.launch_turn),
                coefficient=np.asarray(value).reshape(-1).tolist(), state=np.asarray(state).tolist()))
            if self.capture:
                stamp = self.clock.stamp('fresh_upper', env.current_time, trip_id=int(trip.launch_turn),
                                         direction=bool(trip.direction))
                q = dict(obs.snapshot(env, stamp), trip_id=int(trip.launch_turn), direction=bool(trip.direction), fresh_upper=True,
                         native_coefficient=self.commands[-1]['coefficient'])
                self.fresh_queries.append(q)
                if self.anchor is not None and self.followup is None and protocol.upper_followup(self.anchor, q):
                    self.followup = q
            return value

        def upper(state, trip):
            self.current_trip = trip
            before, count = obs.plans(env), len(self.commands)
            try:
                value = self.native_upper(state, trip)
            finally:
                self.current_trip = None
            after = obs.plans(env)
            assert before.keys() == after.keys()
            self.callbacks.append(dict(time_s=float(env.current_time), trip_id=int(trip.launch_turn),
                direction=bool(trip.direction), target_s=float(value), commands=self.commands[count:],
                changes=[dict(trip_id=k, before=before[k], after=after[k]) for k in before if before[k] != after[k]]))
            self.clock.stamp('plan_update', env.current_time, trip_id=int(trip.launch_turn),
                             direction=bool(trip.direction))
            return value

        def launch(trip, actual_launch=None):
            ready = {int(b.bus_id) for b in env.bus_in_terminal if b.direction == bool(trip.direction)}
            bus = self.native_launch(trip, actual_launch=actual_launch)
            if bus is not None:
                assert int(bus.bus_id) in ready and len(env.bus_all) == 12
                self.dispatches.append(dict(trip=int(trip.launch_turn), bus=int(bus.bus_id),
                    direction=bool(trip.direction), actual_s=float(env.current_time),
                    public_baseline_s=float(getattr(trip, '_freqduet_base_launch',
                                                   getattr(trip, '_original_launch', trip.launch_time))),
                    eligible_s=float(trip._freqduet_first_eligible_time)))
                self.clock.stamp('launch', env.current_time, trip_id=int(trip.launch_turn), bus_id=int(bus.bus_id))
            return bus

        def arrive(bus, *args, **kwargs):
            active, trip, direction = bool(bus.on_route), int(bus.trip_id), bool(bus.direction)
            value = self.native_arrive(bus, *args, **kwargs)
            if active and not bus.on_route:
                start = self.launches[trip]
                account = prior.time_account(start['time_s'], float(bus.back_to_terminal_time),
                    float(bus.episode_hold_vehicle_seconds)-start['holding_counter_at_launch'])
                self.returns.append(dict(account, trip_id=trip, bus_id=int(bus.bus_id),
                                         inbound_direction=direction, target_direction=bool(bus.direction)))
                assert bool(bus.direction) != direction
                self.clock.stamp('return', env.current_time, trip_id=trip, bus_id=int(bus.bus_id))
            return value

        runner._upper_callback_v2, runner.upper_trainer.policy_net.get_action = upper, action
        env.launch_bus, Bus.arrive_station = launch, arrive
        if self.capture:
            Bus._prepare_for_action = prepare

    def detach(self):
        self.runner._upper_callback_v2 = self.native_upper
        self.runner.upper_trainer.policy_net.get_action = self.native_action
        self.env.launch_bus, self.bus_type.arrive_station = self.native_launch, self.native_arrive
        self.bus_type._prepare_for_action = self.native_prepare
        super().detach()

    def __call__(self, policy, states, features, theta):
        self.policy_inputs.append(states.numpy().tolist())
        law = super().__call__(policy, states, features, theta)
        if self.capture:
            runner, env = self.runner, self.env
            bus = runner._bus_for_agent(runner._pilot_agent)
            context = runner._pending_lower_action_context[int(runner._pilot_agent)]
            key = (int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction))
            prepared = self.observed_ready[key]
            stamp = self.clock.stamp('lower_query', env.current_time, bus_id=key[0], trip_id=key[1],
                                     station_id=key[2], direction=key[3])
            probabilities = law[0].numpy()
            assert np.count_nonzero(probabilities) == 1 and probabilities.max() == 1.
            base = int(probabilities.argmax())
            q = dict(obs.snapshot(env, stamp), bus_id=key[0], trip_id=key[1], station_id=key[2], direction=key[3],
                state_captured_s=prepared['time_s'], prepare_event_seq=prepared['event_seq'],
                base_bin=base, command_s=protocol.ACTIONS[base],
                feasible_bins=np.flatnonzero(policy.feasible_action_mask(states)[0].numpy()).tolist(),
                forward_valid=bool(context.evidence_valid), follower_valid=bool(context.follower_evidence_valid),
                terminal_masked=bool(runner._lower_terminal_action_masked(bus) or bus.last_station.station_type == 0),
                context=asdict(context), onboard=len(bus.passengers),
                executed_holding_before_s=float(bus.episode_hold_vehicle_seconds)
                    -self.launches[key[1]]['holding_counter_at_launch'])
            self.queries.append(q)
            if self.anchor is None:
                self.anchor = protocol.lower_anchor(q)
                if self.anchor is not None:
                    self.anchor_query = q
        return law

    def finish(self, runner, records, holds):
        env = runner.env
        active = []
        for b in env.bus_all:
            if b.on_route:
                start = self.launches[int(b.trip_id)]
                account = prior.time_account(start['time_s'], float(env.current_time),
                    float(b.episode_hold_vehicle_seconds)-start['holding_counter_at_launch'])
                account['observed_until_s'] = account.pop('return_s')
                active.append(dict(account, trip_id=int(b.trip_id), bus_id=int(b.bus_id)))
        assert len(env.bus_all) == 12 and len(self.dispatches) == len(self.launches)
        assert {r['trip_id'] for r in self.returns} | {r['trip_id'] for r in active} == set(self.launches)
        assert len(self.returns)+len(active) == len(self.launches)
        self.ledger = dict(physical_vehicles=12, completed_trips=len(self.returns), incomplete_trips=len(active),
            censored_trips=active, **prior.holding_account(self.returns, active, sum(h['action_s'] for h in holds), not active))
        assert self.scalars == (float(runner.lower_trainer.alpha), float(runner.lower_trainer.lambda_param),
                                float(runner.upper_trainer.alpha))
        self.contract = prior.prior.factorial.prior.physical_contract(env, runner.cfg)
        self.dispatch_metrics = prior.prior.factorial.prior.dispatch_metrics(env)
        self.actions = [dict(r, features=r['features'].tolist(), probabilities=r['probabilities'].tolist(),
                             context=asdict(r['context'])) for r in records]
        if self.capture:
            self.validation = obs.validate_capture(self.queries, self.fresh_queries, self.clock)
            assert len(records) == len(self.queries)
            for q, r in zip(self.queries, records):
                assert (q['trip_id'], q['station_id'], q['direction'], q['time_s'], q['command_s']) == (
                    r['trip_id'], r['station_id'], r['direction'], r['time_s'], protocol.ACTIONS[r['action']])
            self.supply = obs.supply_label(self.anchor, self.returns, self.dispatches,
                                           self.clock.events, self.departures, self.anchor_query)
        prior.prior.frozen.source.experiment.LogitCross.finish(self, runner, records, holds)


def replay(root, context, capture):
    seed, day = protocol.CONTEXTS[context]
    sampler = 187000000+seed+100*day
    rule, credit = QueryAudit(seed, capture), prior.pilot.CreditObserver()
    label = f'v216_c{context}_capture{int(capture)}'
    row, events, _, holds = prior.pilot.rollout(root, seed, day, 'stochastic', sampler, [0.]*4, label,
        probability_rule=rule, trace_observer=rule, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        holding_person_s=rule.holding_person_s, public_launch_delay_total_s=sum(
            max(0., r['actual_s']-r['public_baseline_s']) for r in rule.dispatches),
        readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in rule.dispatches), **rule.dispatch_metrics)
    result = dict(results=row, ledger=rule.ledger, physical_contract=rule.contract)
    common = dict(lower_actions=rule.actions, lower_policy_inputs=rule.policy_inputs,
        upper_commands=rule.commands, upper_callbacks=rule.callbacks,
        issued_holding=holds, actual_launches=rule.dispatches, vehicle_returns=rule.returns,
        arrivals=rule.arrivals, departures=rule.departures,
        action_uniforms=[[list(k), v] for k, v in events.items()])
    trace(root/'records'/f'{label}.json.gz', dict(result=result, trace=common,
        capture=None if not capture else dict(queries=rule.queries, fresh_upper=rule.fresh_queries,
            events=rule.clock.events, anchor=rule.anchor, followup=rule.followup, supply=rule.supply)))
    return result, common, rule


def pair_summary(context, native, observed, state):
    audit = observed[2]
    window = [q for q in audit.queries if 7200 <= q['time_s'] < 8100]
    positive = [q for q in window if q['base_bin'] > 0 and q['forward_valid'] and q['follower_valid']
                and not q['terminal_masked'] and any(i < q['base_bin'] for i in q['feasible_bins'])]
    return dict(state, status='v216_passive_pair_exact', seed=protocol.CONTEXTS[context][0],
        day=protocol.CONTEXTS[context][1], all_endpoints_and_physical_ledgers_exact=True,
        full_actions_states_plans_and_timing_exact=True, **audit.validation,
        window_queries=len(window), positive_valid_queries=len(positive),
        queue_supported_queries=sum(protocol.lower_anchor(q) is not None for q in window),
        anchor=audit.anchor, native_fresh_followup=None if audit.followup is None else
            {k: audit.followup[k] for k in ('time_s', 'event_seq', 'trip_id', 'direction', 'native_coefficient')},
        supply=audit.supply, results={k: native[0]['results'][k] for k in
            (*protocol.CONSTRAINTS, 'passenger_person_s', 'journey_min', 'holding_s_per_trip')},
        main_policy_changed=False, future_banks_opened=False)


def run_pair(root, context):
    progress = root/'progress'/f'c{context}.json'
    assert not progress.exists(), 'inspect original attempt;no automatic reruns'
    state = dict(context=context, references_started=0, smoke_attempts_started=0,
                 rollouts_completed=0, intervention_paths=0, training_updates=0)
    write(progress, state)
    try:
        state['references_started'] = 1
        write(progress, state)
        native = replay(root, context, False)
        state.update(rollouts_completed=1, smoke_attempts_started=1)
        write(progress, state)
        observed = replay(root, context, True)
        state['rollouts_completed'] = 2
        write(progress, state)
        exact(native[0], observed[0], 'all_endpoints_ledgers_contract')
        exact(native[1], observed[1], 'full_actions_states_plans_events_and_physical_timing')
        write(root/'summary'/f'c{context}.json', pair_summary(context, native, observed, state))
        print('V216_PASSIVE_PAIR_EXACT', context, flush=True)
    except Exception as error:
        state.update(status='v216_passive_pair_failed', failure=str(error))
        write(progress, state)
        raise


def preflight_checks(root, registered):
    sealed = prior.read(registered/'protocol.json')
    expected = protocol.registration('033350c890')
    exact({k: sealed[k] for k in expected}, expected, 'sealed_V215_contract')
    prior.pilot.joint.optimized_runtime(root)
    from env.bus import Bus
    assert 'current_time' in inspect.signature(Bus._prepare_for_action).parameters
    assert 'current_time' in inspect.signature(Bus.arrive_station).parameters
    for pattern in ('test_v215_positive_event.py', 'test_v216_observation.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)


def preflight(root, registered):
    assert not (root/'protocol.json').exists(), 'V216 already preregistered'
    preflight_checks(root, registered)
    write(root/'protocol.json', dict(stage='v216_V215_passive_capture_only',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], parent=str(registered/'protocol.json'),
        contexts=[list(c) for c in protocol.CONTEXTS], max_paths=8,
        budget='4 native references+4 passive-on smoke attempts from V215;2 smoke attempts remain. '
            'Incomplete attempts count;no automatic retry/amendment.',
        experiment='Native reference with existing light physical/action recorder versus same deployment '
            'plus current-query inventory snapshots,prepare timestamps and unified native event order. '
            'Full endpoint,holding,lower probabilities/states,upper commands/plans,uniforms,arrivals,departures, '
            'returns and dispatch equality required. No controller inputs or action changes.',
        label='FIRST frozen V215 positive anchor and FIRST qualifying native fresh follow-up only. '
            'After completion label that same inbound bus/trip return and its FIRST subsequent physical '
            'launch;match original due IDs. Never search later launches,replace points or use future labels as inputs.',
        future_banks_opened=False, intervention_paths=0, training_updates=0, main_policy_changed=False))
    write(root/'preflight/done.json', dict(status='v216_preflight_passed',
        source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], new_rollouts=0, main_policy_changed=False))
    print('V216_PREFLIGHT_PASSED_NO_ROLLOUTS', flush=True)


def aggregate(root, smoke_used=4, smoke_cap=6, new_rollouts=8):
    rows = [prior.read(root/'summary'/f'c{i}.json') for i in range(4)]
    assert all(r['all_endpoints_and_physical_ledgers_exact'] and r['full_actions_states_plans_and_timing_exact'] for r in rows)
    supported = all(r['anchor'] is not None for r in rows)
    followups = all(r['native_fresh_followup'] is not None for r in rows)
    same_supply = all(r['supply'].get('same_vehicle_due_supply', False) for r in rows)
    write(root/'summary/status.json', dict(status='v216_passive_capture_complete', contexts=rows,
        references_used=4, smoke_attempts_used=smoke_used, smoke_attempts_remaining=smoke_cap-smoke_used,
        all_action_timing_endpoint_equivalence_passed=True, all_contexts_have_positive_queue_anchor=supported,
        all_contexts_have_native_upper_followup=followups, all_anchors_observed_same_vehicle_due_supply=same_supply,
        all_anchor_holding_executions_verified=all(r['supply'].get('anchor_holding_execution_matches', False) for r in rows),
        all_service_cleared=all(r['results']['trips_unlaunched'] == r['results']['trips_incomplete'] == 0 for r in rows),
        observation_only=True, causal_control_benefit_tested=False, intervention_matrix_admitted=False,
        new_rollouts=new_rollouts, training_updates=0, future_banks_opened=False, main_policy_changed=False,
        next_decision='Stop opportunity pilot if any anchor missing;otherwise inspect native fresh follow-up '
            'and physical supply labels before separately authorizing any intervention smoke. No point replacement.'))
    print('V216_PASSIVE_CAPTURE_COMPLETE', supported, followups, same_supply, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'capture'))
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--registered-root', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'preflight':
        preflight(a.out_root, a.registered_root)
    else:
        assert prior.read(a.out_root/'preflight/done.json')['status'] == 'v216_preflight_passed'
        assert not (a.out_root/'capture_started.json').exists(), 'capture already attempted;inspect saved progress'
        write(a.out_root/'capture_started.json', dict(contexts=4, paths=8, intervention_paths=0))
        with ProcessPoolExecutor(max_workers=4, mp_context=get_context('spawn')) as pool:
            futures = [pool.submit(run_pair, a.out_root, i) for i in range(4)]
            for future in futures:
                future.result()
        aggregate(a.out_root)


if __name__ == '__main__':
    main()
