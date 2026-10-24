"""V253/V254 one-query seven-bin adapter; no rollout or online action selector."""

import copy

import numpy as np
import torch

from preregister_freqduet_v246_compensation_supply import ACTIONS
from seal_freqduet_v254_action_manifest import SMOKE_ROOT


STREAMS = ('fleet', 'upper_residual_selector', 'headway_value_selector',
           'terminal_value_selector', 'fixed_expert_selector', 'tpc_mixture', 'reachability_replay')


def smoke_plan(native_bin, feasible):
    assert native_bin in feasible and len(feasible) == len(set(feasible)) >= 2
    assert all(b in range(7) for b in feasible)
    return dict(physical=['native','forced_native']+[f'b{b}' for b in range(7)
                if b in feasible and b != native_bin],
        aliases={f'b{native_bin}': 'native'},
        masked=[f'b{b}' for b in range(7) if b not in feasible], max_starts=8)


def prefix_snapshot(a, runner, policy, states, features, theta, native, uniforms, repeated, joint):
    def passenger(p):
        return (p.appear_time, p.boarding_time, p.arrive_time, p.appear_station.station_id,
                p.destination_station.station_id, p.boarded, p.arrived)
    env = runner.env
    buses = [dict(bus_id=b.bus_id, trip_id=b.trip_id, direction=b.direction, on_route=b.on_route,
        state=b.state.name, speed=b.current_speed, last_station=b.last_station.station_id,
        next_station=b.next_station.station_id, travel_distance=b.travel_distance,
        absolute_distance=b.absolute_distance, last_station_dis=b.last_station_dis,
        next_station_dis=b.next_station_dis, remaining_dwell=b.holding_time,
        active_dwell=b.dwelling_time, holding_counter=b.episode_hold_vehicle_seconds,
        passengers=[passenger(p) for p in b.passengers]) for b in env.bus_all]
    assert len(buses) == 12
    return copy.deepcopy(dict(query=a.pulse_capture.queries[-1], lower_state=states,
        lower_features=features, theta=theta, native_action=native,
        physical=repeated.physical_prefix(env), buses=buses,
        waiting=[[passenger(p) for p in s.waiting_passengers] for s in env.stations],
        ready_order=[b.bus_id for b in env.bus_in_terminal],
        resources=joint.resource_snapshot(env, a.launches),
        native_plan_caches=runner._active_timetable_plans,
        pending_lower_context=runner._pending_lower_action_context,
        legacy=dict(events=a.clock.events, commands=a.commands, callbacks=a.callbacks,
            queries=a.queries, inputs=a.policy_inputs, dispatches=a.dispatches,
            returns=a.returns, arrivals=a.arrivals, departures=a.departures),
        v224=a.handoff.data(), v230=a.resource.data(False), v235=a.cycle_capture.data(False),
        v241=a.request_capture.data(False), v248=a.prepare_timing_capture.data(False),
        pulse=a.pulse_capture.data(False), prior_requests=a.requests,
        rng=joint.pilot.joint.rng_state(runner, torch),
        lower_sampling=policy.sampling_state(),
        upper_sampling=runner.upper_trainer.policy_net.sampling_state(),
        scenario=dict(seed=env.scenario_tape.seed, version=env.scenario_tape.version,
            streams={k:r.get_state() for k,r in env.scenario_tape._streams.items()}),
        event_uniforms=dict(seed=uniforms.seed, counts=dict(uniforms.counts), records=uniforms.records)))


def checked_snapshot(a, runner, policy, states, features, theta, native, uniforms, repeated, joint):
    same = joint.pilot.joint.assert_same_state
    before = joint.pilot.joint.rng_state(runner, torch)
    calls = a.handoff.calls.copy()
    result = prefix_snapshot(a, runner, policy, states, features, theta, native, uniforms, repeated, joint)
    same(before, joint.pilot.joint.rng_state(runner, torch), 'V257_snapshot_rng_unchanged')
    assert calls == a.handoff.calls, 'snapshot invoked native computation'
    return result


def execution_chain(requests, actions, holds, departures):
    """Match native query occurrences to samples,issued commands and measured H increments."""
    assert len(requests) == len(actions)
    issued, left, cycles = {}, {}, {}
    for h in holds:
        key = tuple(h[k] for k in ('time_s','bus_id','trip_id','station_id'))
        issued.setdefault(key, []).append(h)
    for d in departures:
        identity = d['identity']
        key = tuple(identity[k] for k in ('bus_id','trip_id','station_id','direction'))
        left.setdefault(key, []).append(d)
    for i,q in enumerate(requests):
        cycles.setdefault(tuple(q['cycle_key']), []).append(i)
    next_query = {a:b for ids in cycles.values() for a,b in zip(ids, ids[1:])}
    rows = []
    for i,(q,action) in enumerate(zip(requests, actions)):
        p, clock, chosen = q['identity'], q['clock'], q['selected_bin']
        assert chosen in q['feasible_bins']
        assert q['is_anchor'] or chosen == q['native_bin'], 'later feedback was overridden'
        assert tuple(action[k] for k in ('time_s','trip_id','station_id','direction')) == (
            clock['query_s'], p['trip_id'], p['station_id'], p['direction'])
        assert action['action'] == chosen and np.array_equal(action['probabilities'], np.eye(7)[chosen])
        key = (clock['query_s'],p['bus_id'],p['trip_id'],p['station_id'])
        commands = issued.get(key, [])
        assert len(commands) == 1 and commands[0]['action_s'] == ACTIONS[chosen]
        ends = []
        for d in left.get(tuple(p[k] for k in ('bus_id','trip_id','station_id','direction')), []):
            if d['departure_order'] > clock['query_order']:
                ends.append((d['departure_order'],d['completed']['timing']['realized_holding'],'departure'))
        if i in next_query:
            later = requests[next_query[i]]['clock']
            ends.append((later['query_order'],later['query_holding_s'],'next_same_cycle_query'))
        actual, status = None, 'missing_execution_endpoint'
        if ends:
            _, counter, status = min(ends, key=lambda r:r[0])
            actual = counter-clock['query_holding_s']
            assert 0 <= actual <= ACTIONS[chosen], 'holding attribution crossed another command'
            if actual != ACTIONS[chosen]:
                status = 'incomplete_before_'+status
        rows.append(dict(query_index=i, issued_s=ACTIONS[chosen], executed_s=actual,
            status=status, complete=actual == ACTIONS[chosen]))
    anchors = [i for i,q in enumerate(requests) if q['is_anchor']]
    assert len(anchors) == 1
    i = anchors[0]
    return dict(rows=rows, complete=all(r['complete'] for r in rows),
        sampled_actions_exact=True, issued_commands_exact=True,
        anchor_executed_s=rows[i]['executed_s'],
        changed_requests=sum(q['selected_bin'] != q['native_bin'] for q in requests))


def atomic_audit(base, repeated, event, joint):
    same = joint.pilot.joint.assert_same_state

    class AtomicAudit(base):
        def __init__(self, seed, branch, uniforms, reference=None):
            assert seed == 12037 and isinstance(uniforms, event.EventUniforms)
            assert uniforms.seed == 402112137
            assert branch == 'native' or reference is not None, 'native prefix must be collected first'
            if reference is not None:
                q = reference['anchor']['query']
                plan = smoke_plan(q['native_bin'], q['feasible_bins'])
                assert branch in plan['physical'], 'masked bins and native-bin aliases are unstarted'
            else:
                assert branch == 'native'
            super().__init__(seed, 'v107')
            self.branch, self.uniforms, self.reference = branch, uniforms, reference
            self.fork_anchor, self.future_receipt, self.requests = None, None, []

        def attach(self, runner):
            super().attach(runner)
            assert tuple(runner._runtime_numpy_streams()) == STREAMS
            self.atomic_runner = runner

        def rollout_kwargs(self, credit_observer):
            return dict(probability_rule=self, trace_observer=self,
                        event_uniforms=self.uniforms, credit_observer=credit_observer)

        def __call__(self, policy, states, features, theta):
            native = super().__call__(policy, states, features, theta)
            row = self.pulse_capture.queries[-1]
            is_anchor = self.fork_anchor is None and row['eligible']
            selected = row['native_bin']
            if is_anchor:
                assert self.pulse_capture.anchor['query'] == row, 'not the frozen first query'
                assert row['query_index'] == len(self.queries)-1
                chosen = row['native_bin'] if self.branch in ('native','forced_native') else int(self.branch[1:])
                assert chosen in row['feasible_bins']
                anchor = checked_snapshot(self, self.atomic_runner, policy, states, features, theta,
                                         native, self.uniforms, repeated, joint)
                if self.reference is not None:
                    same(anchor, self.reference['anchor'], 'V257_common_precommand_prefix')
                before_native = native.clone()
                startup = repeated.fork_future(self.atomic_runner, policy, SMOKE_ROOT)
                self.uniforms.seed = SMOKE_ROOT+10000000
                after = checked_snapshot(self, self.atomic_runner, policy, states, features, theta,
                                        native, self.uniforms, repeated, joint)
                for key in anchor.keys()-{'rng','lower_sampling','upper_sampling','scenario','event_uniforms'}:
                    same(anchor[key], after[key], 'V257_preserved_'+key)
                same(anchor['event_uniforms']['counts'], dict(self.uniforms.counts), 'V257_kept_occurrences')
                same(anchor['event_uniforms']['records'], self.uniforms.records, 'V257_kept_prefix_uniforms')
                assert torch.equal(native, before_native)
                receipt = dict(startup=startup, root=SMOKE_ROOT, event_seed=self.uniforms.seed,
                               event_counts=dict(self.uniforms.counts))
                if self.reference is not None:
                    same(receipt, self.reference['future'], 'V257_common_future_start')
                self.fork_anchor, self.future_receipt = anchor, receipt
                selected = chosen
            value = native
            if selected != row['native_bin']:
                value = torch.zeros_like(native)
                value[0, selected] = 1.
                self.queries[-1]['command_s'] = ACTIONS[selected]
            self.requests.append(dict(copy.deepcopy(row), selected_bin=selected,
                is_anchor=is_anchor, explicit_native_alias=is_anchor and self.branch == 'forced_native'))
            return value

        def finish(self, runner, records, holds):
            assert self.fork_anchor is not None, 'registered first anchor absent; no replacement'
            super().finish(runner, records, holds)
            self.atomic_execution = execution_chain(self.requests, self.actions, holds,
                                                     self.prepare_timing_capture.departures)
            self.atomic_holds = copy.deepcopy(holds)

        def branch_data(self, results):
            assert self.fork_anchor is not None
            cycles = self.cycle_capture.data(True)
            anchor_cycle = self.fork_anchor['query']['cycle_key']
            matched = [c for c in cycles['cycles'] if c['cycle_key'] == anchor_cycle]
            assert len(matched) == 1 and matched[0]['executed_holding_s'] is not None
            total = sum(c['executed_holding_s'] for c in cycles['cycles'])
            assert total == self.ledger['executed_holding_s']
            return copy.deepcopy(dict(branch=self.branch, anchor=self.fork_anchor,
                future=self.future_receipt, requests=self.requests, uniforms=self.uniforms.records,
                execution=self.atomic_execution, actions=self.actions, holds=self.atomic_holds,
                payloads=dict(legacy=dict(events=self.clock.events, commands=self.commands,
                    callbacks=self.callbacks, queries=self.queries, inputs=self.policy_inputs,
                    dispatches=self.dispatches, returns=self.returns, arrivals=self.arrivals,
                    departures=self.departures), v224=self.handoff.data(), v230=self.resource.data(True),
                    v235=cycles, v241=self.request_capture.data(True),
                    v248=self.prepare_timing_capture.data(True), pulse=self.pulse_capture.data(True)),
                holding_summary=dict(anchor_executed_s=self.atomic_execution['anchor_executed_s'],
                    anchor_cycle_H_s=matched[0]['executed_holding_s'], all_cycles_H_s=total),
                contract=self.contract, ledger=self.ledger, results=results,
                completion={k:self.env._measurement_details[k] for k in ('trips_unlaunched','trips_incomplete')},
                selector_defined=False, main_policy_changed=False, v238_changed=False))
    return AtomicAudit


def validate_smoke(branches, same, alignment):
    native = branches['native']
    q = native['anchor']['query']
    plan = smoke_plan(q['native_bin'], q['feasible_bins'])
    assert set(branches) == set(plan['physical']), 'missing/extra physical path; aliases cannot add starts'
    coupling = {}
    for name,b in branches.items():
        assert b['branch'] == name and not (b['selector_defined'] or b['main_policy_changed'] or b['v238_changed'])
        same(native['anchor'],b['anchor'],'V257_common_prefix')
        same(native['future'],b['future'],'V257_common_future')
        assert b['future']['root'] == SMOKE_ROOT and b['future']['event_seed'] == SMOKE_ROOT+10000000
        same(native['contract'],b['contract'],'V257_native_authority')
        assert b['ledger']['physical_vehicles'] == 12 and b['ledger']['incomplete_trips'] == 0
        assert b['completion'] == dict(trips_unlaunched=0,trips_incomplete=0)
        assert b['ledger']['executed_holding_s'] == b['ledger']['commanded_holding_s']
        measured = execution_chain(b['requests'], b['actions'], b['holds'], b['payloads']['v248']['departures'])
        same(measured,b['execution'],'V257_recomputed_execution')
        anchors = [r for r in b['requests'] if r['is_anchor']]
        assert len(anchors) == 1 and anchors[0]['query_index'] == q['query_index']
        chosen = q['native_bin'] if name in ('native','forced_native') else int(name[1:])
        assert anchors[0]['selected_bin'] == chosen
        assert b['holding_summary']['anchor_executed_s'] == measured['anchor_executed_s']
        assert b['execution']['complete'] and b['execution']['sampled_actions_exact'] and b['execution']['issued_commands_exact']
        expected = 0 if name in ('native','forced_native') else 1
        assert b['execution']['changed_requests'] == expected
        coupling[name] = alignment(native['uniforms'],b['uniforms'])
    forced = branches['forced_native']
    for key in ('payloads','results','execution','actions','holds','ledger','holding_summary','completion','uniforms'):
        same(native[key],forced[key],'V257_new_future_native_alias_'+key)
    actual = native['execution']['anchor_executed_s']
    assert any(b['execution']['anchor_executed_s'] != actual for b in branches.values()), 'no physical pulse difference'
    return dict(passed=True, physical_paths=len(branches), plan=plan, coupling=coupling,
        native_alias_exact=True, actual_execution_complete=True,
        interpretation='Engineering evidence only;no selection,promotion or policy-benefit gate.')
