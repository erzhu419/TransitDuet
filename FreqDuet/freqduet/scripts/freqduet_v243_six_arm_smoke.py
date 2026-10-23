"""V240 six-arm callback adapter; this module never starts a rollout."""

import copy

import numpy as np
import torch

from preregister_freqduet_v238_persistent_requests import ACTIONS
from preregister_freqduet_v240_paired_coordination import ARMS, CONTEXTS, SMOKE_SEED


ARM_RULES = {name: (upper, lower) for name, upper, lower in ARMS}
UPPER_PAIRS = (('upper_only', 'v107'), ('joint_supply', 'lower_supply'),
               ('joint_blind', 'lower_blind'))


def anchor_snapshot(audit, runner, policy, state, native, uniforms, repeated, joint):
    """Equality evidence only; latent state is never a request/gating input."""
    def passenger(p):
        return (p.appear_time, p.boarding_time, p.arrive_time, p.appear_station.station_id,
                p.destination_station.station_id, p.boarded, p.arrived)

    env, resource = runner.env, audit.resource
    buses = [dict(bus_id=b.bus_id, trip_id=b.trip_id, direction=b.direction,
        on_route=b.on_route, state=b.state.name, speed=b.current_speed,
        last_station=b.last_station.station_id, next_station=b.next_station.station_id,
        travel_distance=b.travel_distance, absolute_distance=b.absolute_distance,
        last_station_dis=b.last_station_dis, next_station_dis=b.next_station_dis,
        remaining_dwell=b.holding_time, holding_counter=b.episode_hold_vehicle_seconds,
        passengers=[passenger(p) for p in b.passengers]) for b in env.bus_all]
    assert len(buses) == 12
    return copy.deepcopy(dict(fresh_query=audit.fresh_queries[-1],
        upper_state=np.asarray(state), native_action=np.asarray(native),
        physical=repeated.physical_prefix(env), buses=buses,
        waiting=[[passenger(p) for p in s.waiting_passengers] for s in env.stations],
        ready_order=[b.bus_id for b in env.bus_in_terminal],
        resources=joint.resource_snapshot(env, audit.launches),
        native_plan_caches=runner._active_timetable_plans,
        legacy=dict(events=audit.clock.events, commands=audit.commands,
            callbacks=audit.callbacks, queries=audit.queries, inputs=audit.policy_inputs,
            dispatches=audit.dispatches, returns=audit.returns,
            arrivals=audit.arrivals, departures=audit.departures),
        v224=audit.handoff.data(), v230=resource.data(False),
        v235=audit.cycle_capture.data(False),
        rng=joint.pilot.joint.rng_state(runner, torch),
        lower_sampling=policy.sampling_state(),
        upper_sampling=runner.upper_trainer.policy_net.sampling_state(),
        scenario=dict(seed=env.scenario_tape.seed, version=env.scenario_tape.version,
            streams={k: r.get_state() for k, r in env.scenario_tape._streams.items()}),
        event_uniforms=dict(seed=uniforms.seed, counts=dict(uniforms.counts), records=uniforms.records)))


def selected_request(arm, packet, anchor_seq):
    _, lower = ARM_RULES[arm]
    active = anchor_seq is not None and packet['native_event_seq'] > anchor_seq
    if not active or lower == 'native':
        return packet['native_bin'], 'native'
    name = 'supply_request' if lower == 'v238' else 'blind_request'
    return packet[name]['shadow_bin'], lower


def execution_chain(queries, actions, holds, departures):
    assert len(queries) == len(actions), 'missing sampled lower action'
    issued, left = {}, {}
    for h in holds:
        key = tuple(h[k] for k in ('time_s', 'bus_id', 'trip_id', 'station_id'))
        issued.setdefault(key, []).append(h)
    for d in departures:
        key = tuple(d[k] for k in ('trip_id', 'station_id', 'direction'))
        left.setdefault(key, []).append(d)
    changed, executed, missing, total = 0, 0, [], 0.
    for q, action in zip(queries, actions):
        ident = q['identity']
        assert (action['time_s'], action['trip_id'], action['station_id'], action['direction']) == (
            q['query_s'], ident['trip_id'], ident['station_id'], ident['direction'])
        assert action['action'] == q['selected_bin'], 'sampled bin differs from returned law'
        assert np.array_equal(action['probabilities'], np.eye(7)[q['selected_bin']]), 'returned law changed'
        if q['selected_bin'] == q['native_bin']:
            continue
        changed += 1
        key = (q['query_s'], ident['bus_id'], ident['trip_id'], ident['station_id'])
        commands = issued.get(key, [])
        assert len(commands) == 1 and commands[0]['action_s'] == q['issued_s'], 'command not issued exactly once'
        key = (ident['trip_id'], ident['station_id'], ident['direction'])
        ds = [d for d in left.get(key, []) if d['time_s'] >= q['query_s']]
        if not ds:
            missing.append(dict(query_index=q['query_index'], executed_s=None, status='no_departure'))
            continue
        assert len(ds) == 1, 'ambiguous departure for a changed command'
        actual = ds[0]['realized_holding']-q['executed_holding_before_s']
        assert actual == q['issued_s'], 'issued holding did not reach physical departure'
        executed += 1
        total += actual
    return dict(changed_requests=changed, executed_changed_requests=executed,
        missing_departures=missing, changed_executed_holding_s=total,
        sampled_actions_exact=True, issued_commands_exact=True,
        complete=not missing)


def smoke_audit(passive_base, repeated, event, joint):
    """Compose frozen V241/V235/V224/V221; never use legacy lower/joint arms."""
    same = joint.pilot.joint.assert_same_state

    class SixArmAudit(passive_base):
        def __init__(self, seed, arm, uniforms, reference=None):
            assert arm in ARM_RULES and seed == CONTEXTS[0][0]
            assert isinstance(uniforms, event.EventUniforms)
            assert arm == 'v107' or reference is not None, 'non-reference arm needs the frozen V107 prefix'
            upper, _ = ARM_RULES[arm]
            super().__init__(seed, 'upper' if upper == 'v220' else 'v107')
            self.branch, self.uniforms, self.prefix_reference = arm, uniforms, reference
            assert uniforms.seed == 187000000+seed+100*CONTEXTS[0][1]
            self.fork_anchor, self.future_receipt, self.lower_policy = None, None, None
            self.issued_requests = []

        def attach(self, runner):
            super().attach(runner)
            self.sample_original = runner.upper_trainer.policy_net.get_action

            def action(state, *args, **kwargs):
                count = len(self.fresh_queries)
                native = self.sample_original(state, *args, **kwargs)
                assert len(self.fresh_queries) == count+1, 'anchor hook was not a native fresh sample'
                if self.fork_anchor is not None or self.env.current_time < 7200.:
                    return native
                assert self.lower_policy is not None, 'no initialized lower policy at anchor'
                q = self.fresh_queries[-1]
                assert q['fresh_upper'] and q['time_s'] == self.env.current_time
                assert q['trip_id'] == self.current_trip.launch_turn
                np.testing.assert_array_equal(np.asarray(native).reshape(-1), q['native_coefficient'])
                anchor = anchor_snapshot(self, runner, self.lower_policy, state, native,
                                         self.uniforms, repeated, joint)
                if self.prefix_reference is not None:
                    same(anchor, self.prefix_reference['anchor'], 'V243_exact_precommand_prefix')
                # No actor resample: the same native object returns to the frozen V221 planner.
                current = np.asarray(native).copy()
                startup = repeated.fork_future(runner, self.lower_policy, SMOKE_SEED)
                self.uniforms.seed = SMOKE_SEED+10000000
                same(anchor['physical'], repeated.physical_prefix(self.env), 'V243_preserved_current_window')
                after = anchor_snapshot(self, runner, self.lower_policy, state, native,
                                        self.uniforms, repeated, joint)
                for key in anchor.keys()-{'rng', 'lower_sampling', 'upper_sampling', 'scenario', 'event_uniforms'}:
                    same(anchor[key], after[key], 'V243_fork_preserves_'+key)
                same(anchor['event_uniforms']['counts'], dict(self.uniforms.counts), 'V243_kept_occurrences')
                same(anchor['event_uniforms']['records'], self.uniforms.records, 'V243_kept_event_prefix')
                np.testing.assert_array_equal(native, current)
                receipt = dict(startup=startup, event_seed=self.uniforms.seed,
                    event_counts=dict(self.uniforms.counts), scenario_seed=SMOKE_SEED)
                if self.prefix_reference is not None:
                    same(receipt, self.prefix_reference['future'], 'V243_exact_coupled_future_start')
                self.fork_anchor, self.future_receipt = anchor, receipt
                return native

            runner.upper_trainer.policy_net.get_action = action

        def __call__(self, policy, states, features, theta):
            self.lower_policy = policy
            native = super().__call__(policy, states, features, theta)
            p = self.request_capture.queries[-1]
            selected, rule = selected_request(self.branch, p,
                None if self.fork_anchor is None else self.fork_anchor['fresh_query']['event_seq'])
            q = self.queries[-1]
            assert p['native_bin'] == int(native.argmax(dim=1).item()) == q['base_bin']
            assert selected in q['feasible_bins']
            value = native
            if selected != p['native_bin']:
                value = torch.zeros_like(native)
                value[0, selected] = 1.
                q['command_s'] = ACTIONS[selected]
            row = {k: copy.deepcopy(v) for k, v in p.items() if k not in ('returned_rule', 'executed_shadow')}
            row.update(selected_bin=selected, issued_s=ACTIONS[selected], returned_rule=rule,
                       changed=selected != p['native_bin'])
            self.issued_requests.append(row)
            return value

        def detach(self):
            self.runner.upper_trainer.policy_net.get_action = self.sample_original
            super().detach()

        def finish(self, runner, records, holds):
            assert self.fork_anchor is not None, 'registered first fresh anchor missing; no replacement'
            super().finish(runner, records, holds)
            self.action_execution = execution_chain(self.issued_requests, self.actions, holds, self.departures)

        def branch_data(self):
            assert self.fork_anchor is not None
            return dict(arm=self.branch, anchor=self.fork_anchor, future=self.future_receipt,
                requests=copy.deepcopy(self.issued_requests), execution=copy.deepcopy(self.action_execution),
                uniforms=copy.deepcopy(self.uniforms.records), physical_contract=copy.deepcopy(self.contract),
                ledger=copy.deepcopy(self.ledger), tasks=copy.deepcopy(list(self.tasks.values())),
                dispatches=copy.deepcopy(self.dispatches), launch_plans=copy.deepcopy(self.launch_plans),
                completion={k: self.env._measurement_details[k] for k in ('trips_unlaunched', 'trips_incomplete')},
                main_policy_changed=False)
    return SixArmAudit


def runtime_audit():
    """Construct the adapter class only; caller must separately authorize any rollout."""
    import run_freqduet_v230_resource_capture as resource
    import freqduet_v235_passive_cycle_capture as cycle
    import freqduet_v241_passive_requests as requests
    old = resource.frozen()
    base = requests.recording_audit(cycle.recording_audit(old.RecordingAudit))
    joint = old.legacy.passive.prior
    assert 'frozen_future_c45b2d0a93' in joint.repeated.__file__
    assert 'frozen_future_c45b2d0a93' in old.legacy.pilot.event.__file__
    return smoke_audit(base, joint.repeated, old.legacy.pilot.event, joint)


def check_smoke(branches, same, alignment, task_execution, tick_s):
    assert set(branches) == set(ARM_RULES), 'six registered arms required; no replacements'
    reference = branches['v107']
    lower_checks, coupling = {}, {}
    for name, branch in branches.items():
        assert branch['arm'] == name and not branch['main_policy_changed']
        same(reference['anchor'], branch['anchor'], 'V243_smoke_common_prefix')
        same(reference['future'], branch['future'], 'V243_smoke_common_future')
        assert branch['future']['scenario_seed'] == SMOKE_SEED
        same(reference['physical_contract'], branch['physical_contract'], 'V243_smoke_authority')
        assert branch['ledger']['physical_vehicles'] == 12 and branch['ledger']['incomplete_trips'] == 0
        assert branch['completion'] == dict(trips_unlaunched=0, trips_incomplete=0)
        assert branch['ledger']['executed_holding_s'] == branch['ledger']['commanded_holding_s']
        coupling[name] = alignment(reference['uniforms'], branch['uniforms'])
        result = branch['execution']
        assert result['sampled_actions_exact'] and result['issued_commands_exact']
        lower_checks[name] = bool(result['complete'] and
            (result['changed_requests'] == 0 if ARM_RULES[name][1] == 'native' else
             result['changed_requests'] == result['executed_changed_requests'] > 0))
    upper_checks = {}
    for candidate, base in UPPER_PAIRS:
        b, c = branches[base], branches[candidate]
        before = {d['trip']: d for d in b['dispatches']}
        after = {d['trip']: d for d in c['dispatches']}
        plans = lambda data, trip: data['launch_plans'].get(trip, data['launch_plans'].get(str(trip)))
        rows = [task_execution(t, before.get(t['trip_id']), after.get(t['trip_id']),
            plans(b, t['trip_id']), plans(c, t['trip_id']), tick_s) for t in c['tasks']]
        upper_checks[candidate+'-'+base] = dict(tasks=rows, passed=any(r['passed'] for r in rows))
    differences = sum(p['supply_request']['shadow_bin'] != p['blind_request']['shadow_bin']
                      for b in branches.values() for p in b['requests'])
    passed = all(lower_checks.values()) and all(r['passed'] for r in upper_checks.values()) and differences > 0
    return dict(passed=passed, lower_execution=lower_checks, upper_execution=upper_checks,
        same_state_dual_request_differences=differences, coupling=coupling,
        interpretation='Engineering support only;no policy gain or promotion gate evaluated.')
