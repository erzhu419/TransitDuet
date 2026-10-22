#!/usr/bin/env python3
"""V214 passive equivalence and one fixed inbound-chain joint smoke, no matrix."""

import argparse
from dataclasses import asdict
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import torch

import preregister_freqduet_v214_inbound_chain as protocol
import run_freqduet_v213_passive_coordination as passive


prior = passive.prior
SMOKE_FUTURE = 2149000000
SMOKE_CANDIDATE = 3  # -30s native coefficient, one-bin reduction per committed trip.


def controlled_stop_ahead(bus):
    index = bus.effective_station.index(bus.last_station)
    if bus.state.name not in ('HOLDING', 'WAITING_ACTION'):
        index += 1
    # B200 uses terminal "transition": holding at the last nonterminal stop is
    # legal even when its next station is the terminal.
    return any(i >= 1 and i + 1 < len(bus.effective_station)
               for i in range(index, len(bus.effective_station) - 1))


def supply_frame(env, trip):
    ready, active = [], []
    now = float(env.current_time)
    for bus in env.bus_all:
        row = dict(bus_id=int(bus.bus_id), direction=bool(bus.direction), position_sample_s=now)
        if not bus.on_route:
            ready.append(row)
            continue
        length = float(sum(r.distance for r in bus.effective_route))
        active.append(dict(row, trip_id=int(bus.trip_id), last_station_id=int(bus.last_station.station_id),
            next_station_id=int(bus.next_station.station_id), progress_m=float(bus.travel_distance),
            remaining_distance_m=length-float(bus.travel_distance), state=bus.state.name,
            controlled_stop_ahead=controlled_stop_ahead(bus)))
    return dict(time_s=now, direction=bool(trip.direction), fresh_upper=True, ready=ready, active=active,
        plans=[dict(trip_id=int(t.launch_turn), direction=bool(t.direction), launched=bool(t.launched),
                    planned_s=float(getattr(t, '_freqduet_scheduled_launch', t.launch_time)))
               for t in env.timetables if not t.launched])


class InboundAudit(passive.PassiveAudit):
    def __init__(self, seed, candidate=0, **kwargs):
        # Reuse the frozen resource recorder, not its old same-direction selector.
        super().__init__(seed, 5*(candidate//2), **kwargs)
        self.inbound_candidate = candidate
        self.opportunities_seen, self.anchor_frames = set(), []
        self.observation_times = {}

    def attach(self, runner):
        super().attach(runner)
        self.prepare_original = self.bus_class._prepare_for_action

        def prepare(bus, current_time, *args, **kwargs):
            value = self.prepare_original(bus, current_time, *args, **kwargs)
            identity = (int(bus.bus_id), int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction))
            self.observation_times[identity] = float(current_time)
            return value

        self.bus_class._prepare_for_action = prepare

        def action(state, *args, **kwargs):
            native = self.get_action_original(state, *args, **kwargs)
            value, trip = native, self.current_upper_trip
            if self.anchor is None and trip is not None and 7200 <= self.env.current_time < 8100:
                frame = supply_frame(self.env, trip)
                inbound = protocol.choose_supply(frame)
                self.anchor_frames.append(dict(frame=frame, selected=inbound))
                if inbound:
                    prior.require(self.policy is not None, 'inbound anchor precedes lower initialization')
                    anchor = dict(time_s=frame['time_s'], trip=int(trip.launch_turn), direction=frame['direction'],
                        inbound=inbound, frame=frame, upper_state=np.asarray(state).tolist(),
                        native_action=np.asarray(native).tolist(), resources=prior.resource_snapshot(self.env, self.launches),
                        upper_prefix=list(self.upper_prefix), lower_prefix=list(self.lower_prefix),
                        event_prefix=[[list(k), v] for k, v in self.uniforms.records.items()])
                    if self.reference is not None:
                        prior.pilot.joint.assert_same_state(anchor, self.reference['anchor'], 'V214_common_prefix')
                    self.anchor = anchor
                    if self.fork is not None:
                        prior.repeated.fork_future(runner, self.policy, self.fork)
                        self.uniforms.seed = self.fork+10000000
                    shift, _ = protocol.CANDIDATES[self.inbound_candidate]
                    value = prior.upper_candidate(native, runner.upper_action_low, runner.upper_action_high, shift)
                    self.anchor_command = dict(native=np.asarray(native).tolist(), candidate=np.asarray(value).tolist())
            self.upper_commands.append(dict(time_s=float(self.env.current_time),
                trip_id=int(trip.launch_turn) if trip is not None else None,
                coefficient=np.asarray(value).reshape(-1).tolist()))
            return value

        # The passive upper wrapper still records every plan update; replace only
        # V211's old anchor/action selector with the sealed V214 selector.
        runner.upper_trainer.policy_net.get_action = action

    def detach(self):
        self.bus_class._prepare_for_action = self.prepare_original
        super().detach()

    def __call__(self, policy, states, features, theta):
        self.policy, self.method = policy, 'bilateral025'
        law = prior.prior.TradeoffAudit.__call__(self, policy, states, features, theta)
        runner, env = self.runner, self.env
        bus = runner._bus_for_agent(runner._pilot_agent)
        context = runner._pending_lower_action_context[int(runner._pilot_agent)]
        identity = [int(bus.trip_id), int(bus.last_station.station_id), bool(bus.direction), int(bus.bus_id)]
        base = int(law.argmax(dim=1).item())
        feasible = np.flatnonzero(policy.feasible_action_mask(states)[0].numpy()).tolist()
        captured = self.observation_times[identity[3], identity[0], identity[1], identity[2]]
        prior.require(captured <= env.current_time, 'lower context clock is in the future')
        query = dict(time_s=float(env.current_time), bus_id=identity[3], trip_id=identity[0], direction=identity[2],
            terminal_masked=bool(runner._lower_terminal_action_masked(bus) or bus.last_station.station_type == 0),
            forward_valid=bool(context.evidence_valid), follower_valid=bool(context.follower_evidence_valid),
            base_bin=base, feasible_bins=feasible)
        chosen = base
        if self.anchor is None:
            self.lower_prefix.append([float(env.current_time), *identity, base])
        else:
            smaller = protocol.reduction(self.anchor, query, self.opportunities_seen)
            if smaller is not None:
                self.opportunities_seen.add((identity[3], identity[0]))
                if protocol.CANDIDATES[self.inbound_candidate][1]:
                    chosen = smaller
                self.lower_changes.append(dict(time_s=float(env.current_time), identity=identity,
                    base_bin=base, selected_bin=chosen, smaller_bin=smaller,
                    actual_command_s=float(protocol.ACTIONS[chosen]), state_captured_s=captured))
        if chosen != base:
            law = torch.zeros_like(law)
            law[0, chosen] = 1.
        launch = self.launches[identity[0]]
        self.queries.append(dict(ordinal=len(self.queries), **query, station_id=identity[1],
            state_captured_s=captured, position_sample_s=float(env.current_time),
            progress_m=float(bus.travel_distance), command_s=float(protocol.ACTIONS[chosen]),
            context=asdict(context), onboard=len(bus.passengers), launch_s=launch['time_s'],
            executed_holding_before_s=float(bus.episode_hold_vehicle_seconds)-launch['holding_counter_at_launch']))
        return law

    def finish(self, runner, records, holds):
        prior.require(self.anchor is not None, 'V214 no supported anchor; fresh-query frames='+str(self.anchor_frames))
        super().finish(runner, records, holds)
        self.execution = dict(committed_trips=len(self.anchor['inbound']), opportunities=len(self.lower_changes),
            changed_commands=sum(r['selected_bin'] != r['base_bin'] for r in self.lower_changes),
            decisions=self.lower_changes, done_reason=runner.env._done_reason)
        self.full_trace['anchor_frames'] = self.anchor_frames


def replay(root, *, candidate=0, reference=None, future=None, label):
    seed, day = protocol.CONTEXTS[0]
    sampler = 187000000+seed+100*day
    uniforms, credit = prior.pilot.event.EventUniforms(sampler), prior.pilot.CreditObserver()
    rule = InboundAudit(seed, candidate, reference=reference, fork=future, uniforms=uniforms)
    row, events, _, _ = prior.pilot.rollout(root, seed, day, 'stochastic', sampler, [0.]*4, label,
        probability_rule=rule, trace_observer=rule, event_uniforms=uniforms, credit_observer=credit)
    public_total = sum(max(0., r['actual_s']-r['public_baseline_s']) for r in rule.dispatch_ledger)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=rule.holding_person_s,
        inclusive_launch_delay_s=rule.ledger['inclusive_launch_delay_s'], public_launch_delay_total_s=public_total,
        readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in rule.dispatch_ledger), **rule.dispatch)
    result = dict(context=0, seed=seed, day=day, candidate=candidate, future_seed=future,
        results={k: row[k] for k in (*prior.METRICS, 'action_counts', 'public_launch_delay_total_s', 'readiness_delay_total_s')},
        anchor=rule.anchor, upper=rule.applied_upper, anchor_command=rule.anchor_command,
        lower=rule.lower_changes, ledger=rule.ledger, authority=rule.authority,
        physical_contract=rule.contract, execution=rule.execution)
    passive.save_trace(root/'records'/f'{label}.json.gz', dict(result=result, trace=rule.full_trace))
    return result, rule, events


def exact(a, b, label):
    prior.pilot.joint.assert_same_state(a, b, 'V214_'+label)


def compare_paths(a, b):
    exact(a[2], b[2], 'full_action_events')
    for name in ('arrivals', 'departures', 'dispatch_ledger', 'return_ledger'):
        exact(getattr(a[1], name), getattr(b[1], name), name)


def smoke(root, registered, old):
    sealed = prior.read(registered/'protocol.json')
    expected = protocol.registration('a572e63a8b')
    exact(sealed, expected, 'sealed_registration')
    for pattern in ('test_v214_inbound_preregistration.py', 'test_v214_inbound_smoke.py'):
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
            str(Path(__file__).resolve().parents[1]/'tests'), '-p', pattern, '-v'], check=True)
    prior.require(not (root/'started.json').exists(), 'smoke already started; inspect progress before any retry')
    progress = dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'], candidate=SMOKE_CANDIDATE,
        future_seed=SMOKE_FUTURE, rollouts_started=0, rollouts_completed=0,
        selection_opened=False, evaluation_opened=False, training_updates=0, main_policy_changed=False)
    prior.pilot.write_json(root/'started.json', progress)

    def run(phase, function, **kwargs):
        progress.update(phase=phase, rollouts_started=progress['rollouts_started']+1)
        prior.require(progress['rollouts_started'] <= sealed['budget']['smoke_max'], 'smoke budget exceeded')
        prior.pilot.write_json(root/'progress.json', progress)
        value = function(**kwargs)
        progress['rollouts_completed'] += 1
        prior.pilot.write_json(root/'progress.json', progress)
        print('V214_SMOKE_PATH_DONE', phase, flush=True)
        return value

    try:
        native = run('observer_off', prior.replay, root=root, ci=0, label='v214_observer_off')
        cached = prior.read(old/'references/c0.json')
        exact(native[0]['results'], cached['results'], 'archived_baseline_endpoint')
        observed = run('observer_on', replay, root=root, label='observer_on')
        exact({k: observed[0]['results'][k] for k in native[0]['results']}, native[0]['results'], 'passive_endpoint')
        compare_paths(native, observed)
        base = run('forked_baseline', replay, root=root, reference=observed[0], future=SMOKE_FUTURE,
            label='forked_baseline')
        joint = run('joint', replay, root=root, candidate=SMOKE_CANDIDATE, reference=observed[0],
            future=SMOKE_FUTURE, label='joint')
        repeat = run('joint_repeat', replay, root=root, candidate=SMOKE_CANDIDATE, reference=observed[0],
            future=SMOKE_FUTURE, label='joint_repeat')
        exact(joint[0], repeat[0], 'same_future_full_result')
        exact(joint[1].full_trace, repeat[1].full_trace, 'same_future_full_trace')
        compare_paths(joint, repeat)
        prior.pilot.event.alignment(base[2], joint[2])
        changed = [r for r in joint[0]['lower'] if r['selected_bin'] != r['base_bin']]
        plan_deltas = require_real_execution(base, joint, changed)
        progress.update(status='v214_inbound_passive_equivalence_and_nonzero_smoke_passed',
            archived_baseline_endpoint_exact=True, passive_full_action_timing_resource_exact=True,
            joint_repeat_exact=True, lower_opportunities=len(joint[0]['lower']), nonzero_reductions=changed,
            changed_anchor_plan_clocks=plan_deltas,
            anchor_s=joint[0]['anchor']['time_s'], anchor_direction=joint[0]['anchor']['direction'],
            committed_inbound=joint[0]['anchor']['inbound'],
            smoke_effects={k: joint[0]['results'][k]-base[0]['results'][k]
                for k in ('passenger_person_s', 'journey_min', *protocol.CONSTRAINTS)},
            scope='Execution/equivalence smoke only; no selection/evaluation or policy admission')
        prior.pilot.write_json(root/'summary/status.json', progress)
        print('V214_INBOUND_SMOKE_PASSED', flush=True)
    except Exception as error:
        progress.update(status='v214_inbound_smoke_stopped', failure=str(error))
        prior.pilot.write_json(root/'summary/status.json', progress)
        raise


def changed_plan_clocks(base, joint):
    bm, jm = ({r['trip_id']: r for r in call['changes']} for call in (base, joint))
    deltas = []
    for trip in sorted(bm.keys() | jm.keys()):
        initial = (bm.get(trip) or jm[trip])['before']['planned_s']
        b = bm[trip]['after']['planned_s'] if trip in bm else initial
        j = jm[trip]['after']['planned_s'] if trip in jm else initial
        if abs(j-b) > 1e-8:
            deltas.append(dict(trip_id=trip, base_s=b, joint_s=j, delta_s=j-b))
    return deltas


def require_real_execution(base, joint, changed):
    command = joint[0]['anchor_command']
    prior.require(command['candidate'] != command['native'], 'smoke upper command is clipped to noop')
    b = next(r for r in base[1].upper_calls if r['trip_id'] == base[0]['upper']['trip'])
    j = next(r for r in joint[1].upper_calls if r['trip_id'] == joint[0]['upper']['trip'])
    plan_deltas = changed_plan_clocks(b, j)
    prior.require(bool(plan_deltas), 'smoke upper request did not change a plan clock')
    prior.require(bool(changed), 'smoke has no real nonzero inbound holding opportunity; do not extend/search')
    for r in changed:
        prior.require(r['base_bin'] > r['selected_bin'] and r['actual_command_s'] >= 0,
            'smoke reduction exceeded native authority')
        issued = [h for h in joint[1].full_trace['issued_holding']
            if h['trip_id'] == r['identity'][0] and h['station_id'] == r['identity'][1]]
        prior.require(len(issued) == 1 and issued[0]['action_s'] == r['actual_command_s'],
            'registered lower reduction differs from issued holding')
    for result, rule, _ in (base, joint):
        prior.require(result['results']['trips_unlaunched'] == result['results']['trips_incomplete'] == 0,
            'uncleared smoke service; no zero imputation')
        prior.require(rule.ledger['unexecuted_command_s'] == 0., 'smoke commands not fully executed')
    return plan_deltas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'registered-root', 'v211-root'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    smoke(a.out_root, a.registered_root, a.v211_root)


if __name__ == '__main__':
    main()
