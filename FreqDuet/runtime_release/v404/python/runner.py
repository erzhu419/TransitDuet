"""Isolated full-day V314 paths with frozen fast control and live slow plans."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from coupled_model import CoupledCurrentController
from law import SlowManager
from freqduet_native_event_graph import native_query
import run_freqduet_v267_bilateral_factorial as fixed
import run_freqduet_v276_native_continuation as previous


ACTOR, SAMPLER = 12011, 27011001
DAYS = (2077001, 2077002, 2077003, 2077004)
ARMS = ('fixed', 'plan', 'budget', 'joint')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False,
                  separators=(',', ':'), default=lambda value: value.item())
        stream.write('\n')


class Audit(fixed.BenchmarkAudit):
    def __init__(self, arm, credit):
        # Install the exact fixed360 initial timetable for every factorial arm.
        super().__init__(ACTOR, 'fixed_bilateral025')
        self.variant, self.credit = arm, credit
        self.bridge, self.manager = None, None
        self.decisions, self.callback_wall, self.publish_ticks = [], [], []
        self.public_clocks, self.return_cursor = None, 0

    def attach(self, runner):
        super().attach(runner)
        from env.bus import Bus
        self.manager = SlowManager(self.variant, self.env, self.credit)
        self.bridge = CoupledCurrentController(self.env, self.launches, self.credit, self.manager)
        self.bridge.attach(Bus)
        self.started = time.perf_counter()
        self.saved_step, self.saved_launch = self.env.step, self.env.launch_bus

        def launch(trip, actual_launch=None):
            ready = {int(bus.bus_id): float(bus.back_to_terminal_time)
                     for bus in self.env.bus_in_terminal
                     if bool(bus.direction) == bool(trip.direction)}
            bus = self.saved_launch(trip, actual_launch=actual_launch)
            if bus is not None:
                # BenchmarkAudit has just appended this successful dispatch.
                self.dispatches[-1]['ready_return_s'] = ready[int(bus.bus_id)]
                self.manager.on_launch(trip, bus, float(self.env.current_time))
            return bus

        def step(*args, **kwargs):
            now = float(self.env.current_time)
            if self.public_clocks is None:
                self.public_clocks = {int(t.launch_turn): float(t.launch_time)
                                      for t in self.env.timetables}
            if now >= 7200 and now % 900 == 0:
                # Same-tick lower commands may already have been submitted.
                # They stay irrevocable; SlowManager sees the actual pending list.
                self.publish_ticks.append(dict(time_s=now,
                    already_submitted=[dict(bus=c.bus_id, trip=c.trip_id,
                        station=c.station_id, action_s=c.action_s)
                        for c in self.credit.stream.pending_commands]))
                self.manager.publish(self.env, self.bridge)
            result = self.saved_step(*args, **kwargs)
            for event in self.credit.stream.returns[self.return_cursor:]:
                self.manager.on_return(int(event['bus_id']), int(event['trip_id']),
                                       float(event['physical_time_s']))
            self.return_cursor = len(self.credit.stream.returns)
            return result

        self.env.launch_bus, self.env.step = launch, step

    def __call__(self, policy, states, features, theta):
        begin = time.perf_counter()
        law = super().__call__(policy, states, features, theta)
        bus = self.runner._bus_for_agent(self.runner._pilot_agent)
        legal = tuple(int(fixed.pilot.step.ACTIONS[i]) for i in np.flatnonzero(
            policy.feasible_action_mask(states)[0].numpy()))
        native = int(fixed.pilot.step.ACTIONS[int(law.argmax(dim=1).item())])
        query = native_query(self.env, bus, legal, native, self.credit.stream.pending_commands)
        profile = self.bridge.observe(bus, legal, native)
        action = int(profile['shadow_action_s'])
        assert action in legal, 'budget controller requested an illegal physical action'
        # This point-mass action is irrevocably submitted. Do not wait for physical
        # dwell acknowledgement, nor debit again when that acknowledgement arrives.
        self.manager.commit(int(bus.bus_id), int(bus.trip_id),
                            int(bus.last_station.station_id), float(self.env.current_time), action)
        query['executed_request_s'] = action
        self.decisions.append(query)
        self.callback_wall.append(time.perf_counter()-begin)
        if len(self.decisions) % 100 == 0:
            print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='v314_'+self.variant,
                current=float(self.env.current_time), total=float(self.env.protocol.evaluation_end_time_s),
                unit='sim_seconds', queries=len(self.decisions),
                elapsed_s=time.perf_counter()-self.started)), flush=True)
        return previous.selected_law(law, action, legal)

    def detach(self):
        self.env.step, self.env.launch_bus = self.saved_step, self.saved_launch
        if self.bridge is not None:
            self.bridge.detach()
        super().detach()

    def finish(self, runner, records, holds):
        # The original BenchmarkAudit.finish also asserts fixed360 for all plans.
        # Retain its physical/accounting checks here while allowing published
        # schedules in plan/joint arms to change. Historical code is untouched.
        env = runner.env
        assert self.scalars == (float(runner.lower_trainer.alpha),
            float(runner.lower_trainer.lambda_param), float(runner.upper_trainer.alpha))
        self.contract = fixed.prior.factorial.prior.physical_contract(env, runner.cfg)
        self.dispatch = fixed.prior.factorial.prior.dispatch_metrics(env)
        self.holding_person_s = float(sum(bus.episode_hold_person_seconds for bus in env.bus_all))
        commands = float(sum(row['action_s'] for row in holds))
        executed = float(sum(bus.episode_hold_vehicle_seconds for bus in env.bus_all))
        assert len(env.bus_all) == 12 and executed <= commands+1e-8
        if env._done_reason == 'service_cleared':
            np.testing.assert_allclose(commands, executed, atol=1e-8, rtol=0)
        self.execution = dict(commanded_s=commands, executed_s=executed,
            unexecuted_s=commands-executed, done_reason=env._done_reason,
            decisions=len(records), physical_vehicles=12)
        self.delays = dict(
            public_launch_delay_total_s=sum(max(0., r['actual_s']-r['public_s']) for r in self.dispatches),
            readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in self.dispatches))
        assert self.public_clocks is not None
        assert {int(t.launch_turn) for t in env.timetables} == self.public_clocks.keys()
        assert all(r['public_s'] == self.public_clocks[r['trip']] for r in self.dispatches)
        if self.variant in ('fixed', 'budget'):
            assert all(r['target_s'] == 360. for r in self.dispatches)
            for direction in (False, True):
                times = [r['scheduled_s'] for r in self.dispatches if r['direction'] == direction]
                np.testing.assert_allclose(np.diff(times), 360., atol=1e-8, rtol=0)
        self.chain = previous.executed_chain(self.decisions, holds)
        assert self.execution['done_reason'] == 'service_cleared'
        assert self.execution['unexecuted_s'] == 0
        assert not self.manager.active and not self.manager.remaining
        assert sum(len(rows) for rows in self.manager.completed.values()) == len(self.dispatches)


def run_one(args):
    label = f'd{args.day}_{args.arm}'
    output = args.out/label
    output.mkdir(parents=True, exist_ok=False)
    write(output/'started.json', dict(day=args.day, arm=args.arm, actor=ACTOR, sampler=SAMPLER,
        physical_start_cap=1, automatic_retry=False, training_updates=0))
    # optimized_runtime resolves its existing proof from this root's parent.
    runtime = args.results/('protocol_v6_v314_runtime03_'+label)
    runtime.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    credit = fixed.pilot.CreditObserver()
    audit = Audit(args.arm, credit)
    try:
        row, uniforms, _, holds = fixed.pilot.rollout(runtime, ACTOR, args.day, 'stochastic',
            SAMPLER, [0.]*4, 'v314_'+label, probability_rule=audit,
            trace_observer=audit, credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]),
            onboard_person_s=float(credit.stream.total[1]), backlog_trip_s=float(credit.stream.total[4]),
            holding_person_s=audit.holding_person_s, **audit.dispatch, **audit.delays)
        previous.label_resources(audit.bridge, audit.dispatches, holds)
        trace = dict(holds=holds, dispatches=audit.dispatches, arrivals=audit.arrivals,
            departures=audit.departures, queries=audit.decisions, uniforms=uniforms,
            returns=credit.stream.returns, graph_queries=audit.bridge.queries,
            resource_predictions=audit.bridge.predictions, slow_events=audit.manager.events,
            publish_ticks=audit.publish_ticks, public_clocks=audit.public_clocks)
        raw_path = output/'raw/path.pkl.gz'
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(raw_path, 'xb') as stream:
            pickle.dump(trace, stream, pickle.HIGHEST_PROTOCOL)
        quota_probes = [q['quota_probe'] for q in audit.bridge.queries if 'quota_probe' in q]
        result = dict(status='complete_development_path', day=args.day, arm=args.arm,
            actor=ACTOR, sampler=SAMPLER, results=row, contract=audit.contract,
            coverage=audit.coverage, execution=audit.execution, action_chain=audit.chain,
            graph=audit.bridge.summary(), slow=audit.manager.summary(),
            quota_probes=dict(attempted=len(quota_probes),
                supported=sum(p['supported'] is True for p in quota_probes),
                changed=sum(p['changed'] is True for p in quota_probes),
                extra_evaluations=sum(p['evals'] for p in quota_probes)),
            wall_s=time.perf_counter()-begin,
            callback_latency_s=dict(mean=float(np.mean(audit.callback_wall)),
                p95=float(np.quantile(audit.callback_wall, .95)), max=max(audit.callback_wall)),
            physical_starts=1, training_updates=0, main_policy_changed=False,
            confirmation_expanded=False, controller_reference='same-plan budget-conditioned native',
            limits='Consumed-day development path; outcomes are not independent confirmation.')
        write(output/'summary/results.json', result)
        print('V314_PATH_COMPLETE '+label, flush=True)
        return result
    except Exception:
        write(output/'summary/failure.json', dict(error=traceback.format_exc()[-7000:],
            time_s=float(audit.env.current_time) if audit.env is not None else None,
            day=args.day, arm=args.arm, automatic_retry=False, physical_starts=1,
            training_updates=0, main_policy_changed=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--day', type=int, choices=DAYS, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    run_one(parser.parse_args())
