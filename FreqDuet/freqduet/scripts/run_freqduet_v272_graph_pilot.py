#!/usr/bin/env python3
"""Twelve authorized deterministic graph-allocation development paths."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback


SEED, SAMPLER = 12011, 27011001
DAYS = (2071001, 2071002, 2071003, 2071004)
ARMS = ('fixed360_bilateral025', 'event_graph_apc', 'event_graph_homogeneous_cost')
ITEMS = tuple((day, arm) for arm in ARMS for day in DAYS)
SHARDS = tuple(tuple(range(i, len(ITEMS), 8)) for i in range(8))
GRAPH_SOURCE, PREFLIGHT_SOURCE = '0a68748858', '40eddb8fb6'
METRICS = ('passenger_person_s', 'journey_min', 'cv_squared', 'readiness_delay_s',
           'public_launch_delay_total_s', 'trips_unlaunched', 'trips_incomplete',
           'holding_s_per_trip', 'holding_person_s', 'waiting_person_s', 'onboard_person_s')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def selected_law(native, action, legal):
    import torch
    assert action in legal and action in (0, 5, 10, 15, 20, 30, 45)
    result = torch.zeros_like(native)
    result[0, (0, 5, 10, 15, 20, 30, 45).index(action)] = 1.
    return result


def executed_chain(decisions, holds):
    expected = {(r['time_s'], r['bus'], r['trip'], r['station']): r['executed_request_s'] for r in decisions}
    actual = {(r['time_s'], r['bus_id'], r['trip_id'], r['station_id']): r['action_s'] for r in holds}
    assert len(expected) == len(decisions) == len(holds) == len(actual) and expected == actual
    return dict(verified_commands=len(holds), changed_from_native=sum(
        r['executed_request_s'] != r['action_s'] for r in decisions))


def make_audit(fixed, arm, credit):
    import numpy as np
    from freqduet_native_event_graph import NativeNoop, native_query
    import freqduet_causal_event_graph as graph

    class Audit(fixed.BenchmarkAudit):
        def __init__(self):
            super().__init__(SEED, 'fixed_bilateral025')
            self.bridge, self.decisions, self.callback_wall = None, [], []

        def attach(self, runner):
            super().attach(runner)
            if arm != ARMS[0]:
                from env.bus import Bus
                self.bridge = NativeNoop(runner.env, self.launches, credit)
                self.bridge.planner = graph.OnlinePlanner(homogeneous=arm == ARMS[2])
                self.bridge.attach(Bus)

        def __call__(self, policy, states, features, theta):
            start = time.perf_counter()
            law = super().__call__(policy, states, features, theta)
            bus = self.runner._bus_for_agent(self.runner._pilot_agent)
            legal = tuple(int(fixed.pilot.step.ACTIONS[i]) for i in np.flatnonzero(
                policy.feasible_action_mask(states)[0].numpy()))
            native = int(fixed.pilot.step.ACTIONS[int(law.argmax(dim=1).item())])
            query = native_query(self.env, bus, legal, native, credit.stream.pending_commands)
            action = native if self.bridge is None else self.bridge.observe(bus, legal, native)['shadow_action_s']
            assert action in legal
            query['executed_request_s'] = int(action)
            self.decisions.append(query)
            self.callback_wall.append(time.perf_counter()-start)
            return law if self.bridge is None else selected_law(law, int(action), legal)

        def detach(self):
            if self.bridge is not None:
                self.bridge.detach()
            super().detach()

        def finish(self, runner, records, holds):
            super().finish(runner, records, holds)
            self.command_chain = executed_chain(self.decisions, holds)

    return Audit()


def collect(a):
    assert a.index in range(len(ITEMS))
    day, arm = ITEMS[a.index]
    protocol = read(a.root/'protocol.json')
    assert protocol['source'] == a.source and protocol['unique_paths'] == 12
    write(a.root/'started'/f'p{a.index}.json', dict(source=a.source, day=day, arm=arm, automatic_retry=False))
    import numpy as np
    import run_freqduet_v267_bilateral_factorial as fixed
    credit = fixed.pilot.CreditObserver()
    audit = make_audit(fixed, arm, credit)
    start = time.perf_counter()
    row, uniforms, _, holds = fixed.pilot.rollout(a.root, SEED, day, 'stochastic', SAMPLER,
        [0.]*4, f'v272_p{a.index}', probability_rule=audit, trace_observer=audit, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=audit.holding_person_s,
        **audit.dispatch, **audit.delays)
    trace = dict(holds=holds, dispatches=audit.dispatches, arrivals=audit.arrivals,
        departures=audit.departures, queries=audit.decisions, uniforms=uniforms)
    if audit.bridge is not None:
        trace.update(graph_queries=audit.bridge.queries, completed_visits=audit.bridge.prefix.visits)
    path = a.root/'traces'/f'p{a.index}.pkl.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'xb') as stream:
        pickle.dump(trace, stream, protocol=pickle.HIGHEST_PROTOCOL)
    cached_exact = None
    if arm == ARMS[0] and day == DAYS[0]:
        cached = read(a.root.parent/'protocol_v6_v271_native_event_graph/paths/reference.json')
        for field in ('contract', 'coverage', 'execution'):
            fixed.pilot.joint.assert_same_state(getattr(audit, field), cached[field], 'V272_reference_'+field)
        for metric in (*fixed.METRICS, 'action_counts'):
            fixed.pilot.joint.assert_same_state(row[metric], cached['results'][metric], 'V272_reference_'+metric)
        with gzip.open(a.root.parent/'protocol_v6_v271_native_event_graph/traces/reference.pkl.gz', 'rb') as stream:
            old = pickle.load(stream)
        for field in ('holds', 'dispatches', 'arrivals', 'departures', 'uniforms'):
            fixed.pilot.joint.assert_same_state(trace[field], old[field], 'V272_reference_trace_'+field)
        cached_exact = True
    graph_summary = None if audit.bridge is None else audit.bridge.summary()
    if graph_summary is not None:
        del graph_summary['executed_shadow_actions']
        graph_summary['executed_model_overrides'] = audit.command_chain['changed_from_native']
    write(a.root/'pairs'/f'p{a.index}.json', dict(source=a.source, graph_source=GRAPH_SOURCE,
        index=a.index, day=day, arm=arm, actor_seed=SEED, sampler_seed=SAMPLER, results=row,
        contract=audit.contract, coverage=audit.coverage, execution=audit.execution,
        action_chain=audit.command_chain, callback_latency_s=dict(mean=float(np.mean(audit.callback_wall)),
            p95=float(np.quantile(audit.callback_wall,.95)), max=float(max(audit.callback_wall))),
        graph=graph_summary, cached_reference_exact=cached_exact,
        wall_s=time.perf_counter()-start, training_updates=0, main_policy_changed=False))
    print('V272_PATH_COMPLETE', a.index, day, arm, flush=True)


def run_unit(a):
    assert a.unit in range(8)
    write(a.root/'units_started'/f'{a.unit}.json', dict(source=a.source, items=SHARDS[a.unit]))
    returns = {}
    for index in SHARDS[a.unit]:
        # One fresh process per authorized path; failure never retries or changes
        # the next fixed item, and at most eight units can run concurrently.
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'path', '--root', str(a.root),
               '--source', a.source, '--index', str(index)]
        returns[str(index)] = subprocess.run(cmd, check=False).returncode
    write(a.root/'units'/f'{a.unit}.json', dict(source=a.source, returncodes=returns,
        status='complete' if not any(returns.values()) else 'contains_failed_path', automatic_retry=False))
    assert not any(returns.values()), returns


def summarize(cube):
    import numpy as np
    rng = np.random.default_rng(2700201)
    indices = rng.integers(0, len(DAYS), size=(10000, len(DAYS)))
    result = {}
    for base in (ARMS[0], ARMS[2]):
        delta = np.asarray([[cube[d, ARMS[1]][m]-cube[d, base][m] for m in METRICS] for d in DAYS])
        boot = delta[indices].mean(axis=1)
        stats = {m:dict(mean=float(delta[:,i].mean()), ci95=np.quantile(boot[:,i],[.025,.975]).tolist(),
            by_day={str(d):float(delta[j,i]) for j,d in enumerate(DAYS)}) for i,m in enumerate(METRICS)}
        passed = stats['passenger_person_s']['ci95'][1] < 0 and all(
            stats[m]['ci95'][1] <= 0 for m in ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s'))
        passed = passed and all(delta[:,METRICS.index(m)].max() <= 0 for m in ('trips_unlaunched','trips_incomplete'))
        result['apc_vs_'+base] = dict(metrics=stats, joint_development_gate=bool(passed))
    return result


def aggregate(a):
    import run_freqduet_v267_bilateral_factorial as fixed
    rows = [read(a.root/'pairs'/f'p{i}.json') for i in range(len(ITEMS))]
    assert {p.stem for p in (a.root/'started').glob('*.json')} == {f'p{i}' for i in range(12)}
    assert not list((a.root/'failures').glob('*.json'))
    cube = {}
    alignment = {}
    for index, row in enumerate(rows):
        assert row['source'] == a.source and (row['day'],row['arm']) == ITEMS[index]
        cube[row['day'],row['arm']] = row['results']
    for day in DAYS:
        cell = [r for r in rows if r['day'] == day]
        for row in cell[1:]:
            fixed.pilot.joint.assert_same_state(row['contract'],cell[0]['contract'],'V272_matched_day_contract')
        with gzip.open(a.root/'traces'/f"p{cell[0]['index']}.pkl.gz", 'rb') as stream:
            base = pickle.load(stream)['uniforms']
        for row in cell[1:]:
            with gzip.open(a.root/'traces'/f"p{row['index']}.pkl.gz", 'rb') as stream:
                other = pickle.load(stream)['uniforms']
            alignment[str(day)+'_'+row['arm']] = fixed.pilot.event.alignment(base,other)
    effects = summarize(cube)
    write(a.root/'summary/results.json', dict(source=a.source, status='twelve_unique_development_paths_complete',
        effects=effects, joint_development_gate=all(v['joint_development_gate'] for v in effects.values()),
        independent_demand_days=4, independent_future_replications=0, event_uniform_alignment=alignment,
        physical_paths=12, confirmation_started=False, training_updates=0, main_policy_changed=False,
        controller_admitted=False, scope='Consumed-day deterministic development screen, not independent-future or actor confirmation.'))


def check(a):
    import unittest
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v272_graph_pilot.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.testsRun > 0 and result.wasSuccessful()
    write(a.root/'checks/status.json', dict(source=a.source, status='execution_and_unique_grid_checks_passed',
        tests=result.testsRun, physical_starts=0, training_updates=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('check','path','unit','aggregate'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--index', type=int)
    p.add_argument('--unit', type=int)
    a = p.parse_args()
    try:
        dict(check=check, path=collect, unit=run_unit, aggregate=aggregate)[a.command](a)
    except Exception as error:
        if a.command == 'path':
            write(a.root/'failures'/f'p{a.index}.json', dict(source=a.source, error=repr(error),
                traceback=traceback.format_exc()[-8000:], automatic_retry=False, slot_consumed=True))
        raise


if __name__ == '__main__':
    main()
