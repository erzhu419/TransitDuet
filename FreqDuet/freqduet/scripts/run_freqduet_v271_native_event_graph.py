#!/usr/bin/env python3
"""V270 engineering admission, including one approved no-op correction slot."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import time
import traceback


SEED, DAY, FUTURE = 12011, 2071001, 27011001
GRAPH_SOURCE, SEAL = '0a68748858', 'b4c9fd23e3'
REFERENCE_SOURCE = '83d681d0ae'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(a):
    # The scheduler submission has already consumed this slot, including failure
    # before Python entry. This second marker prevents a duplicate physical start.
    account = a.results_root/'protocol_v6_v270_causal_event_graph/engineering'
    reservation = read(account/'budget'/f'{a.command}.json')
    assert reservation['source'] == a.source and not reservation['automatic_retry']
    write(account/'started'/f'{a.command}.json', dict(source=a.source, physical_slot=a.command))
    if a.command != 'reference':
        gate = read(a.root/'summary/reference.json')
        assert gate['source'] == REFERENCE_SOURCE and gate['status'] == 'reference_cached_parity_passed'
    import numpy as np
    import run_freqduet_v267_bilateral_factorial as fixed
    from freqduet_native_event_graph import NativeNoop, native_query

    credit = fixed.pilot.CreditObserver()

    class Audit(fixed.BenchmarkAudit):
        def __init__(self):
            super().__init__(SEED, 'fixed_bilateral025')
            self.native_queries, self.native_wall, self.callback_wall = [], [], []
            self.bridge = None

        def attach(self, runner):
            super().attach(runner)
            if a.command != 'reference':
                from env.bus import Bus
                self.bridge = NativeNoop(runner.env, self.launches, credit)
                self.bridge.attach(Bus)

        def __call__(self, policy, states, features, theta):
            start = time.perf_counter()
            law = super().__call__(policy, states, features, theta)
            bus = self.runner._bus_for_agent(self.runner._pilot_agent)
            legal = tuple(int(fixed.pilot.step.ACTIONS[i]) for i in np.flatnonzero(
                policy.feasible_action_mask(states)[0].numpy()))
            action = int(fixed.pilot.step.ACTIONS[int(law.argmax(dim=1).item())])
            self.native_wall.append(time.perf_counter()-start)
            self.native_queries.append(native_query(self.env, bus, legal, action, credit.stream.pending_commands))
            if self.bridge is not None:
                self.bridge.observe(bus, legal, action)
            self.callback_wall.append(time.perf_counter()-start)
            # The SAME original tensor returns to native categorical/quantization.
            return law

        def detach(self):
            if self.bridge is not None:
                self.bridge.detach()
            super().detach()

    audit = Audit()
    start = time.perf_counter()
    row, uniforms, episode, holds = fixed.pilot.rollout(a.root, SEED, DAY, 'stochastic', FUTURE,
        [0.]*4, 'v271_'+a.command, probability_rule=audit, trace_observer=audit, credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]), holding_person_s=audit.holding_person_s,
        **audit.dispatch, **audit.delays)
    trace = dict(holds=holds, dispatches=audit.dispatches, arrivals=audit.arrivals,
                 departures=audit.departures, queries=audit.native_queries, uniforms=uniforms)
    dest = a.root/'traces'/f'{a.command}.pkl.gz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, 'xb') as stream:
        pickle.dump(trace, stream, protocol=pickle.HIGHEST_PROTOCOL)
    result = dict(source=a.source, graph_source=GRAPH_SOURCE, protocol_seal=SEAL,
        seed=SEED, demand_day=DAY, future=FUTURE, results=row,
        contract=audit.contract, coverage=audit.coverage, execution=audit.execution)
    write(a.root/'paths'/f'{a.command}.json', result)
    if a.command == 'reference':
        old_root = a.results_root/'protocol_v6_v267_bilateral_factorial'
        name = fixed.key((SEED, DAY, 'fixed_bilateral025'))
        old = read(old_root/'pairs'/f'{name}.json')
        assert old['source'] == '25ed615133'
        for field in ('contract', 'coverage', 'execution'):
            fixed.pilot.joint.assert_same_state(result[field], old[field], 'V271_reference_'+field)
        for metric in (*fixed.METRICS, 'action_counts'):
            fixed.pilot.joint.assert_same_state(row[metric], old['results'][metric], 'V271_cached_'+metric)
        with gzip.open(old_root/'traces'/f'{name}.pkl.gz', 'rb') as stream:
            cached = pickle.load(stream)
        fixed.pilot.joint.assert_same_state({k:trace[k] for k in cached}, cached, 'V271_cached_action_and_timing')
        status = 'reference_cached_parity_passed'
        extra = dict(cached_actions_timing_endpoints_exact=True)
    else:
        old = read(a.root/'paths/reference.json')
        with gzip.open(a.root/'traces/reference.pkl.gz', 'rb') as stream:
            reference = pickle.load(stream)
        fixed.pilot.joint.assert_same_state(trace, reference, 'V271_noop_full_native_actions_timing_queries_uniforms')
        for field in ('results', 'contract', 'coverage', 'execution'):
            fixed.pilot.joint.assert_same_state(result[field], old[field], 'V271_noop_'+field)
        extra = audit.bridge.summary()
        assert extra['queries'] == len(audit.native_queries)
        assert sum(q['evaluations'] > 0 for q in audit.bridge.queries) > 0, 'no query reached the graph evaluator'
        assert extra['pending_locks'] > 0 and extra['model_locks_verified'] > 0, 'no irreversible pending command was modeled'
        with gzip.open(a.root/'traces'/f'graph_queries_{a.command}.pkl.gz', 'xb') as stream:
            pickle.dump(dict(queries=audit.bridge.queries, completed_visits=audit.bridge.prefix.visits), stream)
        status = 'noop_native_equivalence_passed'
    write(a.root/'summary'/f'{a.command}.json', dict(source=a.source, graph_source=GRAPH_SOURCE, status=status,
        physical_slots_consumed=len(list((account/'budget').glob('*.json'))), physical_slot_cap=reservation['physical_slot_cap'],
        native_queries=len(audit.native_queries), actual_commands=len(holds), wall_s=time.perf_counter()-start,
        native_query_latency_s=dict(mean=float(np.mean(audit.native_wall)), p95=float(np.quantile(audit.native_wall,.95)),
                                    max=float(max(audit.native_wall))),
        callback_latency_s=dict(mean=float(np.mean(audit.callback_wall)), p95=float(np.quantile(audit.callback_wall,.95)),
                                max=float(max(audit.callback_wall)), total=float(sum(audit.callback_wall))),
        endpoint_metrics={k:row[k] for k in fixed.KEYS}, **extra,
        raw_traces_remain_server_side=True, training_updates=0, main_policy_changed=False,
        matrix_started=False, automatic_retry=False))
    print('V271_'+a.command.upper()+'_COMPLETE', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('reference', 'noop', 'noop_fix'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    try:
        run(a)
    except Exception as error:
        write(a.root/'failures'/f'{a.command}.json', dict(source=a.source, status='failed_slot_consumed',
            error=repr(error), traceback=traceback.format_exc()[-10000:], automatic_retry=False,
            matrix_started=False, main_policy_changed=False))
        raise


if __name__ == '__main__':
    main()
