#!/usr/bin/env python3
"""V274 passive entry and synthetic admission; no real start without a release."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys
import time
import traceback
import unittest

from freqduet_v274_passive_flow import FlowRecorder, compare_legacy, validate_flow
from seal_freqduet_v274_flow_equivalence import REFERENCE, OLD_SOURCE, read, write


SLOT = 'flow_passive_c0'


def require_release(root, source):
    sealed = read(root/'protocol.json')
    assert sealed['source'] == 'd43947ed64'
    assert sealed['budget']['slot_ids'] == [SLOT] and sealed['budget']['physical_start_cap'] == 1
    release = read(root/'budget'/f'{SLOT}.json')
    assert release['source'] == source and release['slot'] == SLOT
    assert release['context'] == sealed['context'] and release['automatic_retry'] is False
    checks = read(root/'entry_checks/status.json')
    assert checks['source'] == source and checks['status'] == 'synthetic_passive_entry_passed'
    write(root/'started'/f'{SLOT}.json', dict(source=source, slot=SLOT, automatic_retry=False))


def make_audit(fixed, credit):
    import numpy as np
    from freqduet_native_event_graph import NativeNoop, native_query

    class Audit(fixed.BenchmarkAudit):
        def __init__(self):
            super().__init__(12011, 'fixed_bilateral025')
            self.flow = FlowRecorder()
            self.native_queries, self.native_wall, self.flow_wall, self.callback_wall = [], [], [], []

        def attach(self, runner):
            super().attach(runner)
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
            query = native_query(self.env,bus,legal,action,credit.stream.pending_commands)
            self.native_queries.append(query)
            self.bridge.observe(bus,legal,action)
            flow_start = time.perf_counter()
            self.flow.capture(self.bridge.prefix,query)
            self.flow_wall.append(time.perf_counter()-flow_start)
            self.callback_wall.append(time.perf_counter()-start)
            return law

        def detach(self):
            self.bridge.detach()
            super().detach()

    return Audit()


def load_trace(path):
    with gzip.open(path,'rb') as stream:
        return pickle.load(stream)


def save_trace(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'xb') as stream:
        pickle.dump(value,stream,protocol=pickle.HIGHEST_PROTOCOL)


def persist_timings(root, source, audit):
    import numpy as np
    def latency(values):
        return dict(mean=float(np.mean(values)),p95=float(np.quantile(values,.95)),
                    max=float(max(values)),total=float(sum(values)))
    values = dict(native_latency_s=latency(audit.native_wall),flow_latency_s=latency(audit.flow_wall),
                  callback_latency_s=latency(audit.callback_wall))
    write(root/'timings'/f'{SLOT}.json',dict(source=source,**values))
    return values


def passive(a):
    require_release(a.out_root,a.source)
    try:
        import run_freqduet_v267_bilateral_factorial as fixed
        credit = fixed.pilot.CreditObserver()
        audit = make_audit(fixed,credit)
        start = time.perf_counter()
        row,uniforms,_,holds = fixed.pilot.rollout(a.out_root,12011,2071001,'stochastic',27011001,
            [0.]*4,'v274_'+SLOT,probability_rule=audit,trace_observer=audit,credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
                   backlog_trip_s=float(credit.stream.total[4]),holding_person_s=audit.holding_person_s,
                   **audit.dispatch,**audit.delays)
        result = dict(source=a.source,results=row,contract=audit.contract,
                      coverage=audit.coverage,execution=audit.execution)
        native = dict(holds=holds,dispatches=audit.dispatches,arrivals=audit.arrivals,
                      departures=audit.departures,queries=audit.native_queries,uniforms=uniforms)
        graph = dict(queries=audit.bridge.queries,completed_visits=audit.bridge.prefix.visits)
        flow = audit.flow.bundle(audit.bridge.prefix,audit.bridge.routes)
        save_trace(a.out_root/'traces'/f'{SLOT}.pkl.gz',native)
        save_trace(a.out_root/'traces'/f'graph_queries_{SLOT}.pkl.gz',graph)
        save_trace(a.out_root/'traces'/f'flow_{SLOT}.pkl.gz',flow)
        write(a.out_root/'paths'/f'{SLOT}.json',result)
        timings = persist_timings(a.out_root,a.source,audit)
        old = a.results_root/REFERENCE
        old_result = read(old/'paths/noop_fix.json')
        assert old_result['source'] == OLD_SOURCE
        same = fixed.pilot.joint.assert_same_state
        compare_legacy(native,result,graph,load_trace(old/'traces/noop_fix.pkl.gz'),old_result,
                       load_trace(old/'traces/graph_queries_noop_fix.pkl.gz'),same)
        validation_start = time.perf_counter()
        flow_gate = validate_flow(flow,native,graph,same)
        same((flow_gate['cases']['first_arrivals'],flow_gate['completed_visits']), (42,5460), 'V274_visit_counts')
        assert len(native['queries']) == len(holds) == 5200
        assert audit.execution['commanded_s'] == audit.execution['executed_s'] == 38930
        assert audit.execution['physical_vehicles'] == 12 and audit.execution['done_reason'] == 'service_cleared'
        write(a.out_root/'summary/passive.json',dict(source=a.source,status='native_passive_equivalence_passed',
            flow=flow_gate,**timings,wall_s=time.perf_counter()-start,
            flow_validation_wall_s=time.perf_counter()-validation_start,
            physical_slots_consumed=1,physical_start_cap=1,automatic_retry=False,
            training_updates=0,main_policy_changed=False,controller_admitted=False,
            raw_records_remain_server_side=True))
    except Exception as error:
        write(a.out_root/'failures'/f'{SLOT}.json',dict(source=a.source,status='failed_slot_consumed',
            error=repr(error),traceback=traceback.format_exc()[-6000:],automatic_retry=False))
        raise
    print('V274_PASSIVE_EQUIVALENCE_COMPLETE',flush=True)


def check(a):
    suite = unittest.TestSuite(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'),pattern=pattern) for pattern in
        ('test_v274_passive_flow.py','test_v273_flow_contract.py','test_v271_native_event_graph.py'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.testsRun > 0 and result.wasSuccessful()
    assert 'runner_v3' not in sys.modules and 'run_freqduet_v267_bilateral_factorial' not in sys.modules
    write(a.out_root/'entry_checks/status.json',dict(source=a.source,status='synthetic_passive_entry_passed',
        tests=result.testsRun,collector_entry_implemented=True,passive_equivalence_established=False,
        real_starts_released=0,physical_slots_consumed=0,raw_trace_reads=0,
        training_updates=0,main_policy_changed=False,controller_admitted=False))
    print('V274_PASSIVE_ENTRY_SYNTHETIC_COMPLETE',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('check','passive'))
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--source',required=True)
    a = p.parse_args()
    (check if a.command == 'check' else passive)(a)


if __name__ == '__main__':
    main()
