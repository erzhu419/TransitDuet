"""One detached complete-day check against the canonical archived V276 path."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import time
import traceback
import unittest

from fast_continuation import FastNativeGraph
import freqduet_flow_controller as flow
from freqduet_consistent_flow import ConsistentFlowController
from freqduet_indexed_continuation import IndexedContinuation, IndexedNativeGraph
from profile_freqduet_v277_solver import load_module
import run_freqduet_v267_bilateral_factorial as fixed
import run_freqduet_v276_native_continuation as previous


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def graph_record(row):
    return {k:v for k,v in row.items() if k != 'total_s'}


def exact(left, right, label):
    fixed.pilot.joint.assert_same_state(left, right, label)


def reference(a):
    row = json.loads((a.results/'protocol_v6_v276_indexed_replacements/pairs/p8.json').read_text())
    with gzip.open(a.results/'protocol_v6_v276_indexed_replacements/traces/p8.pkl.gz', 'rb') as stream:
        trace = pickle.load(stream)
    assert row['day'] == 2076001 and row['arm'] == previous.ARMS[2]
    assert row['execution']['done_reason'] == 'service_cleared'
    assert len(trace['queries']) == len(trace['holds']) == 5200
    assert row['graph']['supported_queries'] == 3499
    return row, trace


def preflight(a):
    p = json.loads((Path(__file__).parent/'protocol.json').read_text())
    assert p['physical_start_cap'] == 1 and not p['automatic_retry']
    assert a.root.parent == a.results, 'runtime proof must remain in the shared results parent'
    row, trace = reference(a)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_fast_continuation.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun == 7
    _, _, _, _ = fixed.pilot.joint.optimized_runtime(a.root)
    joint = fixed.pilot.joint
    checkpoints = joint.TRAIN_ROOT/'logs'/f'{joint.CURRENT}_seed12011'/'checkpoints'
    files = [checkpoints/f'{component}_ep{joint.BASE_EP}.pt' for component in ('runner', 'lower', 'upper')]
    assert all(path.is_file() for path in files), files
    old = a.results/'protocol_v6_v276_native_continuation/source_7023e2e784/FreqDuet/freqduet/scripts'
    assert (old/'freqduet_causal_event_graph.py').is_file()
    assert (old/'freqduet_flow_controller.py').is_file()
    write(a.root/'preflight/passed.json', dict(protocol=p, tests=result.testsRun,
        reference_day=row['day'], queries=len(trace['queries']), supported_queries=3499,
        checkpoints_present=[str(path) for path in files], traffic_starts=0, training_updates=0))
    print('DETACHED_FULL_DAY_PREFLIGHT_COMPLETE', flush=True)


def make_audit(a, credit, oldtrace):
    old_source = a.results/'protocol_v6_v276_native_continuation/source_7023e2e784/FreqDuet/freqduet/scripts'
    old_kernel = load_module(old_source/'freqduet_causal_event_graph.py', 'full_day_frozen_kernel')._simulate
    old_choose = load_module(old_source/'freqduet_flow_controller.py', 'full_day_frozen_search').choose

    class FrozenGraph(IndexedNativeGraph):
        def evaluate(self, commands=None, homogeneous=False):
            return old_kernel(self, commands or {}, homogeneous, IndexedContinuation(self))

    class CandidateController(ConsistentFlowController):
        def __init__(self, *args):
            super().__init__(*args)
            self.real_state_check = dict(passed=False, logical_evaluations=0, extra_reference_wall_s=0.)

        def build_model(self, value, visits):
            base, info = super().build_model(value, visits)
            return FastNativeGraph(base, base.observed), info

        def select_plan(self, model):
            result = flow.choose(model, self.homogeneous, self.proposals)
            if not self.real_state_check['passed'] and result['evaluations'] > 1:
                begin = time.perf_counter()
                original = old_choose(FrozenGraph(model, model.observed), self.homogeneous, self.proposals)
                self.real_state_check['extra_reference_wall_s'] += time.perf_counter() - begin
                exact(result, original, 'first_real_state_complete_solver')
                self.real_state_check.update(passed=True, logical_evaluations=result['evaluations'])
                print('DETACHED_REAL_STATE_EXACT', json.dumps(self.real_state_check), flush=True)
            return result

    Base = type(previous.make_audit(fixed, previous.ARMS[2], credit, False))

    class Audit(Base):
        def attach(self, runner):
            fixed.BenchmarkAudit.attach(self, runner)
            from env.bus import Bus
            self.bridge = CandidateController(self.env, self.launches, credit)
            self.bridge.attach(Bus)
            self.started = time.perf_counter()

        def __call__(self, *args):
            law = super().__call__(*args)
            i = len(self.decisions) - 1
            exact(self.decisions[-1], oldtrace['queries'][i], f'query_before_execution_{i}')
            exact(graph_record(self.bridge.queries[-1]), graph_record(oldtrace['graph_queries'][i]),
                  f'graph_decision_before_execution_{i}')
            if len(self.decisions) % 100 == 0:
                print('SCHEDULEURM_PROGRESS ' + json.dumps(dict(phase='detached_complete_day',
                    current=len(self.decisions), total=5200, unit='matched_queries',
                    sim_s=float(self.env.current_time), elapsed_s=time.perf_counter()-self.started)), flush=True)
            return law

    return Audit()


def replay(a):
    assert (a.root/'preflight/passed.json').is_file()
    before, oldtrace = reference(a)
    write(a.root/'started/path.json', dict(day=2076001, actor=12011, sampler=27011001,
        physical_start=1, cap=1, automatic_retry=False))
    begin = time.perf_counter()
    credit = fixed.pilot.CreditObserver()
    audit = make_audit(a, credit, oldtrace)
    try:
        row, uniforms, _, holds = fixed.pilot.rollout(a.root, 12011, 2076001, 'stochastic', 27011001,
            [0.]*4, 'detached_exact_p8', probability_rule=audit, trace_observer=audit, credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]), holding_person_s=audit.holding_person_s,
            **audit.dispatch, **audit.delays)
        previous.label_resources(audit.bridge, audit.dispatches, holds)
        trace = dict(holds=holds, dispatches=audit.dispatches, arrivals=audit.arrivals, departures=audit.departures,
            queries=audit.decisions, uniforms=uniforms, graph_queries=audit.bridge.queries,
            promise_records=audit.bridge.promise_records, resource_predictions=audit.bridge.predictions)
        for key, value in trace.items():
            if key == 'graph_queries':
                exact([graph_record(r) for r in value], [graph_record(r) for r in oldtrace[key]], key)
            else:
                exact(value, oldtrace[key], key)
        for key, value in dict(results=row, contract=audit.contract, coverage=audit.coverage,
                               execution=audit.execution, action_chain=audit.chain).items():
            exact(value, before[key], key)
        assert audit.bridge.real_state_check['passed']
        destination = a.root/'traces/candidate.pkl.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, 'xb') as stream:
            pickle.dump(trace, stream, pickle.HIGHEST_PROTOCOL)
        wall = time.perf_counter() - begin
        extra = audit.bridge.real_state_check['extra_reference_wall_s']
        measured = audit.bridge.summary()
        old_overhead = before['graph']['kernel_equivalence_audit']['extra_reference_wall_s']
        write(a.root/'summary/results.json', dict(status='complete_day_exact_equivalence_passed',
            traffic_starts=1, training_updates=0, main_policy_changed=False,
            old_code_changed=False, running_tasks_changed=False, queries=len(audit.decisions),
            reference='protocol_v6_v276_indexed_replacements/pairs/p8.json', day=2076001,
            endpoints_exact=True, all_saved_trace_records_exact=True,
            real_state_check=audit.bridge.real_state_check, results=row,
            candidate_wall_s=wall, candidate_wall_minus_reference_check_s=wall-extra,
            reference_archived_wall_s=before['wall_s'], reference_archived_check_s=old_overhead,
            descriptive_archived_wall_speedup=(before['wall_s']-old_overhead)/(wall-extra),
            callback_latency_s=measured['callback_latency_s'],
            limits='One archived day; archived wall time is not a simultaneous hardware-controlled benchmark.'))
        print('DETACHED_FULL_DAY_EXACT_COMPLETE', flush=True)
    except Exception as error:
        write(a.root/'summary/failure.json', dict(status='failed_one_slot_no_retry', error=repr(error),
            traceback=traceback.format_exc()[-5000:], physical_starts=1, automatic_retry=False,
            matched_queries=len(getattr(audit, 'decisions', [])), main_policy_changed=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'replay', 'all'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'all':
        preflight(args)
        replay(args)
    else:
        {'preflight':preflight, 'replay':replay}[args.phase](args)
