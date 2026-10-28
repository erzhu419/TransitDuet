"""One detached replay of archived V277 p17; no new policy or demand day."""

import argparse
import gzip
import json
from pathlib import Path
import pickle
import time
import traceback
import unittest

import freqduet_causal_event_graph as graph
from freqduet_arrival_metric_control import ArrivalMetricGraph, CurrentActionController, choose_current
from freqduet_flow_controller import FlowController
from freqduet_indexed_continuation import IndexedContinuation, IndexedNativeGraph
from freqduet_native_continuation import inputs
from fast_continuation import FastArrivalMetricGraph
from profile_freqduet_v277_solver import load_module
from run_full_day import exact, graph_record, write
import run_freqduet_v267_bilateral_factorial as fixed
import run_freqduet_v276_native_continuation as previous


def read_reference(a):
    root = a.results/'protocol_v6_v277_metric_alignment/matrix_repaired'
    row = json.loads((root/'pairs/p17.json').read_text())
    with gzip.open(root/'traces/p17.pkl.gz', 'rb') as stream:
        trace = pickle.load(stream)
    assert (row['source'], row['day'], row['arm']) == ('b9909ddc03', 2077002, 'arrival_cv_current')
    assert row['execution']['done_reason'] == 'service_cleared'
    assert len(trace['queries']) == len(trace['holds']) == 5200
    assert row['graph']['supported_queries'] == 3315
    return row, trace


def preflight(a):
    protocol = json.loads((Path(__file__).parent/'protocol.json').read_text())
    assert protocol['physical_start_cap'] == 1 and protocol['automatic_retry'] is False
    assert a.root.parent == a.results
    row, trace = read_reference(a)
    tests = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_fast_continuation.py')
    checked = unittest.TextTestRunner(verbosity=2).run(tests)
    assert checked.wasSuccessful() and checked.testsRun == 7
    fixed.pilot.joint.optimized_runtime(a.root)
    joint = fixed.pilot.joint
    checkpoints = joint.TRAIN_ROOT/'logs'/f'{joint.CURRENT}_seed12011'/'checkpoints'
    assert all((checkpoints/f'{part}_ep{joint.BASE_EP}.pt').is_file() for part in ('runner', 'lower', 'upper'))
    source = a.results/'protocol_v6_v276_native_continuation/source_7023e2e784/FreqDuet/freqduet/scripts'
    assert (source/'freqduet_causal_event_graph.py').is_file()
    write(a.root/'preflight/passed.json', dict(protocol=protocol, tests=checked.testsRun,
        reference_queries=len(trace['queries']), reference_supported=row['graph']['supported_queries'],
        traffic_starts=0, training_updates=0))
    print('CURRENT_EXACT_PREFLIGHT_COMPLETE', flush=True)


def make_audit(a, credit, oldtrace):
    old_path = a.results/'protocol_v6_v276_native_continuation/source_7023e2e784/FreqDuet/freqduet/scripts/freqduet_causal_event_graph.py'
    old_kernel = load_module(old_path, 'current_exact_frozen_kernel')._simulate

    class FrozenKernelGraph(IndexedNativeGraph):
        def evaluate(self, commands=None, homogeneous=False):
            return old_kernel(self, commands or {}, homogeneous, IndexedContinuation(self))

    class FrozenArrivalGraph(ArrivalMetricGraph, FrozenKernelGraph):
        pass

    class FastCurrentController(CurrentActionController):
        def __init__(self, *args):
            super().__init__(*args)
            self.real_state_check = dict(passed=False, evaluations=0, extra_reference_wall_s=0.)

        def build_model(self, value, visits):
            base, info = FlowController.build_model(self, value, visits)
            model = FastArrivalMetricGraph(base, inputs(self.env, self.prefix, self.evidence))
            recorded = self.env.headway_events.headways_s
            observed = model.arrival_prefix.moments
            assert observed.count == len(recorded)
            assert observed.total == sum(recorded)
            assert observed.square_total == sum(h*h for h in recorded)
            return model, info

        def select_plan(self, model):
            result = super().select_plan(model)
            if not self.real_state_check['passed'] and result['evaluations'] > 1:
                begin = time.perf_counter()
                expected = choose_current(FrozenArrivalGraph(model, model.observed), self.homogeneous)
                self.real_state_check['extra_reference_wall_s'] += time.perf_counter()-begin
                exact(result, expected, 'first_supported_current_complete_solver')
                assert result['evaluations'] <= 7
                self.real_state_check.update(passed=True, evaluations=result['evaluations'])
                print('CURRENT_REAL_STATE_EXACT', json.dumps(self.real_state_check), flush=True)
            return result

    Base = type(previous.make_audit(fixed, previous.ARMS[0], credit, False))

    class Audit(Base):
        def attach(self, runner):
            fixed.BenchmarkAudit.attach(self, runner)
            from env.bus import Bus
            self.bridge = FastCurrentController(self.env, self.launches, credit)
            self.bridge.attach(Bus)
            self.started = time.perf_counter()

        def __call__(self, *args):
            law = super().__call__(*args)
            i = len(self.decisions)-1
            exact(self.decisions[-1], oldtrace['queries'][i], f'query_before_execution_{i}')
            exact(graph_record(self.bridge.queries[-1]), graph_record(oldtrace['graph_queries'][i]),
                  f'graph_decision_before_execution_{i}')
            if len(self.decisions) % 100 == 0:
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='current_exact_complete_day',
                    current=len(self.decisions), total=5200, unit='matched_queries',
                    sim_s=float(self.env.current_time), elapsed_s=time.perf_counter()-self.started)), flush=True)
            return law

    return Audit()


def replay(a):
    assert (a.root/'preflight/passed.json').is_file()
    before, oldtrace = read_reference(a)
    write(a.root/'started/path.json', dict(day=2077002, actor=12011, sampler=27011001,
        physical_start=1, cap=1, automatic_retry=False))
    begin = time.perf_counter()
    credit = fixed.pilot.CreditObserver()
    audit = make_audit(a, credit, oldtrace)
    try:
        row, uniforms, _, holds = fixed.pilot.rollout(a.root, 12011, 2077002, 'stochastic', 27011001,
            [0.]*4, 'detached_current_p17', probability_rule=audit, trace_observer=audit, credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]), onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]), holding_person_s=audit.holding_person_s,
            **audit.dispatch, **audit.delays)
        previous.label_resources(audit.bridge, audit.dispatches, holds)
        trace = dict(holds=holds, dispatches=audit.dispatches, arrivals=audit.arrivals, departures=audit.departures,
            queries=audit.decisions, uniforms=uniforms, graph_queries=audit.bridge.queries,
            resource_predictions=audit.bridge.predictions)
        assert trace.keys() == oldtrace.keys()
        for key, value in trace.items():
            if key == 'graph_queries':
                exact([graph_record(r) for r in value], [graph_record(r) for r in oldtrace[key]], key)
            else:
                exact(value, oldtrace[key], key)
        for key, value in dict(results=row, contract=audit.contract, coverage=audit.coverage,
                               execution=audit.execution, action_chain=audit.chain).items():
            exact(value, before[key], key)
        assert audit.bridge.real_state_check['passed']
        wall = time.perf_counter()-begin
        extra = audit.bridge.real_state_check['extra_reference_wall_s']
        core = lambda r:{k:v for k,v in r.items() if k != 'callback_latency_s'}
        measured = audit.bridge.summary()
        exact(core(measured), core(before['graph']), 'complete_graph_summary')
        destination = a.root/'traces/candidate.pkl.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(destination, 'xb') as stream:
            pickle.dump(trace, stream, pickle.HIGHEST_PROTOCOL)
        write(a.root/'summary/results.json', dict(status='current_complete_day_exact_equivalence_passed',
            reference='protocol_v6_v277_metric_alignment/matrix_repaired/pairs/p17.json', day=2077002,
            queries=len(audit.decisions), endpoints_exact=True, all_saved_trace_records_exact=True,
            real_state_check=audit.bridge.real_state_check, results=row, candidate_wall_s=wall,
            candidate_wall_minus_reference_check_s=wall-extra, reference_archived_wall_s=before['wall_s'],
            descriptive_archived_wall_speedup=before['wall_s']/(wall-extra),
            callback_latency_s=measured['callback_latency_s'], traffic_starts=1, training_updates=0,
            old_code_changed=False, running_matrix_changed=False, main_policy_changed=False,
            limits='One archived current-action path; engineering parity only, not eligibility or a four-day traffic result.'))
        print('CURRENT_EXACT_COMPLETE_DAY_PASSED', flush=True)
    except Exception as error:
        write(a.root/'summary/failure.json', dict(status='failed_one_slot_no_retry', error=repr(error),
            traceback=traceback.format_exc()[-5000:], matched_queries=len(getattr(audit, 'decisions', [])),
            physical_starts=1, automatic_retry=False, main_policy_changed=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    preflight(args)
    replay(args)
