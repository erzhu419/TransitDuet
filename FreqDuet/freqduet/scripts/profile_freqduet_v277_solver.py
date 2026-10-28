#!/usr/bin/env python3
"""Server-only synthetic solver profile, without attaching to traffic processes."""

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import freqduet_causal_event_graph as graph
import freqduet_flow_controller as flow
from freqduet_arrival_metric_control import ArrivalMetricGraph, choose_current
from freqduet_indexed_continuation import IndexedNativeGraph
from check_freqduet_v276_indexed import profile_call, stress_state


class CountedModel:
    def __init__(self,base):
        self.base,self.snapshot = base,base.snapshot
        self.counts = Counter()

    def evaluate(self,commands=None,homogeneous=False):
        self.counts[bool(homogeneous),tuple(sorted((commands or {}).items()))] += 1
        return self.base.evaluate(commands,homogeneous)


def timed(fn):
    start,cpu = time.perf_counter(),time.process_time()
    result = fn()
    return result,time.perf_counter()-start,time.process_time()-cpu


def load_module(path,name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(a):
    test_count = 0
    if a.compare:
        assert a.frozen_graph and a.frozen_flow
        os.environ['FREQDUET_PROFILE_FROZEN_GRAPH'] = str(a.frozen_graph)
        os.environ['FREQDUET_PROFILE_FROZEN_FLOW'] = str(a.frozen_flow)
        tests = Path(__file__).resolve().parents[1]/'tests'
        patterns = ('test_v270_causal_event_graph.py','test_v275_flow_controller.py',
                    'test_v277_efficiency.py','test_v276_indexed_continuation.py',
                    'test_v277_arrival_metric.py','test_v277_matrix.py')
        suite = unittest.TestSuite(unittest.defaultTestLoader.discover(str(tests),pattern=p,
            top_level_dir=str(tests)) for p in patterns)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        assert result.wasSuccessful() and result.testsRun >= 34
        test_count = result.testsRun
    old = load_module(a.frozen_graph,'v277_frozen_graph_profile')._simulate if a.frozen_graph else graph._simulate
    frozen_choose = load_module(a.frozen_flow,'v277_frozen_flow_profile').choose if a.frozen_flow else flow.choose
    output = dict(source=a.source,synthetic=True,traffic_starts=0,training_updates=0,
        active_traffic_untouched=True,fleet=12,tests=test_count,
        frozen_graph=str(a.frozen_graph) if a.frozen_graph else graph.__file__)
    cases = {}
    workloads = dict(late_regression=stress_state(),
        midday_full_roster=stress_state(history_count=2600,now_s=24000.,plans_per_direction=13))
    for workload,seed in workloads.items():
        for name,cls in (('v276_indexed',IndexedNativeGraph),('v277_arrival_cv',ArrivalMetricGraph)):
            label = workload+'_'+name
            base,build_s,_ = timed(lambda:cls(seed,seed.observed))
            baseline = base.evaluate()
            assert baseline['coverage'] and not baseline['failure'], 'benchmark took an unsupported shortcut'
            model = CountedModel(base)
            with patch.object(graph,'_simulate',old):
                frozen,frozen_s,frozen_cpu = timed(lambda:frozen_choose(model))
            profile_model = CountedModel(base)
            with patch.object(graph,'_simulate',old):
                profiled,profile = profile_call(a.root/'profiles'/(label+'.prof'),lambda:frozen_choose(profile_model))
            assert frozen == profiled
            row = dict(build_s=build_s,unprofiled_solver_s=frozen_s,unprofiled_cpu_s=frozen_cpu,
                history=len(seed.observed['departures']),published_plans=len(seed.snapshot.plans),
                kernel_calls=sum(model.counts.values()),unique_plans=len(model.counts),
                repeated_exact_plans=sum(n-1 for n in model.counts.values()),
                solver_evaluations=frozen['evaluations'],profile=profile,
                selected_action_s=frozen['action_s'],selected_cost=frozen['selected']['cost'],
                selected_cv2=frozen['selected']['cv2'],reference_coverage=True)
            if a.compare:
                counted = CountedModel(base)
                optimized,optimized_s,optimized_cpu = timed(lambda:flow.choose(counted))
                assert frozen == optimized, 'full solver result changed'
                checked,optimized_profile = profile_call(a.root/'profiles'/(label+'_optimized.prof'),lambda:flow.choose(base))
                assert checked == frozen
                row.update(optimized_solver_s=optimized_s,optimized_cpu_s=optimized_cpu,
                    optimized_kernel_calls=sum(counted.counts.values()),speedup=frozen_s/optimized_s,
                    optimized_profile=optimized_profile,complete_solver_output_exact=True)
            if name == 'v277_arrival_cv':
                current,current_s,current_cpu = timed(lambda:choose_current(base))
                with patch.object(graph,'_simulate',old):
                    before,before_s,_ = timed(lambda:choose_current(base))
                assert before == current
                row['current_only'] = dict(solver_s=current_s,cpu_s=current_cpu,
                    frozen_solver_s=before_s,evaluations=current['evaluations'],
                    action_s=current['action_s'],cost=current['selected']['cost'],complete_output_exact=True)
            cases[label] = row
            print('V277_PROFILE_CASE_COMPLETE',label,flush=True)
    output['cases'] = cases
    a.root.joinpath('summary').mkdir(parents=True,exist_ok=True)
    with (a.root/'summary/results.json').open('x') as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print('V277_SOLVER_PROFILE_COMPLETE',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--frozen-graph',type=Path)
    p.add_argument('--frozen-flow',type=Path)
    p.add_argument('--compare',action='store_true')
    run(p.parse_args())
