#!/usr/bin/env python3
"""Server-only synthetic solver profile, without attaching to traffic processes."""

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time
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
    start = time.perf_counter()
    result = fn()
    return result,time.perf_counter()-start


def load_simulator(path):
    spec = importlib.util.spec_from_file_location('v277_frozen_graph_profile',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module._simulate


def run(a):
    seed = stress_state()
    old = load_simulator(a.frozen_graph) if a.frozen_graph else graph._simulate
    output = dict(source=a.source,synthetic=True,traffic_starts=0,training_updates=0,
        active_traffic_untouched=True,history=len(seed.observed['departures']),fleet=12,
        frozen_graph=str(a.frozen_graph) if a.frozen_graph else graph.__file__)
    cases = {}
    for name,cls in (('v276_indexed',IndexedNativeGraph),('v277_arrival_cv',ArrivalMetricGraph)):
        base,build_s = timed(lambda:cls(seed,seed.observed))
        base.evaluate()  # Warm imports, not part of query search or timings.
        model = CountedModel(base)
        with patch.object(graph,'_simulate',old):
            frozen,frozen_s = timed(lambda:flow.choose(model))
        profile_model = CountedModel(base)
        with patch.object(graph,'_simulate',old):
            profiled,profile = profile_call(a.root/'profiles'/(name+'.prof'),lambda:flow.choose(profile_model))
        assert frozen == profiled
        row = dict(build_s=build_s,unprofiled_solver_s=frozen_s,
            logical_model_calls=sum(model.counts.values()),unique_plans=len(model.counts),
            repeated_exact_plans=sum(n-1 for n in model.counts.values()),
            repeats_by_multiplicity=dict(Counter(model.counts.values())),
            solver_evaluations=frozen['evaluations'],profile=profile,
            selected_action_s=frozen['action_s'],selected_cost=frozen['selected']['cost'],
            selected_cv2=frozen['selected']['cv2'],reference_coverage=frozen['reference']['coverage'])
        if a.compare:
            optimized,optimized_s = timed(lambda:flow.choose(base))
            assert frozen == optimized, 'full solver result changed'
            row.update(optimized_solver_s=optimized_s,speedup=frozen_s/optimized_s,
                       complete_solver_output_exact=True)
        if name == 'v277_arrival_cv':
            current,current_s = timed(lambda:choose_current(base))
            row['current_only'] = dict(solver_s=current_s,evaluations=current['evaluations'],
                action_s=current['action_s'],cost=current['selected']['cost'])
        cases[name] = row
        print('V277_PROFILE_CASE_COMPLETE',name,flush=True)
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
    p.add_argument('--compare',action='store_true')
    run(p.parse_args())
