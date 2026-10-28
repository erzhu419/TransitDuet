from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import freqduet_causal_event_graph as graph
import freqduet_flow_controller as flow
from freqduet_arrival_metric_control import ArrivalMetricGraph
from freqduet_indexed_continuation import IndexedNativeGraph
from profile_freqduet_v277_solver import load_module
from test_v276_native_continuation import model_fixture


class EfficiencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_sim = staticmethod(load_module(Path(os.environ['FREQDUET_PROFILE_FROZEN_GRAPH']),
                                             'efficiency_frozen_graph')._simulate)
        cls.old_flow = load_module(Path(os.environ['FREQDUET_PROFILE_FROZEN_FLOW']),
                                  'efficiency_frozen_flow')

    def fixture(self):
        base,native = model_fixture()
        arrivals = list(native.observed['arrivals'])
        for v in native.snapshot.vehicles:
            if v.arrival_s is not None:
                arrivals.append((v.direction,native.snapshot.routes[v.direction][v.index].station,
                                 v.trip,v.arrival_s,v.bus))
        native.observed = dict(native.observed,arrivals=sorted(arrivals,key=lambda r:(r[3],r[4])))
        return base,native

    def assert_model_exact(self,model,commands,homogeneous=False):
        with patch.object(graph,'_simulate',self.old_sim):
            before = model.evaluate(commands,homogeneous)
        self.assertEqual(before,model.evaluate(commands,homogeneous))

    def test_all_seven_bins_and_two_stop_transfers_keep_every_model_field(self):
        base,native = self.fixture()
        for model in (base,IndexedNativeGraph(native,native.observed),
                      ArrivalMetricGraph(native,native.observed)):
            for first in graph.ACTIONS:
                for second in graph.ACTIONS:
                    self.assert_model_exact(model,{(0,0,0):int(first),(0,0,1):int(second)})

    def test_committed_holding_and_homogeneous_cost_keep_every_model_field(self):
        _,native = self.fixture()
        native.snapshot = replace(native.snapshot,vehicles=tuple(
            replace(v,phase='hold',remaining_s=12.,arrival_s=1990.,boarded=0.) if v.bus==2 else v
            for v in native.snapshot.vehicles))
        for model in (IndexedNativeGraph(native,native.observed),
                      ArrivalMetricGraph(native,native.observed)):
            for homogeneous in (False,True):
                for action in (0,30,45):
                    self.assert_model_exact(model,{(0,0,0):action},homogeneous)

    def test_complete_solver_matches_frozen_function_and_same_logical_budget(self):
        base,native = self.fixture()
        for model in (base,IndexedNativeGraph(native,native.observed),
                      ArrivalMetricGraph(native,native.observed)):
            for homogeneous in (False,True):
                with patch.object(graph,'_simulate',self.old_sim):
                    before = self.old_flow.choose(model,homogeneous)
                self.assertEqual(before,flow.choose(model,homogeneous))

    def test_identical_plan_reuses_kernel_without_adding_budget(self):
        model = PlanningTable()
        wrapped = flow.FallbackGraph(model)
        wrapped.evaluate()
        request = {(0,0,0):0}
        first = wrapped.evaluate(request)
        kernels = model.calls
        self.assertEqual(first,wrapped.evaluate(dict(request)))
        self.assertEqual(model.calls,kernels)
        self.assertEqual(wrapped.calls,4)
        with patch.object(graph,'MAX_EVALUATIONS',4):
            self.assertEqual(wrapped.evaluate(request)['failure'],'query_evaluation_budget')
        self.assertEqual(model.calls,kernels)

    def test_caches_are_query_local_and_homogeneous_mode_does_not_cross(self):
        model = PlanningTable()
        a = flow.FallbackGraph(model).evaluate()
        model.offset = 10.
        b = flow.FallbackGraph(model).evaluate()
        c = flow.FallbackGraph(model,True).evaluate(homogeneous=True)
        self.assertEqual(b['cost']-a['cost'],10.)
        self.assertEqual(c['cost']-b['cost'],100.)

    def test_non_native_optimum_stays_identical_with_warm_start(self):
        model = PlanningTable()
        incumbent = {(0,0,1):10}
        expected = self.old_flow.choose(model,proposals=incumbent)
        model.calls = 0
        actual = flow.choose(model,proposals=incumbent)
        self.assertEqual(expected,actual)
        self.assertEqual(actual['action_s'],0)
        self.assertLess(model.calls,actual['evaluations'])


class PlanningTable:
    def __init__(self):
        self.calls,self.offset = 0,0.
        self.snapshot = SimpleNamespace(current_bus=0,native_action_s=30,
            vehicles=(graph.Vehicle(0,0,False,0,'prepare',20.,arrival_s=1990.,boarded=0.),),
            routes={False:(graph.Stop(0),graph.Stop(1))})

    def evaluate(self,commands=None,homogeneous=False):
        self.calls += 1
        commands = commands or {}
        actions = {(0,0,0):(2000.,commands.get((0,0,0),30)),
                   (0,0,1):(2100.,commands.get((0,0,1),0))}
        return dict(cost=1000.+sum(v[1] for v in actions.values())+self.offset+100*homogeneous,
            actions=actions,legal={key:tuple(graph.ACTIONS) for key in actions},
            cv2=.1,returns={0:3000.},next_launch={0:3001.},delays={10:0.},
            missing_commands=set(),coverage=True,failure=None)


if __name__ == '__main__':
    unittest.main()
