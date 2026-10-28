from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

import freqduet_causal_event_graph as graph
import freqduet_flow_controller as flow
from freqduet_arrival_metric_control import ArrivalMetricGraph, ArrivalPrefix, choose_current
from env.evaluation import HeadwayEventRecorder
from test_v276_native_continuation import model_fixture


def routes():
    return {False:(graph.Stop(0),graph.Stop(1,False,True)),
            True:(graph.Stop(2),graph.Stop(3,False,True))}


def recorder_cv(routes, prefix, future):
    recorder = HeadwayEventRecorder()
    allowed = {(d,s.station) for d,rs in routes.items() for s in rs if not s.terminal}
    for direction,station,trip,time,_ in prefix:
        if (direction,station) in allowed:
            recorder.record(station,direction,time,trip)
    for (direction,station), times in future.items():
        if (direction,station) in allowed:
            for trip,time in enumerate(sorted(times),10000):
                recorder.record(station,direction,time,trip)
    return recorder.summary()['headway_cv']**2, len(recorder.headways_s)


class ArrivalMetricTests(unittest.TestCase):
    def test_all_history_and_zero_headway_match_actual_recorder(self):
        arrivals = [(False,0,i,t,0) for i,t in enumerate((10.,20.,20.,25.,60.))]
        prefix = ArrivalPrefix(routes(),arrivals,60.)
        expected,count = recorder_cv(routes(),arrivals,{})
        self.assertEqual(prefix.moments.count,count)
        self.assertAlmostEqual(prefix.moments.cv_squared,expected,places=14)

    def test_terminal_arrivals_never_enter_regularization(self):
        arrivals = [(False,0,0,10.,0),(False,0,1,20.,1),
                    (False,1,0,30.,0),(False,1,1,1000.,1)]
        prefix = ArrivalPrefix(routes(),arrivals,1000.)
        self.assertEqual(prefix.moments.count,1)
        self.assertEqual(prefix.extend({(False,1):[1001.,9000.]}).cv_squared,0.)

    def test_arrived_but_unfinished_current_visit_is_already_history(self):
        arrivals = [(False,0,0,10.,0),(False,0,1,60.,1)]
        prefix = ArrivalPrefix(routes(),arrivals,60.)
        projected = prefix.extend({(False,0):[60.,90.]})
        expected,count = recorder_cv(routes(),arrivals,{(False,0):[60.,90.]})
        self.assertEqual(projected.count,count)
        self.assertEqual(count,3)
        self.assertAlmostEqual(projected.cv_squared,expected,places=14)

    def test_first_arrival_creates_no_gap_and_same_tick_distinct_arrivals_do(self):
        prefix = ArrivalPrefix(routes(),[],20.)
        moments = prefix.extend({(False,0):[20.,20.,30.]})
        self.assertEqual(moments.count,2)
        self.assertEqual(moments.cv_squared,1.)

    def test_candidates_do_not_mutate_prefix_or_each_other(self):
        prefix = ArrivalPrefix(routes(),[(False,0,0,10.,0),(False,0,1,20.,1)],20.)
        a = prefix.extend({(False,0):[40.,70.]}).cv_squared
        prefix.extend({(False,0):[21.,22.]})
        self.assertEqual(prefix.extend({(False,0):[40.,70.]}).cv_squared,a)
        self.assertEqual(prefix.moments.count,1)
        self.assertEqual(prefix.last[False,0],20.)

    def test_future_history_and_past_projection_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'beyond'):
            ArrivalPrefix(routes(),[(False,0,0,21.,0)],20.)
        with self.assertRaisesRegex(ValueError,'precedes'):
            ArrivalPrefix(routes(),[],20.).extend({(False,0):[19.]})

    def model(self):
        _,base = model_fixture()
        arrivals = list(base.observed['arrivals'])
        for v in base.snapshot.vehicles:
            if v.arrival_s is not None:
                station = base.snapshot.routes[v.direction][v.index].station
                arrivals.append((v.direction,station,v.trip,v.arrival_s,v.bus))
        arrivals.sort(key=lambda r:(r[3],r[4]))
        base.observed = dict(base.observed,arrivals=arrivals)
        return base,ArrivalMetricGraph(base,base.observed)

    def test_all_seven_bins_keep_physics_cost_and_native_continuation(self):
        old,new = self.model()
        for action in graph.ACTIONS:
            commands = {(0,0,0):int(action)}
            before,after = old.evaluate(commands),new.evaluate(commands)
            for key in before:
                if key != 'cv2':
                    self.assertEqual(before[key],after[key],(action,key))
            future = {k:times[new.seed_sizes[k]:] for k,times in after['arrivals'].items()}
            expected,_ = recorder_cv(new.snapshot.routes,new.observed['arrivals'],future)
            self.assertAlmostEqual(after['cv2'],expected,places=12)

    def test_current_arrival_seed_is_not_recounted_as_future(self):
        _,model = self.model()
        result = model.evaluate()
        future = {k:times[model.seed_sizes[k]:] for k,times in result['arrivals'].items()}
        actual = model.arrival_prefix.extend(future)
        _,expected_count = recorder_cv(model.snapshot.routes,model.observed['arrivals'],future)
        self.assertEqual(actual.count,expected_count)
        key = False,0
        self.assertEqual(model.seed_sizes[key],3)
        self.assertNotIn(model.snapshot.vehicles[0].arrival_s,future[key])

    def test_committed_holding_and_homogeneous_cost_unchanged(self):
        old,_ = self.model()
        old.snapshot = replace(old.snapshot,vehicles=tuple(replace(v,phase='hold',remaining_s=12.,
            arrival_s=1990.,boarded=0.) if v.bus==2 else v for v in old.snapshot.vehicles))
        old.observed = dict(old.observed,arrivals=old.observed['arrivals']+[(False,0,2,1990.,2)])
        new = ArrivalMetricGraph(old,old.observed)
        for homogeneous in (False,True):
            a,b = old.evaluate(homogeneous=homogeneous),new.evaluate(homogeneous=homogeneous)
            for key in ('nodes','arrivals','returns','next_launch','cost','actions'):
                self.assertEqual(a[key],b[key],key)

    def test_enumeration_dominates_beam_objective_in_same_deterministic_model(self):
        _,model = self.model()
        beam = flow.choose(model)
        current = choose_current(model)
        self.assertLessEqual(current['selected']['cost'],beam['selected']['cost']+1e-9)
        self.assertLessEqual(current['evaluations'],7)
        self.assertTrue(graph.feasible(current['selected'],current['reference']))


class TableModel:
    def __init__(self, legal=(0,5,10,15,20,30,45), native=10):
        self.snapshot = SimpleNamespace(current_bus=0,native_action_s=native,
            vehicles=(SimpleNamespace(bus=0,trip=0,direction=False,index=0,legal=legal),),
            routes={False:(graph.Stop(0),)})
        self.costs = {h:100.+abs(h-15) for h in legal}
        self.changes, self.calls = {},[]

    def evaluate(self, commands=None, homogeneous=False):
        commands = commands or {}
        h = commands.get((0,0,0),self.snapshot.native_action_s)
        self.calls.append((h,homogeneous))
        return dict(dict(cost=self.costs[h],cv2=.1,coverage=True,failure=None,missing_commands=set(),
            returns={0:100.},next_launch={0:200.},delays={0:0.}),**self.changes.get(h,{}))


class EnumerationTests(unittest.TestCase):
    def test_all_legal_bins_once_native_is_reused_and_no_warm_future_is_used(self):
        model = TableModel(legal=(0,10,45))
        result = choose_current(model,True)
        self.assertEqual(sorted(model.calls),[(0,True),(10,True),(45,True)])
        self.assertEqual(result['evaluations'],3)
        self.assertEqual(result['fallback'],result['selected'])

    def test_ties_keep_native_and_strict_improvements_tie_to_shorter_bin(self):
        model = TableModel()
        model.costs = dict.fromkeys(model.costs,100.)
        self.assertEqual(choose_current(model)['action_s'],10)
        model.costs[5] = model.costs[20] = 90.
        self.assertEqual(choose_current(model)['action_s'],5)

    def test_cv_resource_and_missing_command_constraints_are_not_removed(self):
        for change in ({'cv2':.2},{'returns':{0:101.}}, {'next_launch':{0:201.}},
                       {'delays':{0:1.}}, {'missing_commands':{(0,0,1)}},
                       {'failure':'event_limit'},{'coverage':False}):
            model = TableModel(legal=(0,10))
            model.costs[0] = 0.
            model.changes[0] = change
            self.assertEqual(choose_current(model)['action_s'],10,change)

    def test_failed_reference_retains_native_with_one_evaluation(self):
        model = TableModel()
        model.changes[10] = {'coverage':False}
        result = choose_current(model)
        self.assertEqual(result['evaluations'],1)
        self.assertEqual(result['action_s'],10)


if __name__ == '__main__':
    unittest.main()
