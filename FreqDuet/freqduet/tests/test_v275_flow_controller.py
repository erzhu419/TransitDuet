from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import freqduet_causal_event_graph as old
import freqduet_flow_controller as ctrl
from freqduet_apc_flow_evidence import APCArrival
from test_v270_causal_event_graph import fixture


def flow_fixture():
    snap, visits = fixture()
    arrivals = []
    for d, stops in snap.routes.items():
        for stop in stops:
            for t,b in zip((200.,500.,800.,1100.,1400.),(0.,10.,0.,50.,40.)):
                arrivals.append(APCArrival(-1,0,0,d,stop.station,t,t,t+1,0.,0.,b,50.))
    arrivals.sort(key=lambda r:r.arrival_s)
    return snap, visits, [replace(e,order=i) for i,e in enumerate(arrivals)]


class FlowControllerTests(unittest.TestCase):
    def test_closed_blocks_recover_full_middle_visit_without_backlog_double_count(self):
        snap, visits, events = flow_fixture()
        model, info = ctrl.assemble(snap,visits,events)
        self.assertAlmostEqual(model.forecasts[False,0]['rate'],100/1200)
        self.assertEqual(model.forecasts[False,0]['historical_arrivals'],100.)
        self.assertFalse(info['future_rate_upper_bound_identified'])
        self.assertLess(info['max_completed_s'],snap.now_s)

    def test_capacity_censored_legacy_demand_can_be_supported_by_closed_blocks(self):
        snap, visits, events = flow_fixture()
        full = [replace(v,incoming_load=0.,alighted=0.,boarded=50.) for v in visits]
        with self.assertRaises(old.Unsupported):
            old.assemble(snap,full)
        model,_ = ctrl.assemble(snap,full,events)
        self.assertGreater(model.forecasts[False,0]['rate'],0)

    def test_repeat_queries_do_not_duplicate_historical_support(self):
        snap, visits, events = flow_fixture()
        a,_ = ctrl.assemble(snap,visits,events)
        b,_ = ctrl.assemble(snap,visits,events)
        self.assertEqual(a.forecasts,b.forecasts)
        self.assertEqual(a.forecasts[False,0]['demand_blocks'],3)

    def test_unfinished_visit_breaks_block_and_both_clocks_are_strict(self):
        snap, visits, events = flow_fixture()
        for kwargs in (dict(completed_s=None),dict(completed_s=snap.now_s),dict(available_s=snap.now_s)):
            bad=[replace(e,**kwargs) if e.arrival_s==800 and e.key==(False,0) else e for e in events]
            with self.assertRaises(old.Unsupported):
                ctrl.assemble(snap,visits,bad)

    def test_late_apc_does_not_change_query_forecast(self):
        snap, visits, events = flow_fixture()
        first,_=ctrl.assemble(snap,visits,events)
        extra=APCArrival(len(events),0,0,False,0,1900.,2100.,2101.,0.,0.,49.,50.)
        second,_=ctrl.assemble(snap,visits,[*events,extra])
        self.assertEqual(first.forecasts,second.forecasts)
        self.assertEqual(first.initial_queues,second.initial_queues)

    def test_open_full_interval_is_not_imputed_as_exact_zero_demand(self):
        snap, visits, events = flow_fixture()
        extra=APCArrival(len(events),0,0,False,0,1900.,1900.,1901.,50.,0.,0.,50.)
        model,info=ctrl.assemble(snap,visits,[*events,extra])
        self.assertGreater(info['open_censored_blocks'],0)
        self.assertGreater(model.initial_queues[False,0],0)
        self.assertAlmostEqual(model.initial_queues[False,0],100/1200*600)

    def test_homogeneous_cost_changes_no_demand_or_vehicle_dynamics(self):
        snap, visits, events = flow_fixture()
        model,_=ctrl.assemble(snap,visits,events)
        a,b=model.evaluate(),model.evaluate(homogeneous=True)
        for field in ('arrivals','actions','launches','returns','waiting','boardings'):
            self.assertEqual(a[field],b[field])

    def test_old_assembler_keeps_existing_queue_behavior(self):
        snap,visits=fixture()
        model=old.assemble(snap,visits)
        self.assertIsNone(model.initial_queues)
        self.assertTrue(model.evaluate()['coverage'])

    def test_solver_counts_fallback_simulations_inside_original_budget(self):
        snap,visits,events=flow_fixture()
        model,_=ctrl.assemble(snap,visits,events)
        with patch.object(old,'MAX_EVALUATIONS',12):
            result=ctrl.choose(model)
        self.assertLessEqual(result['evaluations'],12)
        self.assertIn(result['action_s'],snap.vehicles[0].legal)
        self.assertTrue(old.feasible(result['fallback'],result['reference']))
        self.assertLessEqual(result['fallback']['cost'],result['reference']['cost'])


class ContinuationTests(unittest.TestCase):
    def wrapped(self,damage):
        key=(0,0,0)
        class Model:
            snapshot=SimpleNamespace(current_bus=0,native_action_s=30,
                vehicles=(old.Vehicle(0,0,False,0,'prepare',40.),),routes={False:(old.Stop(0),)})
            def evaluate(self,commands=None,homogeneous=False):
                commands=commands or {}
                changed=commands.get(key,30)!=30
                full=changed and (0,0,1) in commands
                return dict(cost=90. if full else 100.+(damage if changed else 0.),cv2=.1,
                    returns={0:1000.},next_launch={0:1001.},delays={1:0.},
                    missing_commands=set(),coverage=True,failure=None)
        return ctrl.FallbackGraph(Model()),key

    def test_harmful_current_action_is_not_accepted_for_promised_future_gain(self):
        wrapped,key=self.wrapped(20.)
        ref=wrapped.evaluate()
        result=wrapped.evaluate({key:20,(0,0,1):10})
        self.assertEqual(result['full_cost'],90.)
        self.assertEqual(result['fallback']['cost'],120.)
        self.assertFalse(old.feasible(result,ref))
        self.assertEqual(result['failure'],'baseline_continuation_harm')

    def test_both_complete_and_fallback_costs_enter_objective(self):
        wrapped,key=self.wrapped(-5.)
        ref=wrapped.evaluate()
        result=wrapped.evaluate({key:20,(0,0,1):10})
        self.assertTrue(old.feasible(result,ref))
        self.assertEqual(result['cost'],95.)
        self.assertEqual(wrapped.calls,3)
        wrapped.evaluate({key:20,(0,0,1):15})
        self.assertEqual(wrapped.calls,4)

    def test_no_extra_simulation_after_budget_exhaustion(self):
        wrapped,key=self.wrapped(-5.)
        wrapped.evaluate()
        with patch.object(old,'MAX_EVALUATIONS',2):
            result=wrapped.evaluate({key:20})
            self.assertEqual(result['failure'],'query_evaluation_budget')
            wrapped.evaluate({key:15})
        self.assertEqual(wrapped.calls,2)


class DevelopmentGridTests(unittest.TestCase):
    def test_twelve_unique_paths_and_eight_nonoverlapping_units(self):
        import run_freqduet_v275_flow_controller as pilot
        indices=[i for shard in pilot.SHARDS for i in shard]
        self.assertEqual(sorted(indices),list(range(12)))
        self.assertEqual(len(set(pilot.ITEMS)),12)
        self.assertEqual(len(pilot.DAYS),4)
        self.assertEqual(len(pilot.SHARDS),8)

    def test_frozen_joint_gate_rejects_passenger_gain_with_cv_harm(self):
        import run_freqduet_v275_flow_controller as pilot
        cube={(d,a):{m:10. for m in pilot.METRICS} for d,a in pilot.ITEMS}
        for d in pilot.DAYS:
            cube[d,pilot.ARMS[1]]['passenger_person_s']=9.
            cube[d,pilot.ARMS[1]]['cv_squared']=11.
        self.assertTrue(all(not x['joint_development_gate'] for x in pilot.summarize(cube).values()))


if __name__=='__main__':
    unittest.main()
