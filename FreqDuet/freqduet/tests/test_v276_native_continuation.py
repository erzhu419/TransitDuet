from dataclasses import replace
import unittest

import numpy as np

import freqduet_causal_event_graph as graph
from freqduet_native_continuation import (AVLVehicleSnapshot, Continuation,
    NativeGraph, native_request)
from test_v275_flow_controller import flow_fixture
import freqduet_flow_controller as flow


def packet():
    return dict(bus=0,trip=0,station=0,direction=False,prepare_s=2000.,progress_m=1000.,
        service_s=40.,target_s=360.,predecessor=90,
        departure=dict(trip_id=90,time_s=1700.),avl=(
            AVLVehicleSnapshot(0,False,True,1000.,1000.,0.,10.),
            AVLVehicleSnapshot(2,False,True,500.,1500.,10.,10.)))


def model_fixture():
    snap,visits,events=flow_fixture()
    follower=graph.Vehicle(2,2,False,0,'travel',10.,remaining_fraction=.5)
    snap=replace(snap,vehicles=(*snap.vehicles,follower),
        plans=(*snap.plans,graph.Plan(13,True,snap.now_s+1080)))
    base,_=flow.assemble(snap,visits,events)
    data=dict(links={d:((1000.,10.),)*3 for d in (False,True)},
        fleet={0:dict(progress=1000.,launch=1000.,speed=0.,service=40.,predecessor=90,target=360.),
               1:dict(progress=0.,launch=0.,speed=0.,service=0.,predecessor=None,target=360.),
               2:dict(progress=500.,launch=1500.,speed=10.,service=0.,predecessor=None,target=360.)},
        packets={(0,0,0):packet()},arrivals=[(False,0,90,1600.,2)],
        departures=[(False,0,90,1700.,2)])
    return base,NativeGraph(base,data)


class NativeContinuationTests(unittest.TestCase):
    def test_reuses_native_speed_service_context_and_bin_selector(self):
        action,context,follower=native_request(packet(),tuple(graph.ACTIONS))
        self.assertEqual(action,30)
        self.assertEqual(context.forward_headway_s,300.)
        self.assertEqual(follower.departure_gap_s,540.)
        self.assertEqual(follower.follower_bus_id,2)
        self.assertEqual(follower.source,'same_time_avl_journey_speed_eta')

    def test_missing_or_mismatched_predecessor_does_not_invent_forward_gap(self):
        for fields in (dict(predecessor=None),dict(departure=None),
                       dict(departure=dict(trip_id=99,time_s=1700.))):
            action,context,_=native_request(dict(packet(),**fields),tuple(graph.ACTIONS))
            self.assertFalse(context.evidence_valid)
            self.assertEqual(action,0)

    def test_nearest_spatial_follower_is_not_fastest_eta_or_colocated_bus(self):
        p=packet()
        p['avl']=(*p['avl'],AVLVehicleSnapshot(3,False,True,900.,1999.,100.,10.),
                  AVLVehicleSnapshot(4,False,True,1000.,1000.,1.,10.))
        _,_,follower=native_request(p,tuple(graph.ACTIONS))
        self.assertEqual(follower.follower_bus_id,3)

    def test_forward_fallback_and_legal_tie_use_native_rules(self):
        p=dict(packet(),avl=())
        action,context,_=native_request(p,(0,30,45))
        self.assertFalse(context.follower_evidence_valid)
        self.assertEqual(action,0)  # desired15:0/30 tie chooses0.

    def test_prepare_clock_is_not_replaced_by_later_command_clock(self):
        p=packet()
        p['avl']=(*p['avl'][:1],replace(p['avl'][1],progress_m=750.,launch_time_s=1000.))
        a=native_request(p,tuple(graph.ACTIONS))
        b=native_request(dict(p,prepare_s=2040.),tuple(graph.ACTIONS))
        self.assertNotEqual(a[0],b[0])

    def test_current_rule_mismatch_is_detected_before_forced_native_alias(self):
        _,model=model_fixture()
        model.snapshot=replace(model.snapshot,native_action_s=0)
        with self.assertRaisesRegex(AssertionError,'same-state'):
            model.evaluate()

    def test_fixed_commands_preserve_all_existing_physical_dynamics(self):
        base,model=model_fixture()
        first=base.evaluate()
        commands={key:h for key,(_,h) in first['actions'].items()}
        old,new=base.evaluate(commands),model.evaluate(commands)
        for key in ('arrivals','returns','next_launch','launches','waiting','onboard','cv2','boardings','nodes'):
            self.assertEqual(old[key],new[key],key)

    def test_projected_movement_preserves_tick_and_inline_roster_visibility(self):
        _,model=model_fixture()
        state=Continuation(model)
        vehicles={v.bus:dict(bus=v.bus,trip=v.trip,d=v.direction,i=v.index,phase=v.phase,
            frontier=2050.) for v in model.snapshot.vehicles}
        state.initialize(vehicles)
        early={v.bus_id:v for v in state.avl(2020.,0)}
        late={v.bus_id:v for v in state.avl(2020.)}
        self.assertGreater(late[2].progress_m,early[2].progress_m)
        row=vehicles[0]
        row.update(i=1,phase='travel',frontier=2150.)
        state.event('departure',row,2050.)
        at={v.bus_id:v for v in state.avl(2050.)}
        later={v.bus_id:v for v in state.avl(2052.)}
        self.assertEqual(at[0].progress_m,1000.)
        self.assertGreater(later[0].progress_m,1000.)
        row.update(phase='terminal')
        state.event('return',row,2060.)
        before={v.bus_id:v for v in state.avl(2060.,0)}
        after={v.bus_id:v for v in state.avl(2060.,2)}
        self.assertTrue(before[0].on_route)
        self.assertFalse(after[0].on_route)

    def test_same_budget_and_fallback_constraints_remain_in_force(self):
        _,model=model_fixture()
        result=flow.choose(model)
        self.assertTrue(result['reference']['coverage'])
        self.assertLessEqual(result['evaluations'],128)
        self.assertTrue(graph.feasible(result['fallback'],result['reference']))
        self.assertLessEqual(result['fallback']['cost'],result['reference']['cost'])


if __name__=='__main__':
    unittest.main()
