import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from freqduet_apc_flow_evidence import APCArrival, completed_flow
from freqduet_causal_event_graph import Visit
from freqduet_native_event_graph import NativeNoop
from freqduet_v274_passive_flow import FlowRecorder, boundary, compare_legacy, validate_flow
from run_freqduet_protocol_v6_checkpoint_trajectory import assert_same_state as same
from run_freqduet_v274_passive_flow import make_audit, passive, require_release, SLOT
from seal_freqduet_v274_flow_equivalence import protocol, write
from test_v271_native_event_graph import fake_env


def event(i, boarded, time=None, completed=None):
    t = float(i*60 if time is None else time)
    return APCArrival(i,i,i,False,1,t,t,t+1 if completed is None else completed,
                      0.,0.,float(boarded),50.)


def fixture():
    rows = [event(0,10),event(1,50),event(2,40)]
    rows.append(APCArrival(3,2,2,False,2,180.,180.,180.,40.,40.,0.,50.))
    q = dict(time_s=130.,bus=2,trip=2,station=1,direction=False,arrival_s=120.,
             boarded=40,load=40,action_s=5)
    recorder = FlowRecorder()
    prefix = NS(flow_events=rows[:3])
    recorder.capture(prefix,q)
    prefix.flow_events = rows
    flow = recorder.bundle(prefix,{False:[NS(station=2)]})
    trace = dict(arrivals=[dict(trip_id=e.trip,station_id=e.station,direction=e.direction,time_s=e.arrival_s) for e in rows],
        departures=[dict(trip_id=e.trip,station_id=e.station,direction=e.direction,time_s=e.completed_s) for e in rows],
        dispatches=[dict(trip=i,bus=i,direction=False) for i in range(3)],queries=[q],uniforms={},
        holds=[dict(time_s=130.,bus_id=2,trip_id=2,station_id=1,action_s=5)])
    graph = dict(queries=[dict(time_s=130.,native_action_s=5,shadow_action_s=45,build_s=.1,total_s=.2)],
        completed_visits=[Visit(e.direction,e.station,e.arrival_s,e.completed_s,e.available_s,
            60.,e.incoming,e.boarded,e.alighted,60.,e.capacity) for e in rows[1:3]])
    return flow,trace,graph


class PassiveFlowTests(unittest.TestCase):
    def test_first_and_capacity_censored_visits_remain_in_complete_roster(self):
        flow,trace,graph = fixture()
        result = validate_flow(flow,trace,graph,same)
        self.assertEqual((result['arrivals'],result['queries'],result['completed_visits']),(4,1,2))
        self.assertEqual(result['cases']['first_arrivals'],2)
        self.assertEqual(result['cases']['full_visits'],1)
        self.assertEqual(result['case_status']['zero_seat_visits'],'not_exercised')
        rows = [APCArrival(**r) for r in flow['events']]
        self.assertEqual(completed_flow(rows[:3],130.)['exact_blocks'][0]['arrivals'],90.)

    def test_capture_does_not_mutate_prefix_or_old_records(self):
        rows = [event(0,10),event(1,50)]
        prefix = NS(flow_events=rows,visits=['unchanged'])
        old = copy.deepcopy(prefix)
        recorder = FlowRecorder()
        recorder.capture(prefix,dict(time_s=100.,bus=1,trip=1,station=1,direction=False))
        self.assertEqual(prefix,old)

    def test_same_clock_later_arrival_is_not_retroactively_visible(self):
        prefix = NS(flow_events=[event(0,10,time=0)])
        recorder = FlowRecorder()
        q = dict(time_s=60.,bus=0,trip=0,station=1,direction=False)
        recorder.capture(prefix,q)
        prefix.flow_events.append(event(1,0,time=60))
        self.assertEqual(recorder.queries[0]['arrival_seen_count'],1)
        self.assertEqual(recorder.queries[0],boundary(prefix.flow_events[:1],q,0))

    def test_same_clock_order_and_zero_duration_are_not_imputed(self):
        rows = [event(0,50,time=0),event(1,40,time=0),event(2,0,time=0)]
        evidence = completed_flow(rows,2.)
        self.assertEqual([r['first_order'] for r in evidence['intervals']],[0,1])
        self.assertTrue(all(r['lower_arrivals'] == r['upper_arrivals'] == 0 for r in evidence['intervals']))
        self.assertIsNone(evidence['exact_blocks'][0]['interval_average_rate'])

    def test_completed_and_available_clocks_are_both_strict(self):
        q = dict(time_s=100.,bus=1,trip=1,station=1,direction=False)
        for row in (replace(event(1,20),completed_s=100.),replace(event(1,20),available_s=100.)):
            captured = boundary([event(0,10),row],q,0)
            self.assertEqual(captured['usable_strict_count'],1)
            self.assertEqual(captured['max_completed_s'],1.)

    def test_later_completion_does_not_rewrite_earlier_support(self):
        rows = [event(0,10),replace(event(1,50),completed_s=None),event(2,40)]
        q = dict(time_s=130.,bus=2,trip=2,station=1,direction=False)
        captured = boundary(rows,q,0)
        rows[1] = replace(rows[1],completed_s=140.)
        self.assertEqual(captured,boundary(rows,q,0))
        self.assertEqual(completed_flow(rows,130.)['exact_blocks'],[])

    def test_missing_first_arrival_is_rejected(self):
        flow,trace,graph = fixture()
        flow['events'].pop(0)
        with self.assertRaises(AssertionError):
            validate_flow(flow,trace,graph,same)

    def test_duplicate_arrival_is_rejected(self):
        flow,trace,graph = fixture()
        flow['events'].append(dict(flow['events'][0]))
        with self.assertRaises(AssertionError):
            validate_flow(flow,trace,graph,same)

    def test_wrong_vehicle_or_query_APC_is_rejected(self):
        for change in ('vehicle','load'):
            flow,trace,graph = fixture()
            if change == 'vehicle':
                flow['events'][0]['bus'] = 9
            else:
                trace['queries'][0]['load'] = 39
            with self.assertRaises(AssertionError):
                validate_flow(flow,trace,graph,same)

    def test_missing_visible_event_or_forged_support_is_rejected(self):
        for field,value in (('arrival_seen_count',2),('usable_strict_count',0),('max_completed_s',130.)):
            flow,trace,graph = fixture()
            flow['queries'][0][field] = value
            with self.assertRaises(AssertionError):
                validate_flow(flow,trace,graph,same)

    def test_only_graph_wall_clock_fields_may_change(self):
        _,trace,graph = fixture()
        result = {k:{} for k in ('results','contract','coverage','execution')}
        changed = copy.deepcopy(graph)
        changed['queries'][0].update(build_s=7.,total_s=8.)
        compare_legacy(trace,result,changed,trace,result,graph,same)
        changed['queries'][0]['time_s'] += 1
        with self.assertRaises(AssertionError):
            compare_legacy(trace,result,changed,trace,result,graph,same)

    def test_native_action_timing_and_endpoint_changes_fail_legacy_gate(self):
        _,trace,graph = fixture()
        result = {k:{} for k in ('results','contract','coverage','execution')}
        for target,field in (('holds','action_s'),('arrivals','time_s')):
            changed = copy.deepcopy(trace)
            changed[target][0][field] += 1
            with self.assertRaises(AssertionError):
                compare_legacy(changed,result,graph,trace,result,graph,same)
        changed = copy.deepcopy(result)
        changed['results']['journey_min'] = 1.
        with self.assertRaises(AssertionError):
            compare_legacy(trace,changed,graph,trace,result,graph,same)

    def test_callback_returns_same_tensor_even_when_shadow_disagrees(self):
        law = torch.tensor([[0.,1.,0.,0.,0.,0.,0.]])
        class Base:
            def __init__(self,*args): pass
            def __call__(self,*args): return law
        fixed = NS(BenchmarkAudit=Base,pilot=NS(step=NS(ACTIONS=np.array([0,5,10,15,20,30,45]))))
        credit = NS(stream=NS(pending_commands=[]))
        audit = make_audit(fixed,credit)
        bus = NS(bus_id=0,trip_id=0,last_station=NS(station_id=1),direction=False,
                 state=NS(name='WAITING_ACTION'),passengers=[None]*10,last_board_count=10,
                 last_board_time=0.,episode_hold_vehicle_seconds=0.)
        calls = []
        audit.env = NS(current_time=10.)
        audit.runner = NS(_pilot_agent=0,_bus_for_agent=lambda _:bus)
        audit.bridge = NS(prefix=NS(flow_events=[replace(event(0,10),completed_s=None)]),
                          observe=lambda *args:calls.append(args) or dict(shadow_action_s=45))
        policy = NS(feasible_action_mask=lambda _:torch.ones((1,7),dtype=torch.bool))
        self.assertIs(audit(policy,None,None,None),law)
        self.assertEqual(calls[0][2],5)
        self.assertEqual(audit.native_queries[0]['action_s'],5)
        self.assertEqual(len(audit.flow.queries),1)
        self.assertEqual(credit.stream.pending_commands,[])

    def test_native_terminal_hook_deduplicates_and_restores(self):
        class Recorder:
            def record_departure(self,*args,**kwargs): return 'departure'
        class Bus:
            def exchange_passengers(self,*args):
                self.passengers,self.last_board_count = [],0
                return 'arrival'
            def drive(self,current_time,*args):
                self.on_route,self.back_to_terminal_time = False,current_time
                return 'return'
        env = fake_env()
        env.headway_events = Recorder()
        bus = Bus()
        bus.__dict__.update(bus_id=0,trip_id=5,next_station=NS(station_id=3),direction=True,
            passengers=[None]*10,capacity=50.,on_route=True,effective_station=[NS(station_id=3)])
        bridge = NativeNoop(env,{5:dict(time_s=0.,bus_id=0)},NS())
        originals = Bus.exchange_passengers,Bus.drive,Recorder.record_departure
        bridge.attach(Bus)
        try:
            self.assertEqual(bus.exchange_passengers(100.,False),'arrival')
            self.assertEqual(bus.drive(100.),'return')
            self.assertEqual(env.headway_events.record_departure(3,True,100.,5),'departure')
            self.assertEqual(len(bridge.prefix.flow_events),1)
            self.assertEqual(bridge.prefix.flow_events[0].completed_s,100.)
            self.assertEqual(bridge.prefix.first_visits,1)
        finally:
            bridge.detach()
        self.assertEqual((Bus.exchange_passengers,Bus.drive,Recorder.record_departure),originals)

    def test_missing_release_stops_before_runtime_import_or_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'protocol.json',protocol('d43947ed64'))
            with self.assertRaises(FileNotFoundError):
                passive(NS(out_root=root,source='entry',results_root=root.parent))
            self.assertFalse((root/'started').exists())
            self.assertNotIn('run_freqduet_v267_bilateral_factorial',sys.modules)

    def test_one_release_cannot_create_two_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sealed = protocol('d43947ed64')
            write(root/'protocol.json',sealed)
            write(root/'budget'/f'{SLOT}.json',dict(source='entry',slot=SLOT,
                  context=sealed['context'],automatic_retry=False))
            write(root/'entry_checks/status.json',dict(source='entry',status='synthetic_passive_entry_passed'))
            require_release(root,'entry')
            with self.assertRaises(FileExistsError):
                require_release(root,'entry')
            self.assertEqual(len(list((root/'started').glob('*.json'))),1)


if __name__ == '__main__':
    unittest.main()
