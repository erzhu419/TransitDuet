import ast
from collections import defaultdict
import copy
import os
from pathlib import Path
import random
from types import MethodType, SimpleNamespace as NS
import unittest

import numpy as np
import torch

from env.bus import Bus
from env.scenario import ScenarioTape
from freqduet_v224_prepare_capture import shadow
from freqduet_v235_passive_cycle_capture import recording_audit as cycle_audit
from freqduet_v241_passive_requests import recording_audit as request_audit
from freqduet_v248_prepare_timing_capture import recording_audit as timing_audit
from freqduet_v255_passive_pulse import recording_audit as pulse_audit
import freqduet_v257_atomic_smoke as atomic
from test_v248_prepare_timing_capture import NativeAudit
from test_v255_passive_pulse import Fixture as PulseFixture


def definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    body = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in body} == set(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return NS(**{name:namespace[name] for name in names})


common = dict(copy=copy,np=np,torch=torch,random=random)
same = definitions(os.environ['FREQDUET_V257_SAME'],['assert_same_state'],dict(common)).assert_same_state
rng = definitions(os.environ['FREQDUET_V257_RNG'],['rng_state'],dict(common)).rng_state
repeated = definitions(os.environ['FREQDUET_V257_REPEATED'],['physical_prefix','fork_future'],
                       dict(common,joint=NS(assert_same_state=same,rng_state=rng)))
event = definitions(os.environ['FREQDUET_V257_EVENT'],['categorical_index','EventUniforms','alignment'],
                    dict(common,defaultdict=defaultdict))
resource = definitions(os.environ['FREQDUET_V257_RESOURCE'],['resource_snapshot'],dict(common,
    require=lambda ok,message: ok or (_ for _ in ()).throw(AssertionError(message)))).resource_snapshot
joint = NS(resource_snapshot=resource,pilot=NS(joint=NS(assert_same_state=same,rng_state=rng)))


class Policy:
    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)
        self.calls, self.last_return, self.weights = 0,None,torch.tensor([3.,7.])

    def sampling_state(self):
        return self.generator.get_state()

    def set_sampling_state(self, state):
        self.generator.set_state(state)


class NativeBridge(NativeAudit):
    """Native callback producer feeding the actual six-observer + V255 chain."""
    def __init__(self, seed, arm):
        super().__init__(seed,arm)
        self.commands,self.policy_inputs,self.arrivals = [],[],[]

    def __call__(self, policy, states, features, theta):
        policy.calls += 1
        torch.rand((),generator=policy.generator)
        native = torch.eye(7)[[self.query_input['base_bin']]]
        policy.last_return = native
        self.policy_inputs.append(states.numpy().tolist())
        return super().__call__(native,self.query_input,features,theta)

    def finish(self, runner, records, holds):
        self.actions = copy.deepcopy(records)
        cycles = self.cycle_capture.data(True)
        total = sum(c['executed_holding_s'] for c in cycles['cycles'])
        self.ledger = dict(physical_vehicles=12,incomplete_trips=len(cycles['terminal']['open_cycles']),
                          executed_holding_s=total,commanded_holding_s=sum(h['action_s'] for h in holds))
        self.contract = dict(fleet=12,bins=list(atomic.ACTIONS),upper='native_B200',lower='bilateral025')


Base = pulse_audit(timing_audit(request_audit(cycle_audit(NativeBridge))),'observer_only')
Audit = atomic.atomic_audit(Base,repeated,event,joint)


class Fixture(PulseFixture):
    def __init__(self, branch='native', reference=None, mask=None, supported=True):
        random.seed(257); np.random.seed(257); torch.manual_seed(257)
        self.uniforms = event.EventUniforms(402112137)
        self.audit = Audit(12037,branch,self.uniforms,reference)
        station = NS(station_id=0,waiting_passengers=[],_pending_passengers=[])
        self.pending = NS(appear_time=7250.,boarding_time=None,arrive_time=None,appear_station=station,
                          destination_station=station,boarded=False,arrived=False)
        station._pending_passengers.append(self.pending)
        recorder = self.audit.recorder_type(); recorder.result = object()
        self.env = NS(time_step=1.,current_time=0.,headway_events=recorder,
            bus_all=[NS(bus_id=b,on_route=False,direction=False) for b in range(12)],
            timetables=[NS(launch_turn=t,direction=False,launched=False,launch_time=s,target_headway=100.)
                        for t,s in ((1,0.),(2,201.),(4,200.))],stations=[station],routes=[NS(speed_limit=7.)],
            _demand_multipliers={6:1.2},_od_multipliers={(1,True):.8},_peak_shift=1.,
            scenario_tape=ScenarioTape(123),_measurement_details=dict(trips_unlaunched=0,trips_incomplete=0))
        self.env.scenario_tape.normal_stream(0,1,'road',1)
        for b in self.env.bus_all:
            self.enrich(b,station)
        self.env.bus_in_terminal = list(self.env.bus_all)
        self.policy,self.upper = Policy(41),Policy(42)
        streams = {name:np.random.RandomState(i) for i,name in enumerate(atomic.STREAMS)}
        self.runner = NS(env=self.env,upper_trainer=NS(policy_net=self.upper),
            _runtime_numpy_streams=lambda:streams,_active_timetable_plans={False:dict(action=np.array([0.]))},
            _pending_lower_action_context={})
        self.audit.attach(self.runner)
        self.rollout_kwargs=self.audit.rollout_kwargs(NS())
        self.physical,self.running,self.return_times,self.draws = {},{},{},[]
        self.ready_directions = dict.fromkeys(range(12),False)
        self.supported,self.mask = supported,list(range(7)) if mask is None else mask
        self.records,self.holds,self.returned_values = [],[],[]
        for _ in range(20):
            self.audit.handoff.mark('tick')
        self.audit.clock.stamp('fresh_upper',0.,trip_id=1,direction=False)
        self.replan(0.,1,False)

    def enrich(self,b,station):
        for k,v in dict(trip_id=-1,state=NS(name='TRAVEL'),current_speed=7.,last_station=station,
            next_station=station,travel_distance=10.,absolute_distance=10.,last_station_dis=10.,
            next_station_dis=90.,holding_time=0.,dwelling_time=0.,episode_hold_vehicle_seconds=100.,
            passengers=[],_stop_start_time=None,_active_holding_load=0,episode_hold_person_seconds=0.,
            applied_actions=[],holding_action_trace_mode='all_decisions').items():
            setattr(b,k,copy.deepcopy(v))
        b._headway_recorder = self.env.headway_events
        b._normalize_action = MethodType(Bus._normalize_action,b)

    def launch(self,bus,trip,direction,now):
        super().launch(bus,trip,direction,now)
        b = self.physical[bus]
        self.enrich(b,self.env.stations[0])
        b.trip_id,b.bus_id,b.direction,b.on_route = trip,bus,direction,True
        self.env.bus_all[bus] = b
        self.env.bus_in_terminal = [v for v in self.env.bus_in_terminal if v.bus_id != bus]
        next(t for t in self.env.timetables if t.launch_turn == trip).launched = True

    def query(self,bus,now,held,masked=False):
        self.env.current_time = now
        self.holding(bus,held)
        a,b = self.audit,self.physical[bus]
        identity = dict(self.running[bus],station_id=b.last_station.station_id)
        prepare = a.handoff.prepares[-1]
        seq = a.clock.stamp('lower_query',now,**identity)['event_seq']
        feasible = [0] if masked else self.mask
        command = shadow.command(NS(**prepare['context']),feasible)
        ready = [dict(bus_id=i,direction=self.ready_directions[i],position_sample_s=now,
            return_observed=i in self.return_times,last_observed_return_s=self.return_times.get(i))
            for i in range(12) if i not in self.running]
        active = [dict(self.running[i],position_sample_s=now,remaining_distance_m=200.+i*100.,
            state='on_route',last_station_id=4,next_station_id=5,progress_m=50.)
            for i in range(12) if i in self.running]
        q = dict(identity,time_s=now,event_seq=seq,snapshot_read_s=now,snapshot_event_seq=seq,
            state_captured_s=prepare['clock']['prepare_s'],prepare_event_seq=prepare['clock']['legacy_prepare_event_seq'],
            context=prepare['context'],ready=ready,active=active,plans=copy.deepcopy(list(a.resource.plans.values())),
            onboard=3,terminal_masked=masked,feasible_bins=list(feasible),base_bin=command['bin'],
            command_s=atomic.ACTIONS[command['bin']],executed_holding_before_s=held)
        a.query_input = q
        self.runner._pending_lower_action_context[bus] = copy.deepcopy(q['context'])
        value = a(self.policy,torch.zeros((1,4)),np.zeros(2),[0.]*4)
        self.returned_values.append(value is self.policy.last_return)
        key,u = self.rollout_kwargs['event_uniforms'].draw(b.trip_id,b.last_station.station_id,b.direction)
        selected = event.categorical_index(value.numpy()[0],u)
        amount = atomic.ACTIONS[selected]
        self.uniforms.records[key] = dict(uniform=u,order=len(self.uniforms.records),time_s=now,action_s=amount)
        self.records.append(dict(time_s=now,trip_id=b.trip_id,station_id=b.last_station.station_id,
            direction=b.direction,action=selected,probabilities=value.numpy()[0].tolist()))
        # The actual frozen bus methods generate the command fields,H counter and departure.
        Bus._start_dwelling(b,np.array([amount],np.float32),current_time=now)
        self.holds.append(dict(time_s=b.last_action_time,bus_id=bus,trip_id=b.trip_id,
            station_id=b.last_action_station_id,action_s=b.last_action_s))
        for tick in range(max(1,int(amount))):
            self.env.current_time = now+tick+1
            Bus._process_dwelling(b,self.env.current_time)
        return amount

    def returned(self,bus,now,held):
        super().returned(bus,now,held)
        b = self.physical[bus]; b.on_route=False; b.direction=self.ready_directions[bus]
        self.env.bus_in_terminal.append(b)

    def complete(self):
        self.launch(0,1,False,0.); self.launch(2,4,False,200.); self.launch(1,2,False,201.)
        self.departure(0,7,7000.,0.)
        self.prepare(2,7,7100.,0.,front=1)
        pre = self.query(2,7100.,0.)
        self.departure(0,8,7190.,0.)
        self.prepare(2,8,7200.,pre,front=1)
        pulse = self.query(2,7200.,pre)
        self.departure(0,9,7250.,0.)
        self.prepare(2,9,7350.+pulse,pre+pulse,front=1)
        later = self.query(2,7350.+pulse,pre+pulse)
        self.returned(0,7999.,0.); self.returned(2,8000.,pre+pulse+later); self.returned(1,8001.,0.)
        self.audit.finish(self.runner,self.records,self.holds)
        return self.audit.branch_data(dict(passenger_person_s=100.+pulse,cv_squared=.2))


class AtomicSmokeTests(unittest.TestCase):
    def setUp(self):
        self.saved = random.getstate(),np.random.get_state(),torch.get_rng_state()
        self.fixtures=[]

    def tearDown(self):
        for f in reversed(self.fixtures):
            f.audit.detach()
        random.setstate(self.saved[0]); np.random.set_state(self.saved[1]); torch.set_rng_state(self.saved[2])

    def make(self,branch='native',reference=None,**kw):
        f=Fixture(branch,reference,**kw); self.fixtures.append(f); return f

    def branches(self,mask=None):
        f=self.make(mask=mask); native=f.complete()
        ref=dict(anchor=native['anchor'],future=native['future'])
        plan=atomic.smoke_plan(native['anchor']['query']['native_bin'],native['anchor']['query']['feasible_bins'])
        return dict(native=native,**{name:self.make(name,ref,mask=mask).complete() for name in plan['physical'][1:]})

    def test_complete_seven_bins_alias_mask_and_actual_execution_gate(self):
        rows=self.branches()
        report=atomic.validate_smoke(rows,same,event.alignment)
        self.assertTrue(report['passed']); self.assertEqual(report['physical_paths'],8)
        self.assertEqual(sorted(b['execution']['anchor_executed_s'] for n,b in rows.items() if n!='forced_native'),list(atomic.ACTIONS))

    def test_masked_and_native_alias_labels_never_start(self):
        rows=self.branches(mask=[0,2,4])
        report=atomic.validate_smoke(rows,same,event.alignment)
        self.assertEqual(report['physical_paths'],4)
        ref={k:rows['native'][k] for k in ('anchor','future')}
        for branch in (*report['plan']['masked'],*report['plan']['aliases']):
            with self.subTest(branch=branch),self.assertRaises(AssertionError):
                self.make(branch,ref,mask=[0,2,4])

    def test_rollout_binding_uses_the_exact_forked_uniform_object(self):
        f=self.make(); f.complete()
        self.assertIs(f.rollout_kwargs['event_uniforms'],f.audit.uniforms)
        self.assertIs(f.rollout_kwargs['probability_rule'],f.audit)
        self.assertIs(f.rollout_kwargs['trace_observer'],f.audit)
        self.assertEqual(f.rollout_kwargs['event_uniforms'].seed,atomic.SMOKE_ROOT+10000000)

    def test_one_original_call_prefix_and_native_object_are_preserved(self):
        for branch in ('native','forced_native'):
            reference=None
            if branch!='native':
                base=self.make().complete(); reference={k:base[k] for k in ('anchor','future')}
            f=self.make(branch,reference); row=f.complete()
            self.assertEqual(f.policy.calls,3); self.assertEqual(f.upper.calls,0)
            self.assertEqual(f.returned_values,[True,True,True])
            self.assertEqual(row['anchor']['query']['query_index'],1)
            self.assertEqual(row['anchor']['query']['clock']['query_s'],7200.)
            self.assertIs(f.env.stations[0]._pending_passengers[0],f.pending)
            self.assertEqual(f.env.routes[0].speed_limit,7.)
            self.assertTrue(torch.equal(f.policy.weights,torch.tensor([3.,7.])))

    def test_pulse_occurs_once_and_later_feedback_is_branch_local(self):
        rows=self.branches()
        self.assertNotEqual(rows['b0']['requests'][-1]['native_bin'],rows['native']['requests'][-1]['native_bin'])
        for b in rows.values():
            self.assertEqual(sum(r['is_anchor'] for r in b['requests']),1)
            self.assertTrue(all(r['selected_bin']==r['native_bin'] for r in b['requests'] if not r['is_anchor']))

    def test_prefix_mismatch_stops_before_future_fork(self):
        row=self.make().complete(); ref={k:row[k] for k in ('anchor','future')}
        for key in ('native_action','lower_state','physical','v248','query'):
            bad=copy.deepcopy(ref)
            if key in ('native_action','lower_state'):
                bad['anchor'][key].reshape(-1)[0]+=1
            elif key=='physical':
                bad['anchor'][key]['route_speed'][0]+=1
            elif key=='v248':
                bad['anchor'][key]['queries']=[]
            else:
                bad['anchor'][key]['clock']['prepare_holding_s']+=1
            f=self.make('forced_native',bad)
            with self.subTest(key=key),self.assertRaises(AssertionError):
                f.complete()
            self.assertIsNone(f.audit.fork_anchor); self.assertEqual(f.env.scenario_tape.seed,123)

    def test_new_future_coupling_preserves_occurrences_and_branch_only_events(self):
        b=self.branches(mask=[0,6]); n=b['native']; c=b['b0']
        self.assertEqual(n['future']['root'],atomic.SMOKE_ROOT)
        self.assertEqual(n['future']['event_counts'],n['anchor']['event_uniforms']['counts'])
        self.assertEqual(n['future']['startup']['scenario_seed'],atomic.SMOKE_ROOT)
        common_uniforms=event.alignment(n['uniforms'],c['uniforms'])
        self.assertEqual(common_uniforms['uniform_mismatches'],0)
        u=event.EventUniforms(atomic.SMOKE_ROOT+10000000)
        v=event.EventUniforms(atomic.SMOKE_ROOT+10000000)
        a,ua=u.draw(1,3,False); u.draw(99,1,True); z,uz=u.draw(2,4,False)
        zz,vz=v.draw(2,4,False); aa,va=v.draw(1,3,False)
        self.assertEqual((a,ua,z,uz),(aa,va,zz,vz))

    def test_missing_anchor_stops_without_later_replacement(self):
        f=self.make(supported=False)
        with self.assertRaisesRegex(AssertionError,'anchor absent'):
            f.complete()
        self.assertEqual(f.env.scenario_tape.seed,123)

    def test_sampler_command_counter_and_missing_departure_are_checked(self):
        b=self.branches(mask=[0,6]); ref=b['native']
        for field in ('sample','issue','H','missing','later_override'):
            x=copy.deepcopy(ref)
            if field=='sample': x['actions'][1]['action']=0
            elif field=='issue': x['holds'][1]['action_s']=0.
            elif field=='H':
                q=x['requests'][1]
                for d in x['payloads']['v248']['departures']:
                    if d['identity']['trip_id']==q['identity']['trip_id'] and d['identity']['station_id']==q['identity']['station_id']:
                        d['completed']['timing']['realized_holding']+=1.
            elif field=='missing': x['payloads']['v248']['departures']=[]
            else: x['requests'][-1]['selected_bin']=0
            bad=copy.deepcopy(b); bad['native']=x
            with self.subTest(field=field),self.assertRaises(AssertionError):
                atomic.validate_smoke(bad,same,event.alignment)

    def test_repeated_query_cannot_claim_later_holding_as_pulse(self):
        row=self.make().complete()
        requests=copy.deepcopy(row['requests']); i=1
        departures=copy.deepcopy(row['payloads']['v248']['departures'])
        order=requests[i]['clock']['query_order']
        for d in departures:
            if d['departure_order']>order: d['departure_order']+=2
        for q in requests[i+1:]: q['clock']['query_order']+=2
        later=copy.deepcopy(requests[i]); later['is_anchor']=False
        later['clock']['query_order']+=1; later['clock']['query_s']+=1
        requests.insert(i+1,later)
        actions=copy.deepcopy(row['actions']); action=copy.deepcopy(actions[i]); action['time_s']+=1; actions.insert(i+1,action)
        holds=copy.deepcopy(row['holds']); hold=copy.deepcopy(holds[i]); hold['time_s']+=1; holds.insert(i+1,hold)
        report=atomic.execution_chain(requests,actions,holds,departures)
        self.assertFalse(report['complete']); self.assertEqual(report['anchor_executed_s'],0.)

    def test_new_future_alias_not_old_future_and_every_physical_label_required(self):
        rows=self.branches(mask=[0,6]); native=rows['native']
        self.assertNotEqual(native['anchor']['scenario']['seed'],native['future']['root'])
        for field in ('missing_arm','extra_alias','future','uniform','native_endpoint','fleet','clearing'):
            bad=copy.deepcopy(rows)
            if field=='missing_arm': bad.pop('b0')
            elif field=='extra_alias': bad['b6']=bad['native']
            elif field=='future': bad['b0']['future']['root']+=1
            elif field=='uniform': next(iter(bad['b0']['uniforms'].values()))['uniform']+=.01
            elif field=='native_endpoint': bad['forced_native']['results']['cv_squared']+=1.
            elif field=='fleet': bad['b0']['ledger']['physical_vehicles']=13
            else: bad['b0']['completion']['trips_incomplete']=1
            with self.subTest(field=field),self.assertRaises(AssertionError):
                atomic.validate_smoke(bad,same,event.alignment)


if __name__=='__main__':
    unittest.main()
