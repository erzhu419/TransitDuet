"""Synthetic contracts only: no native runner, rollout or optimizer update."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

import basis_integration as legacy
import basis_law as legacy_law
import conditioned_integration as new
import conditioned_law as law
from recipe import divisors, make_trainer, training_cells


def fleet_fixture(rho, observation):
    fleet = [dict(bus_id=i, on_route=False, direction=bool(i % 2), progress_m=0.,
                  load=i, capacity=50, active_trip_id=None, remaining_service_s=0.,
                  remaining_command_hold_s=0.) for i in range(12)]
    captured = dict(fleet=fleet, horizon_s=61380., route_lengths_m=[10500., 10500.], plans=[])
    env = NS(bus_all=[NS(bus_id=i) for i in range(12)])
    manager = NS(env=env, upper_summary=dict(freq_updates=120, freq_low_demand=1.,
        freq_low_slope=.1, freq_low_forecast=1.2, freq_high_energy=.2),
        moments=NS(count=12, total=4320., square_total=1555200.),
        last_actual={d:dict(actual_s=7000., release_s=7000.) for d in (False, True)},
        completed={False:[], True:[]}, _history=lambda *args:None,
        credit=NS(stream=NS(pending_commands=[])), remaining={}, public={},
        retain_fraction=rho, budget_observation=observation)
    return manager, captured


def terminal_fixture(cls, dimension, config):
    manager = cls.__new__(cls)
    start, end = law.START_S, law.START_S+law.PERIOD_S
    state = np.zeros(dimension, dtype=np.float32)
    manager.env = NS(protocol=NS(evaluation_end_time_s=end))
    manager.credit = NS(stream=NS(closed=True, clock_s=end))
    manager.audit = NS(delays={'public_launch_delay_total_s':0.})
    manager.units = new.RewardUnits(**config['reward_units'])
    manager.interval = start, state, 2, np.zeros(4)
    manager.prefix, manager.final_parts = np.zeros(4), None
    manager.parts = lambda clock:np.asarray([360., 180., 0., .01])
    manager.transitions, manager.choices = [], [dict(action=2, mode='budget', planned_changes=0)]
    manager.behavior_epsilon, manager.deterministic = 0., True
    manager.retain_fraction, manager.budget_observation = .25, 'explicit'
    manager.replay_appended = False
    return manager


class Contracts(unittest.TestCase):
    def test_matched_budget_day_update_budget(self):
        left, right = [training_cells(PROTOCOL, a) for a in ('explicit', 'implicit')]
        self.assertEqual([{k:v for k,v in c.items() if k!='arm'} for c in left],
                         [{k:v for k,v in c.items() if k!='arm'} for c in right])
        self.assertEqual(len({c['day'] for c in left}), 32)
        for batch in range(8):
            self.assertEqual(sorted(c['rho'] for c in left if c['batch']==batch), [.25,.25,.5,.5])
        self.assertFalse({c['day'] for c in left} & set(PROTOCOL['development_days_proposed']))
        self.assertFalse(set(range(2400001, 2400017)) & set(PROTOCOL['development_days_proposed']))
        self.assertEqual(2*32, PROTOCOL['proposed_training_paths'])
        self.assertEqual(2*8*192, PROTOCOL['proposed_total_candidate_updates'])
        self.assertEqual(4*2*3, PROTOCOL['proposed_development_paths'])
        self.assertEqual(2+64+24, PROTOCOL['proposed_physical_start_cap'])
        self.assertEqual(PROTOCOL['authorized_physical_starts'], 0)

    def test_exact_legacy_prefix_and_budget_visibility(self):
        encoded = {}
        for mode in ('explicit', 'implicit'):
            for rho in (.25,.5):
                manager, captured = fleet_fixture(rho,mode)
                with patch.object(legacy,'capture_fleet',return_value=captured), \
                     patch.object(new,'capture_fleet',return_value=captured):
                    old = legacy.encode_state(manager,7200.,0.,2,
                        np.asarray(BASE['state_divisors'],dtype=np.float32))
                    state = new.encode_state(manager,7200.,0.,2,divisors(BASE))
                np.testing.assert_array_equal(state[:180],old)
                self.assertEqual(state.shape,(181,))
                self.assertEqual(state[-1],rho if mode=='explicit' else 0.)
                encoded[mode,rho] = state
        np.testing.assert_array_equal(encoded['implicit',.25],encoded['implicit',.5])
        np.testing.assert_array_equal(encoded['explicit',.25][:-1],encoded['explicit',.5][:-1])

    def test_quota_matches_frozen_formula(self):
        saved = legacy_law.RETAIN_FRACTION
        try:
            for rho in (.25,.5):
                legacy_law.RETAIN_FRACTION = rho
                for normal in (0.,15.,83.5,200.):
                    for pressure in (0.,1.,40.,300.):
                        self.assertEqual(law.quota(normal,pressure,rho),legacy_law.quota(normal,pressure))
        finally:
            legacy_law.RETAIN_FRACTION = saved

    def test_instance_budget_and_no_republication_refill(self):
        managers = []
        for rho in (.25,.5):
            env = NS(current_time=7200.,timetables=[])
            manager = law.SlowManager('budget',env,None,rho)
            manager.upper_summary = dict(freq_updates=0,freq_low_demand=0.,freq_low_forecast=0.)
            manager._estimates = lambda env,captured,now:({(0,10):dict(bus=0,trip=10,
                normal_command_s=100.,ordinary_return_s=now+149.,remaining_controls=10)}, {})
            manager._match = lambda captured,estimates,now,use_budget:{20:dict(bus=0,
                active_trip=10,public_s=now+50.)}
            manager.active[0,10] = dict(command_s=0.,residence_s=0.,commands=0)
            managers.append(manager)
        with patch.object(law,'capture_fleet',return_value={}):
            for manager in managers:
                manager.publish(manager.env,None)
                self.assertEqual(manager.remaining[0,10],100*manager.retain_fraction)
                manager.commit(0,10,1,7200.,0)
                self.assertEqual(manager.active[0,10]['residence_s'],1.)
                manager.commit(0,10,2,7201.,5)
                manager.env.current_time += 900.
                manager.publish(manager.env,None)
                self.assertEqual(manager.remaining[0,10],100*manager.retain_fraction-5)
        self.assertEqual([m.remaining[0,10] for m in managers],[20.,45.])

    def test_terminal_credit_and_replay_shape(self):
        old = terminal_fixture(legacy.LearnedManager,180,BASE)
        candidate = terminal_fixture(new.LearnedManager,181,BASE)
        old.finish_upper()
        candidate.finish_upper()
        for key,value in old.upper_receipt.items():
            self.assertEqual(value,candidate.upper_receipt[key])
        self.assertEqual(candidate.transitions[-1]['next_state'].shape,(181,))
        np.testing.assert_array_equal(candidate.transitions[-1]['next_state'],np.zeros(181))
        self.assertEqual(candidate.transitions[-1]['budget_retention'],.25)
        rows=[]
        candidate.trainer=NS(replay_buffer=NS(push=lambda *row:rows.append(row)))
        self.assertEqual(candidate.append_completed_day_to_replay(),1)
        self.assertEqual(rows[0][0].shape,(181,))
        self.assertTrue(rows[0][4])

    def test_real_actor_critic_and_target_without_updates(self):
        trainers=[make_trainer(BASE,PROTOCOL['learner_seed']) for _ in range(2)]
        for name in ('policy_net','q_net','target_q_net'):
            left,right=[getattr(t,name).state_dict() for t in trainers]
            self.assertEqual(left.keys(),right.keys())
            for key in left:
                self.assertTrue(torch.equal(left[key],right[key]))
        trainer=trainers[0]
        state=np.zeros(181,dtype=np.float32)
        state[-1]=.25
        action,pi,mu=new.choose_upper_action(trainer,state,True,0.)
        self.assertIn(action,(2,3))
        self.assertEqual(len(pi),2)
        self.assertEqual(sum(mu),1.)
        for action,done in ((2,False),(3,True)):
            following=np.zeros(181,dtype=np.float32) if done else state.copy()
            trainer.replay_buffer.push(state,np.asarray([action],dtype=np.float32),
                                      -.01,following,done,1.)
        trainer.replay_buffer.sample(2)
        self.assertTrue(trainer.target_receipt['finite'])
        self.assertEqual(trainer.target_receipt['outside_clip'],0)
        self.assertEqual(trainer.successful_updates,0)

    def test_audit_passes_budget_to_installed_manager(self):
        audit=new.LearnedAudit(None,None,new.RewardUnits(**BASE['reward_units']),
            divisors(BASE),True,0.,retain_fraction=.25,budget_observation='implicit')
        env,bridge=NS(),NS(manager=None)
        def attached(obj,runner):
            obj.env,obj.bridge=env,bridge
        with patch.object(new.V314Audit,'attach',attached), \
             patch.object(new,'LearnedManager') as manager:
            audit.attach(NS())
            self.assertEqual(manager.call_args.kwargs['retain_fraction'],.25)
            self.assertEqual(manager.call_args.kwargs['budget_observation'],'implicit')
            self.assertIs(audit.manager,bridge.manager)

    def test_publish_uses_181_state_and_deduplicates(self):
        manager=terminal_fixture(new.LearnedManager,181,BASE)
        manager.env.current_time=7200.
        manager.env.frequency_tracker=NS(total_updates=120)
        manager.credit.stream.clock_s=7200.
        manager.audit.public_clocks={}
        manager.upper_provider=NS(refresh=lambda *args:dict(freq_updates=120))
        manager.last_cycle=-1
        manager.public=None
        manager.prefix,manager.interval=None,None
        manager.choices,manager.events=[],[]
        manager.previous_action=None
        manager.trainer=None
        manager.divisors=divisors(BASE)
        bridge=NS(prefix=NS(flow_events=[]))
        state=np.zeros(181,dtype=np.float32);state[-1]=.25
        def published(obj,env,bridge):
            obj.last_cycle=int((env.current_time-law.START_S)//law.PERIOD_S)
            obj.events.append(dict(kind='publish',time_s=env.current_time,plans=[],budgets=[]))
        with patch.object(new,'encode_state',return_value=state), \
             patch.object(new,'choose_upper_action',return_value=(3,[.2,.8],[0.,1.])) as choose, \
             patch.object(law.SlowManager,'publish',published):
            manager.publish(manager.env,bridge)
            manager.publish(manager.env,bridge)
            self.assertEqual(choose.call_count,1)
            self.assertIs(choose.call_args.args[1],state)
            self.assertEqual(manager.previous_action,3)
            self.assertEqual(manager.arm,'joint')

    def test_terminal_time_rule_is_unchanged(self):
        for low in (0.,1.,5.):
            for forecast in (0.,1.,20.):
                for supported in (False,True):
                    args=(7200.,7000.,7380.,7740.,low,forecast,supported)
                    self.assertEqual(law.terminal_time(*args),legacy_law.terminal_time(*args))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    PROTOCOL=json.loads((Path(__file__).parent/'protocol.json').read_text())
    BASE=json.loads(args.base_config.read_text())
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
        failures=[dict(test=str(t),error=e) for t,e in result.failures+result.errors],
        physical_starts=0,native_runner_constructions=0,optimizer_updates=0,
        candidate_training_updates=0,main_policy_changed=False,
        development_day_freshness_verified=False)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'results.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    raise SystemExit(0 if result.wasSuccessful() else 1)
