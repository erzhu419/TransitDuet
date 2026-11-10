#!/usr/bin/env python3
"""Matched fixed/B200 upper x unchanged V107 bilateral/zero holding pilot."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

import run_freqduet_v107_rule_confirmation as frozen


prior, pilot = frozen.prior, frozen.pilot
SEEDS, DAYS = tuple(frozen.SEEDS[:4]), tuple(frozen.DAYS[:4])
ARMS = ('learned_bilateral025', 'learned_zero', 'fixed_bilateral025', 'fixed_zero')
METRICS = (*frozen.METRICS, 'public_launch_delay_total_s', 'readiness_delay_total_s')
KEYS = ('passenger_person_s','journey_min','cv_squared','readiness_delay_s','public_launch_delay_total_s')
ITEMS = tuple((s,d,m) for s,d,m in itertools.product(SEEDS,DAYS,ARMS[:2])) + tuple(
    (SEEDS[0],d,m) for d,m in itertools.product(DAYS,ARMS[2:]))
SMOKE_MAIN = tuple((SEEDS[0],DAYS[0],m) for m in ARMS)
REMAINING = tuple(x for x in ITEMS if x not in SMOKE_MAIN)
SHARDS = tuple(REMAINING[i::8] for i in range(8))
CONTRASTS = dict(holding_at_learned=(ARMS[0],ARMS[1]), holding_at_fixed=(ARMS[2],ARMS[3]),
    upper_with_bilateral=(ARMS[0],ARMS[2]), upper_without_holding=(ARMS[1],ARMS[3]),
    bilateral_B200_vs_strong_fixed=(ARMS[0],ARMS[3]))


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def key(item):
    seed,day,mode = item
    return f's{seed}_d{day}_{mode}'


def contrasts(values):
    result = {name:{m:values[a][m]-values[b][m] for m in METRICS} for name,(a,b) in CONTRASTS.items()}
    result['interaction'] = {m:result['holding_at_learned'][m]-result['holding_at_fixed'][m] for m in METRICS}
    for m in METRICS:
        np.testing.assert_allclose(result['interaction'][m],
            result['upper_with_bilateral'][m]-result['upper_without_holding'][m],atol=1e-8,rtol=0)
    return result


class BenchmarkAudit(prior.TradeoffAudit):
    def __init__(self,seed,mode):
        assert mode in ARMS
        super().__init__(seed,mode.split('_',1)[1])
        self.arm,self.dispatches = mode,[]

    def attach(self,runner):
        super().attach(runner)
        env = runner.env
        assert env.fleet_inventory_mode == 'fixed_pool' and runner.cfg['upper']['N_fleet'] == 12
        assert pilot.step.ACTIONS.tolist() == [0.,5.,10.,15.,20.,30.,45.]
        self.scalars = (float(runner.lower_trainer.alpha),float(runner.lower_trainer.lambda_param),float(runner.upper_trainer.alpha))
        self.benchmark_launch = env.launch_bus
        def launch(trip,actual_launch=None):
            ready = {int(b.bus_id) for b in env.bus_in_terminal if b.direction == bool(trip.direction)}
            bus = self.benchmark_launch(trip,actual_launch=actual_launch)
            if bus is not None:
                assert int(bus.bus_id) in ready and len(env.bus_all) == 12
                self.dispatches.append(dict(trip=int(trip.launch_turn),bus=int(bus.bus_id),
                    direction=bool(trip.direction),actual_s=float(env.current_time),
                    public_s=float(getattr(trip,'_freqduet_base_launch',getattr(trip,'_original_launch',trip.launch_time))),
                    scheduled_s=float(getattr(trip,'_freqduet_scheduled_launch',trip.launch_time)),
                    eligible_s=float(trip._freqduet_first_eligible_time),target_s=float(trip.target_headway)))
            return bus
        env.launch_bus = launch

    def detach(self):
        self.env.launch_bus = self.benchmark_launch
        super().detach()

    def finish(self,runner,records,holds):
        env = runner.env
        assert self.scalars == (float(runner.lower_trainer.alpha),float(runner.lower_trainer.lambda_param),float(runner.upper_trainer.alpha))
        self.contract = prior.factorial.prior.physical_contract(env,runner.cfg)
        self.dispatch = prior.factorial.prior.dispatch_metrics(env)
        self.holding_person_s = float(sum(b.episode_hold_person_seconds for b in env.bus_all))
        commands = float(sum(h['action_s'] for h in holds))
        executed = float(sum(b.episode_hold_vehicle_seconds for b in env.bus_all))
        assert len(env.bus_all) == 12 and executed <= commands+1e-8
        cleared = env._done_reason == 'service_cleared'
        if cleared:
            np.testing.assert_allclose(commands,executed,atol=1e-8,rtol=0)
        self.execution = dict(commanded_s=commands,executed_s=executed,
            unexecuted_s=commands-executed,done_reason=env._done_reason,decisions=len(records),physical_vehicles=12)
        self.delays = dict(public_launch_delay_total_s=sum(max(0.,r['actual_s']-r['public_s']) for r in self.dispatches),
            readiness_delay_total_s=sum(r['actual_s']-r['eligible_s'] for r in self.dispatches))
        if self.arm.startswith('fixed_'):
            assert all(r['target_s'] == 360. for r in self.dispatches)
            for direction in (False,True):
                times = [r['scheduled_s'] for r in self.dispatches if r['direction'] == direction]
                np.testing.assert_allclose(np.diff(times),360.,atol=1e-8,rtol=0)
        if self.method == 'zero':
            assert executed == commands == 0.


def claim(root,source,name):
    assert name in {key(x) for x in ITEMS} | {'fixed_actor_check','external_fixed_reference'}
    write_new(root/'started'/(name+'.json'),dict(source=source,path=name,automatic_retry=False))


def collect(root,source,item,name=None):
    name = key(item) if name is None else name
    claim(root,source,name)
    seed,day,mode = item
    rule,credit = BenchmarkAudit(seed,mode),pilot.CreditObserver()
    row,events,_,holds = pilot.rollout(root,seed,day,'stochastic',187000000+seed+100*day,
        [0.]*4,'v267_'+name,probability_rule=rule,trace_observer=rule,credit_observer=credit)
    row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
        backlog_trip_s=float(credit.stream.total[4]),holding_person_s=rule.holding_person_s,**rule.dispatch,**rule.delays)
    trace = dict(holds=holds,dispatches=rule.dispatches,arrivals=rule.arrivals,departures=rule.departures)
    dest = root/'traces'/(name+'.pkl.gz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(dest,'xb') as stream:
        pickle.dump(trace,stream,protocol=pickle.HIGHEST_PROTOCOL)
    result = dict(source=source,seed=seed,day=day,mode=mode,results=row,contract=rule.contract,
        coverage=rule.coverage,execution=rule.execution,training_updates=0,main_policy_changed=False)
    write_new(root/'pairs'/(name+'.json'),result)
    print('V267_PATH_DONE '+name,flush=True)
    return result


def smoke(a):
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),pattern='test_v267_bilateral_factorial.py')
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.testsRun > 0 and tests.wasSuccessful()
    root = a.out_root
    write_new(root/'protocol.json',dict(source=a.source,actors=SEEDS,days=DAYS,arms=ARMS,
        formal_unique_paths=40,smoke_paths=6,smoke_formal_reuse=4,physical_start_cap=42,
        remaining_paths=36,max_workers=8,shards=SHARDS,automatic_retry=False,
        scope='Consumed V107 contexts; regime attribution, not fresh confirmation, tuning or frequency superiority.'))
    jobs = [(item,None) for item in SMOKE_MAIN]+[((SEEDS[1],DAYS[0],ARMS[2]),'fixed_actor_check')]
    with ProcessPoolExecutor(max_workers=5,mp_context=get_context('spawn')) as pool:
        results = [f.result() for f in [pool.submit(collect,root,a.source,item,name) for item,name in jobs]]
    for measured in results[:2]:
        method = measured['mode'].split('_',1)[1]
        i = frozen.ITEMS.index((SEEDS[0],DAYS[0],method,0))
        cached = read(a.results_root/'protocol_v6_v107_rule_confirmation/pairs'/f'p{i}.json')
        for metric in (*frozen.METRICS,'action_counts'):
            pilot.joint.assert_same_state(measured['results'][metric],cached['results'][metric],'V267_V107_cached_'+metric)
        pilot.joint.assert_same_state(measured['contract'],cached['contract'],'V267_V107_contract')
        pilot.joint.assert_same_state(measured['coverage'],cached['coverage'],'V267_V107_coverage')
    with gzip.open(root/'traces'/(key(SMOKE_MAIN[2])+'.pkl.gz'),'rb') as stream:
        first = pickle.load(stream)
    with gzip.open(root/'traces/fixed_actor_check.pkl.gz','rb') as stream:
        other = pickle.load(stream)
    pilot.joint.assert_same_state(first,other,'V267_fixed_bilateral_actor_invariance_actions_timing')
    for metric in METRICS:
        np.testing.assert_allclose(results[2]['results'][metric],results[4]['results'][metric],atol=1e-8,rtol=0)
    claim(root,a.source,'external_fixed_reference')
    reference = prior.factorial.prior.fixed_replay(root,DAYS[0],prior.factorial.prior.FixedExposure())
    for metric in frozen.METRICS:
        np.testing.assert_allclose(results[3]['results'][metric],reference['results'][metric],atol=1e-8,rtol=0,err_msg=metric)
    for measured in results:
        pilot.joint.assert_same_state(measured['contract'],reference['contract'],'V267_shared_physical_contract')
    write_new(root/'smoke/done.json',dict(source=a.source,status='v267_engineering_smoke_passed',
        tests=tests.testsRun,physical_starts=6,formal_paths_reusable=4,cached_V107_exact=True,
        fixed_bilateral_actor_invariance=True,external_fixed_zero_parity=True,
        outcomes_used_for_release=False,training_updates=0,main_policy_changed=False))
    print('V267_SMOKE_COMPLETE',flush=True)


def shard(a):
    gate = read(a.out_root/'smoke/done.json')
    assert gate['source'] == a.source and gate['status'] == 'v267_engineering_smoke_passed'
    for item in SHARDS[a.shard]:
        collect(a.out_root,a.source,item)
    write_new(a.out_root/'shards'/f'{a.shard}.json',dict(source=a.source,status='complete',items=SHARDS[a.shard],automatic_retry=False))


def aggregate(a):
    gate = read(a.out_root/'smoke/done.json')
    assert gate['source'] == a.source and gate['status'] == 'v267_engineering_smoke_passed'
    for shard_id,items in enumerate(SHARDS):
        saved = read(a.out_root/'shards'/f'{shard_id}.json')
        assert saved['source'] == a.source and saved['status'] == 'complete'
        assert saved['items'] == json.loads(json.dumps(items)) and not saved['automatic_retry']
    expected = {key(x) for x in ITEMS} | {'fixed_actor_check','external_fixed_reference'}
    assert {p.stem for p in (a.out_root/'started').glob('*.json')} == expected
    records = {item:read(a.out_root/'pairs'/(key(item)+'.json')) for item in ITEMS}
    for item,row in records.items():
        assert row['source'] == a.source and (row['seed'],row['day'],row['mode']) == item
    paired,absolute = [],[]
    for seed,day in itertools.product(SEEDS,DAYS):
        cell = {m:records[(seed if m.startswith('learned_') else SEEDS[0],day,m)] for m in ARMS}
        for row in cell.values():
            pilot.joint.assert_same_state(row['contract'],cell[ARMS[0]]['contract'],'V267_same_day_contract')
        values = {m:cell[m]['results'] for m in ARMS}
        paired.append(dict(seed=seed,day=day,effects=contrasts(values)))
        absolute.append(dict(seed=seed,day=day,values={m:{k:values[m][k] for k in METRICS} for m in ARMS}))
    stats = frozen.confirmation.describe(paired,'effects')
    result = dict(source=a.source,status='v267_bilateral_factorial_complete',effects=stats,
        absolute=frozen.confirmation.describe(absolute,'values'),
        axes={axis:{str(v):{name:{m:float(np.mean([r['effects'][name][m] for r in paired if r[axis] == v]))
            for m in KEYS} for name in paired[0]['effects']} for v in sorted({r[axis] for r in paired})} for axis in ('seed','day')},
        actors=SEEDS,consumed_days=DAYS,physical_paths=len(records),logical_actor_day_arm_cells=64,
        fixed_paths_shared_across_actors=True,bootstrap='Existing 5000-draw crossed actor/day bootstrap; all contrasts retain shared fixed day paths.',
        training_updates=0,main_policy_changed=False,controller_admitted=False,
        scope='Small consumed-day attribution pilot; no new-day confirmation, equivalence or learned frequency claim.')
    write_new(a.out_root/'summary/summary.json',result)
    print('V267_FACTORIAL_COMPLETE',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('smoke','shard','aggregate'))
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--shard',type=int,choices=range(8))
    a = p.parse_args()
    a.out_root = a.results_root/'protocol_v6_v267_bilateral_factorial'
    if a.command == 'smoke':
        smoke(a)
    elif a.command == 'shard':
        assert a.shard is not None
        shard(a)
    else:
        aggregate(a)


if __name__ == '__main__':
    main()
