"""Zero-traffic contracts for copied law, permanent credits and future stop caps."""

import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace

from worker import read,write,bind


def run(args):
    from policies import Manager,CappedCommands,project_native
    from base_law import SlowManager,quota
    results={}
    for native in (0,5,10,15,20,30,45):
        for cap in (0,5,10,15,20,30,45):
            commands=CappedCommands({}, {},{(0,1):cap})
            key=(0,1,2);commands.legal[key]=(0,5,10,15,20,30,45)
            assert commands.get(key,native)==min(native,cap)
            future=(0,1,3);commands.legal[future]=commands.legal[key]
            assert commands.get(future,native)<=cap
    results['current_and_future_cap_projection']=True
    commands=CappedCommands({}, {(0,1):12},{(0,1):10})
    for station,expected in ((2,10),(3,0)):
        key=(0,1,station);commands.legal[key]=(0,5,10,15,20,30,45)
        assert commands.get(key,45)==expected
    assert commands.remaining[(0,1)]==2
    results['cap_and_trip_credit_compose']=True
    m=object.__new__(Manager);m.kind='static';m.parameter=5;m.remaining={};m.capped_cycles=set()
    m.cap_issues=[];m.command_count=0;m.env=SimpleNamespace(current_time=7200)
    item=dict(bus=0,trip=1,remaining_controls=3,normal_command_s=30,
              ordinary_return_s=9000,direction=False)
    issued=m.issue_budgets({(0,1):item},{})
    assert len(issued)==1 and m.remaining[(0,1)]==15
    m.remaining[(0,1)]-=10
    assert not m.issue_budgets({(0,1):dict(item,remaining_controls=9)},{})
    assert m.remaining[(0,1)]==5
    results['static_credit_never_refilled']=True
    m.kind='cap';m.parameter=10
    assert not m.issue_budgets({(0,1):item},{})
    assert m.cap_issues[0]['after_queries']==0 and m.capped_cycles=={(0,1)}
    assert not m.issue_budgets({(0,1):item},{}) and len(m.cap_issues)==1
    results['cap_admission_is_causal_and_once']=True
    m.kind='pressure';m.arm='joint';m.retain_fraction=.5;m.remaining={}
    match={2:dict(bus=0,active_trip=1,public_s=8990)}
    a=m.issue_budgets({(0,1):item},match)
    assert a[0]['remaining_s']==quota(30,11,.5)==19
    results['pressure_formula_unchanged']=True
    for name in ('benchmark_summary.json','training_equivalence/summary.json'):
        target=args.root/'runtime/runtime_optimization_20261014'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.assets/'runtime/traffic/runtime_optimization_20261014'/name,target)
    frozen=bind(args.assets)
    joint=frozen.base.fixed.pilot.joint
    _,_,load_config,_=joint.optimized_runtime(args.root/'runtime/import_check')
    load_config(str(joint.MODEL_ROOT/'configs_freqduet/F_freqduet_protocol_v6_confirmed_main_hiro.yaml'))
    for name in ('passenger_OD.xlsx','stop_news.xlsx','route_news.xlsx','time_table.xlsx'):
        assert (joint.MODEL_ROOT/'env/data'/name).is_file()
    results['native_runtime_and_output_proofs']=True
    value=dict(completed=True,passed=all(results.values()),checks=results,physical_starts=0,
               native_runner_constructions=0,training_updates=0,main_policy_changed=False)
    write(args.root/'checks/summary/results.json',value);write(args.root/'checks/summary/ready.json',value)
    print('V402_SYNTHETIC_PASSED',len(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);run(p.parse_args())
