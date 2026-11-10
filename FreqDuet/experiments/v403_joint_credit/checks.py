"""Zero-traffic admission, common-reference and irreversible-ledger contracts."""

import argparse
from collections import Counter
import copy
from pathlib import Path
import shutil
from types import SimpleNamespace as NS
from unittest.mock import patch

from worker import HERE,read,write,bind


def result(cost=100., cv2=.05, delay=10.):
    return dict(cost=cost,cv2=cv2,public_delays={2:delay},returns={0:100.},
                next_launch={0:120.},delays={2:delay},failure=None,coverage=True,
                missing_commands=set(),actions={(0,1,2):(0.,10)})


def manager():
    from joint_credit import JointManager
    m=object.__new__(JointManager)
    m.kind='candidate';m.arm='joint';m.retain_fraction=.5
    m.env=NS(current_time=100.);m.pending={};m.remaining={};m.ledger=[];m.admissions=[]
    m.capped_cycles=set()
    m.request_counts=Counter();m.command_count=0
    m.active={(0,1):dict(command_s=0.,residence_s=0.,commands=0)}
    m.budgeted_commands=m.budgeted_command_s=m.budgeted_residence_s=0
    return m


def run(args):
    from joint_credit import joint_choice
    from accounting import credit_account
    tests={}
    legal=(0,5,10,15,20,30,45)
    own=NS(bus=0,trip=1,direction=False,index=0,legal=legal)
    snapshot=NS(current_bus=0,vehicles=[own],routes={False:[NS(station=2)]})
    baseline=dict(action_s=10,commands={(0,1,2):10},selected=result(),
                  reference=result(),evaluations=7,solver_proposals=7)
    def choose(model,homogeneous=False):return copy.deepcopy(baseline)
    def evaluate(commands,homogeneous=False):
        h=commands[(0,1,2)]
        return result(90.,.06) if h==0 else result(94.,.049,8.)
    model=NS(snapshot=snapshot,budget_remaining={},evaluate=evaluate)
    with patch('joint_credit.choose_coupled',choose):
        selected,accepted,receipt=joint_choice(model,5)
        assert accepted and selected['action_s']==5 and receipt['cv2_delta']<0
        assert not model.budget_remaining and selected['reference']['cv2']==.05
        tests['tightened_native_cannot_replace_common_reference']=True
        for field,value in (('cv2',.06),('cost',101.),('coverage',False),
                            ('returns',{0:101.}),('next_launch',{0:121.}),('delays',{2:11.})):
            def bad(commands,homogeneous=False):
                row=result(90.,.049,8.);row[field]=value;return row
            model.evaluate=bad
            selected,accepted,_=joint_choice(model,5)
            assert not accepted and selected['action_s']==10
        tests['rejected_credit_does_not_force_clipped_current_action']=True
        model.evaluate=lambda commands,homogeneous=False:result(100.,.049,8.)
        selected,accepted,receipt=joint_choice(model,5)
        assert accepted and receipt['cost_delta']==0 and receipt['public_delay_delta']==-2
        tests['unchanged_cost_tie_uses_public_delay_without_new_weight']=True
    item=dict(bus=0,trip=1,remaining_controls=3,normal_command_s=30.,ordinary_return_s=9000.)
    matches={2:dict(bus=0,active_trip=1,public_s=8990.)}
    m=manager();assert m.issue_budgets({(0,1):item},matches)==[] and not m.remaining
    m.commit(0,1,2,100.,5)
    row,amount=m.request((0,1));assert amount==14 and row['spent_since_proposal_s']==5
    m.admit((0,1),row,amount,dict(cost_delta=-1.,cv2_delta=-.001))
    m.commit(0,1,3,100.,10)
    assert m.remaining[(0,1)]==4
    m.ledger.append(dict(kind='return',bus=0,trip=1,time_s=200.,unused_s=4))
    m.remaining.clear()
    queries=[dict(bus=0,trip=1,station=st,time_s=100.,executed_request_s=h,legal=legal)
             for st,h in ((2,5),(3,10))]
    account=credit_account(dict(credit_ledger=m.ledger,queries=queries),m)
    assert account['issued_s']==14 and account['spent_s']==10 and account['unused_s']==4
    tests['same_tick_query_order_and_pre_admission_spend_conserved']=True
    m=manager();m.issue_budgets({(0,1):item},matches)
    m.commit(0,1,2,100.,30)
    assert m.request((0,1)) is None and not m.remaining
    tests['already_spent_commands_cannot_be_retroactively_capped']=True
    m=manager();m.remaining[(0,1)]=10
    assert not m.issue_budgets({(0,1):item},matches) and not m.pending and m.remaining[(0,1)]==10
    tests['existing_credit_never_increases']=True
    plan=read(HERE/'protocol.json')
    sealed=read(args.root.parent.parent/'FreqDuet-v402-budget-mechanism/run01/selection/summary/results.json')
    for family in ('static','cap'):
        assert plan[family+'_high_probability']==sealed['families'][family]['components'][1]['probability']
    tests['simple_comparator_probabilities_unchanged']=True
    for name in ('benchmark_summary.json','training_equivalence/summary.json'):
        target=args.root/'runtime/runtime_optimization_20261014'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.assets/'runtime/traffic/runtime_optimization_20261014'/name,target)
    frozen=bind(args.assets);joint=frozen.base.fixed.pilot.joint
    _,_,load_config,_=joint.optimized_runtime(args.root/'runtime/import_check')
    load_config(str(joint.MODEL_ROOT/'configs_freqduet/F_freqduet_protocol_v6_confirmed_main_hiro.yaml'))
    tests['native_runtime_proofs_and_config_ready']=True
    output=dict(completed=True,passed=True,checks=tests,physical_starts=0,
                native_runner_constructions=0,training_updates=0)
    write(args.root/'checks/summary/results.json',output)
    write(args.root/'checks/summary/ready.json',output)
    print('V403_CHECKS_PASSED',len(tests),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);run(p.parse_args())
