"""V401 fresh checkpoints, paired collection and unchanged batch updates."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import traceback

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,separators=(',',':'),allow_nan=False,
                  default=lambda v:v.item());stream.write('\n')


def modules():
    global np,torch,frozen,new,recipe
    import numpy as np
    import torch
    import basis_worker as frozen
    import conditioned_integration as new
    import recipe


def helper(assets,name):
    spec=importlib.util.spec_from_file_location('v401_'+name,assets/'support'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def configuration(args):
    plan=read(HERE/'protocol.json');base=read(args.assets/'K4/config.json')
    approval=read(HERE/'authorization.json')
    assert approval['approved_physical_start_cap']==90
    assert approval['candidate_training_update_cap']==3072
    assert base['updates_per_batch']==plan['updates_per_batch']==192
    assert base['candidate_seed']==plan['learner_seed']==314011
    assert base['action_values']==plan['action_values']==[2,3]
    assert base['reward_units']['cv_weight']==0
    assert [d for batch in base['train_batches'] for d in batch]==list(range(1780001,1780033))
    assert read(args.root.parent/'metadata03/summary/ready.json')['passed']
    assert read(args.root.parent/'checks01/summary/results.json')['passed']
    assert read(args.root.parent/'entry01/summary/ready.json')['passed']
    if args.command!='path' or args.split!='engineering':
        assert read(args.root/'engineering_gate/summary/admitted.json')['passed']
    return plan,base


def load_policy(args,plan,base):
    receipt=read(args.checkpoint/'summary/results.json')
    saved=torch.load(args.checkpoint/'policy.pt',map_location='cpu',weights_only=False)
    assert saved['protocol']==plan and saved['base_config']==base and saved['arm']==args.arm
    assert saved['completed_batches']==receipt['completed_batches']
    assert saved['successful_updates']==saved['completed_batches']*192
    trainer=recipe.make_trainer(base,plan['learner_seed'])
    trainer.policy_net.load_state_dict(saved['policy'])
    frozen.action_library(trainer)
    return trainer,saved


def transitions_ok(rows,rho,arm):
    previous=None
    for i,row in enumerate(rows):
        state,next_state=np.asarray(row['state']),np.asarray(row['next_state'])
        assert state.shape==next_state.shape==(181,)
        assert row['budget_retention']==rho and row['budget_observation']==arm
        assert state[-1]==(rho if arm=='explicit' else 0.)
        expected=np.zeros(4,dtype=np.float32)
        if previous is not None:expected[previous]=1.
        np.testing.assert_array_equal(state[12:16],expected)
        action=int(row['action'][0]);assert action in (2,3)
        if row['done']:
            assert i==len(rows)-1
            np.testing.assert_array_equal(next_state,np.zeros(181,dtype=np.float32))
        else:np.testing.assert_array_equal(next_state,rows[i+1]['state'])
        previous=action
    assert sum(r['done'] for r in rows)==1


def semantics(manager):
    transitions_ok(manager.transitions,manager.retain_fraction,manager.budget_observation)
    events=[e for e in manager.events if e['kind']=='publish']
    assert len(events)==len(manager.transitions)==len(manager.choices)>0
    for event,row,choice in zip(events,manager.transitions,manager.choices):
        action=choice['action']
        assert event['learned_action']==int(row['action'][0])==action
        assert event['time_s']==row['start_s']==choice['time_s']
        assert choice['mode']==('budget' if action==2 else 'joint')
        assert not event['plans'] or action==3
        pi,mu=np.asarray(choice['pi']),np.asarray(choice['mu'])
        assert pi.shape==mu.shape==(2,) and np.isfinite(pi).all() and (pi>=0).all()
        np.testing.assert_allclose(pi.sum(),1.,atol=1e-6,rtol=0)
        if manager.deterministic:
            expected=np.zeros(2);expected[int(pi.argmax())]=1.
            assert action==2+int(pi.argmax())
        else:expected=.9*pi+.05
        np.testing.assert_allclose(mu,expected,atol=1e-7,rtol=0)
    return dict(passed=True,action_values=[2,3],records=len(events),state_dim=181,
                actor_critic_execution_labels_equal=True,previous_action_actual=True)


def bind(args):
    joint=frozen.base.fixed.pilot.joint
    joint.MODEL_ROOT=args.assets/'runtime/model';joint.TRAIN_ROOT=args.assets/'runtime/native_training'
    joint.optimized_runtime.__globals__['OPTIMIZED_ROOT']=Path(
        '/home/zhengliang01/scheduleurm_work/FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet')
    import runner_v3
    runner_v3.SCRIPT_DIR=joint.MODEL_ROOT


def case_location(args):
    if args.split=='engineering':return args.root/'engineering'/f'r{round(args.rho*100):03d}'
    if args.split=='train':return args.root/'train'/args.arm/f'b{args.batch}'/f'd{args.day}'
    return args.root/'development'/args.arm/f'r{round(args.rho*100):03d}'/f'd{args.day}'


def collect(args,plan,base):
    assert args.rho in plan['retentions']
    if args.split=='engineering':
        assert args.arm=='implicit' and args.day==2400001 and args.checkpoint is None
    elif args.split=='train':
        assert args.arm in plan['arms'] and args.batch in range(8)
        assert args.day in base['train_batches'][args.batch]
        assert args.rho==plan['rho_by_batch_position'][(args.day-plan['train_day_start'])%4]
        assert args.checkpoint is not None
    else:
        assert args.arm in plan['development_arms'] and args.day in plan['development_days_proposed']
        assert args.batch is None
    out=case_location(args);out.mkdir(parents=True,exist_ok=False)
    identity=dict(protocol_id=plan['protocol_id'],arm=args.arm,day=args.day,split=args.split,
        rho=args.rho,batch=args.batch,learner_seed=314011,checkpoint=str(args.checkpoint) if args.checkpoint else None,
        physical_starts=1,training_updates=0,main_policy_changed=False,automatic_retry=False)
    write(out/'started.json',identity)
    begin=time.perf_counter()
    try:
        modules()
        bind(args)
        if args.arm=='manual_joint':
            frozen.law.RETAIN_FRACTION=args.rho
            trainer=None;credit=frozen.base.fixed.pilot.CreditObserver()
            audit=frozen.base.Audit('joint',credit);label=f'v314_d{args.day}_joint'
        else:
            if args.split=='engineering':
                trainer,_=frozen.load_policy(args.assets/'K4',base,314011)
                # Reference-only adapter: compare old behavior, never warm-start
                # the new 181-channel learners from these weights.
                new.choose_upper_action=lambda t,state,*a:frozen.integration.choose_upper_action(t,state[:180],*a)
                deterministic=True
            else:
                trainer,saved=load_policy(args,plan,base)
                required=args.batch if args.split=='train' else 8
                assert saved['completed_batches']==required
                deterministic=args.split!='train'
            sample_seed=int(np.random.SeedSequence([314011,args.day]).generate_state(1,dtype=np.uint64)[0])
            generator=torch.Generator(device=trainer.device);generator.manual_seed(sample_seed)
            trainer.policy_net.set_sampling_state(generator.get_state())
            before=copy.deepcopy(trainer.policy_net.state_dict())
            credit=frozen.base.fixed.pilot.CreditObserver()
            audit=new.LearnedAudit(credit,trainer,new.RewardUnits(**base['reward_units']),
                recipe.divisors(base),deterministic,0. if deterministic else .1,
                retain_fraction=args.rho,budget_observation=args.arm,upper_fourier_k=4)
            label=(f'train_s314011_b{args.batch}_d{args.day}' if args.split=='train' else
                   f'engineering_k4_development_s314011_bNone_d{args.day}')
        runtime=args.root/'runtime'/f'{args.split}_{args.arm}_r{round(args.rho*100):03d}_d{args.day}'
        runtime.mkdir(parents=True,exist_ok=False)
        row,uniforms,_,holds=frozen.base.fixed.pilot.rollout(runtime,12011,args.day,'stochastic',27011001,
            [0.]*4,label,probability_rule=audit,trace_observer=audit,credit_observer=credit)
        row.update(waiting_person_s=float(credit.stream.total[0]),onboard_person_s=float(credit.stream.total[1]),
            backlog_trip_s=float(credit.stream.total[4]),holding_person_s=audit.holding_person_s,
            **audit.dispatch,**audit.delays)
        frozen.base.previous.label_resources(audit.bridge,audit.dispatches,holds)
        trace=dict(holds=holds,dispatches=audit.dispatches,arrivals=audit.arrivals,departures=audit.departures,
            queries=audit.decisions,uniforms=uniforms,returns=credit.stream.returns,
            graph_queries=audit.bridge.queries,resource_predictions=audit.bridge.predictions,
            slow_events=audit.manager.events,publish_ticks=audit.publish_ticks,public_clocks=audit.public_clocks)
        result=dict(identity,completed=True,status='complete_upper_day',actor=12011,sampler=27011001,
            results=row,contract=audit.contract,
            coverage=audit.coverage,execution=audit.execution,action_chain=audit.chain,
            graph=audit.bridge.summary(),slow=audit.manager.summary(),native_actors_unchanged=True,
            independent_upper_unchanged=True,confirmation_expanded=False)
        probes=[q['quota_probe'] for q in audit.bridge.queries if 'quota_probe' in q]
        result['quota_probes']=dict(attempted=len(probes),supported=sum(p['supported'] is True for p in probes),
            changed=sum(p['changed'] is True for p in probes),extra_evaluations=sum(p['evals'] for p in probes))
        if trainer is not None:
            assert trainer.successful_updates==0 and not len(trainer.replay_buffer)
            assert all(torch.equal(before[k],v) for k,v in trainer.policy_net.state_dict().items())
            result['action_semantics']=semantics(audit.manager);result['upper']=audit.manager.upper_receipt
            trace['upper_choices']=audit.manager.choices
            frozen.dump(out/'raw/upper_transitions.pkl.gz',dict(protocol=plan,base_config=base,identity=identity,
                transitions=audit.manager.transitions,upper=result['upper'],action_semantics=result['action_semantics']))
        frozen.dump(out/'raw/path.pkl.gz',trace)
        account=helper(args.assets,'v326').physical_account(helper(args.assets,'physical'),result,trace)
        quota=helper(args.assets,'budget_account').quota_account(trace,args.rho)
        assert account['passed'] and quota['passed']
        result.update(physical_account=account,quota_account=quota,wall_s=time.perf_counter()-begin,
            endpoints=helper(args.assets,'v324').endpoints(result,262,base['reward_units']))
        write(out/'summary/results.json',result)
        print('V401_PATH_COMPLETE',args.split,args.arm,args.day,args.rho,flush=True)
    except Exception:
        write(out/'summary/failure.json',dict(identity,error=traceback.format_exc()[-6500:]))
        raise
    finally:
        write(out/'summary/finished.json',dict(identity,completed=(out/'summary/results.json').exists()))


def checkpoint(args,plan,base):
    assert args.arm in plan['arms']
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    trainer=recipe.make_trainer(base,314011);frozen.action_library(trainer)
    args.active_trainer=trainer
    consumed=[];completed=0;added=0;metrics=[]
    if args.command=='update':
        prior=read(args.checkpoint/'summary/results.json')
        saved=torch.load(args.checkpoint/'training.pt',map_location='cpu',weights_only=False)
        assert saved['protocol']==plan and saved['base_config']==base and saved['arm']==args.arm
        assert prior['successful_updates']==saved['successful_updates']==saved['completed_batches']*192
        assert args.batch==saved['completed_batches'] and len(args.paths)==4
        trainer.load_training_state_dict(saved['trainer_state']);trainer.successful_updates=saved['successful_updates']
        completed=saved['completed_batches'];consumed=list(saved['consumed_days'])
        cells=[c for c in recipe.training_cells(plan,args.arm) if c['batch']==completed]
        assert consumed==[c['day'] for c in recipe.training_cells(plan,args.arm) if c['batch']<completed]
        receipts={}
        for path in args.paths:
            row=read(path/'summary/results.json')
            assert row['completed'] and row['split']=='train' and row['arm']==args.arm
            assert row['checkpoint']==str(args.checkpoint) and row['batch']==completed
            assert row['day'] not in receipts
            receipts[row['day']]=(row,path)
        assert set(receipts)=={c['day'] for c in cells}
        import gzip,pickle
        for cell in cells:
            row,path=receipts[cell['day']];assert row['rho']==cell['rho']
            with gzip.open(path/'raw/upper_transitions.pkl.gz','rb') as stream:raw=pickle.load(stream)
            assert raw['protocol']==plan and raw['base_config']==base
            assert raw['identity']['checkpoint']==str(args.checkpoint) and raw['identity']['arm']==args.arm
            transitions_ok(raw['transitions'],cell['rho'],args.arm)
            for transition in raw['transitions']:
                trainer.replay_buffer.push(*(transition[k] for k in
                    ('state','action','reward','next_state','done','duration_steps')))
            added+=len(raw['transitions']);consumed.append(cell['day'])
        for _ in range(192):
            measured=trainer.update(base['batch_size']);assert measured
            assert measured['outside_clip']==0 and measured['finite']
            metrics.append(measured)
        completed+=1
    else:
        assert args.command=='initialize' and args.batch is None and args.checkpoint is None
    assert trainer.successful_updates==completed*192
    envelope=dict(protocol=plan,base_config=base,arm=args.arm,learner_seed=314011,
        completed_batches=completed,consumed_days=consumed,successful_updates=trainer.successful_updates)
    for name,payload in (('policy.pt',dict(envelope,policy=trainer.policy_net.state_dict())),
                         ('training.pt',dict(envelope,trainer_state=trainer.training_state_dict()))):
        with (out/name).open('xb') as stream:torch.save(payload,stream)
    write(out/'summary/results.json',dict(completed=True,arm=args.arm,completed_batches=completed,
        successful_updates=trainer.successful_updates,updates_this_task=len(metrics),transitions_added=added,
        consumed_days=consumed,physical_starts=0,main_policy_changed=False,automatic_retry=False))
    write(out/'summary/ready.json',dict(completed=True,arm=args.arm,completed_batches=completed))
    print('V401_CHECKPOINT_COMPLETE',args.arm,completed,trainer.successful_updates,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('path','initialize','update'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--arm',choices=('explicit','implicit','manual_joint'),required=True)
    parser.add_argument('--split',choices=('engineering','train','development'))
    parser.add_argument('--day',type=int)
    parser.add_argument('--rho',type=float)
    parser.add_argument('--batch',type=int)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--paths',type=Path,nargs='*',default=[])
    args=parser.parse_args()
    plan,base=configuration(args)
    if args.command=='path':
        # Charge before importing the native control stack or constructing it.
        # collect() writes the authoritative path marker before bind/rollout.
        collect(args,plan,base)
    else:
        try:modules();checkpoint(args,plan,base)
        except Exception:
            write(args.out/'summary/failure.json',dict(completed=False,error=traceback.format_exc()[-6500:],
                successful_updates=getattr(getattr(args,'active_trainer',None),'successful_updates',0),
                physical_starts=0,automatic_retry=False,main_policy_changed=False));raise
