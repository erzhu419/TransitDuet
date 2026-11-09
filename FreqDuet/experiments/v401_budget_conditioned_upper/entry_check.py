"""Exercise checkpoint I/O and the path-entry contracts without traffic/updates."""

import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS

import numpy as np
import torch

import worker
from recipe import make_trainer


def run(args):
    worker.modules()
    plan=worker.read(Path(__file__).parent/'protocol.json')
    base=worker.read(args.assets/'K4/config.json')
    checks={}
    with tempfile.TemporaryDirectory(prefix='v401-entry-') as folder:
        states=[]
        for arm in ('explicit','implicit'):
            init=NS(arm=arm,out=Path(folder)/arm,command='initialize',batch=None,checkpoint=None)
            worker.checkpoint(init,plan,base)
            trainer,saved=worker.load_policy(NS(checkpoint=init.out,arm=arm),plan,base)
            assert saved['successful_updates']==saved['completed_batches']==0
            full=torch.load(init.out/'training.pt',map_location='cpu',weights_only=False)
            restored=make_trainer(base,314011)
            restored.load_training_state_dict(full['trainer_state'])
            for key,value in trainer.policy_net.state_dict().items():
                assert torch.equal(value,restored.policy_net.state_dict()[key])
            assert not len(restored.replay_buffer) and restored.successful_updates==0
            states.append(copy.deepcopy(trainer.policy_net.state_dict()))
        assert all(torch.equal(states[0][k],states[1][k]) for k in states[0])
        checks['fresh_checkpoint_roundtrip_and_matched_arms']=True
    for arm in ('explicit','implicit'):
        for rho in (.25,.5):
            first=np.zeros(181,dtype=np.float32);first[-1]=rho if arm=='explicit' else 0.
            second=first.copy();second[14]=1.
            rows=[]
            for i,(state,action,done) in enumerate(((first,2,False),(second,3,True))):
                rows.append(dict(state=state,action=np.asarray([action],dtype=np.float32),done=done,
                    next_state=np.zeros(181,dtype=np.float32) if done else second,
                    budget_retention=rho,budget_observation=arm))
            worker.transitions_ok(rows,rho,arm)
            rows[0]['budget_retention']=.5 if rho==.25 else .25
            try:worker.transitions_ok(rows,rho,arm)
            except AssertionError:pass
            else:raise AssertionError('Mixed-budget replay was not rejected')
    checks['paired_replay_budget_labels_and_terminal_shape']=True
    gate=worker.helper(args.assets,'gate_default')
    trace=gate.raw(args.assets/'reference/K4_r050/raw/path.pkl.gz')
    raw=gate.raw(args.assets/'reference/K4_r050/raw/upper_transitions.pkl.gz')
    trainer,_=worker.frozen.load_policy(args.assets/'K4',base,314011)
    for choice,transition in zip(trace['upper_choices'],raw['transitions']):
        state=np.asarray([*transition['state'],.5],dtype=np.float32)
        action,pi,_=worker.frozen.integration.choose_upper_action(trainer,state[:180],True,0.)
        assert action==choice['action']
        np.testing.assert_array_equal(pi,choice['pi'])
    assert len(trace['upper_choices'])==len(raw['transitions'])==47
    checks['old_actor_180_prefix_reference_adapter']=True
    result=dict(completed=True,passed=True,checks=checks,old_actor_reference_calls=47,
        physical_starts=0,native_runner_constructions=0,optimizer_updates=0,
        candidate_training_updates=0,main_policy_changed=False)
    worker.write(args.out/'results.json',result);worker.write(args.out/'ready.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception:
        import traceback
        worker.write(args.out/'failure.json',dict(completed=False,physical_starts=0,
            optimizer_updates=0,error=traceback.format_exc()[-6500:]));raise
