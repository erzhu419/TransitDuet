"""Compare the two reference-only paths against the frozen V400 artifacts."""

import argparse
from pathlib import Path
import traceback

from worker import read,write,helper


def run(args):
    assert read(args.root.parent/'metadata03/summary/ready.json')['passed']
    plan=read(Path(__file__).parent/'protocol.json')
    if args.engineering_dir=='engineering_retry01':
        assert read(Path(__file__).parent/'engineering_restart_authorization.json')['effective_physical_start_cap']==92
    gate=helper(args.assets,'gate_default')
    checks={};roundoff=[]
    for reference in plan['engineering_references']:
        rho=reference['rho']
        new=args.root/args.engineering_dir/f'r{round(rho*100):03d}'
        old=args.reference/'cases'/f'{reference["arm"]}_d{reference["day"]}'
        left,right=read(old/'summary/results.json'),read(new/'summary/results.json')
        assert right['completed'] and right['physical_account']['passed'] and right['quota_account']['passed']
        a,b=gate.raw(old/'raw/path.pkl.gz'),gate.raw(new/'raw/path.pkl.gz')
        differences={}
        for key in ('results','contract','coverage','execution','action_chain','slow','quota_probes'):
            diff=gate.difference(left[key],right[key],key,roundoff)
            if diff is not None:differences[key]=diff
        for key in ('holds','dispatches','arrivals','departures','queries','uniforms','returns',
                    'slow_events','publish_ticks','public_clocks','upper_choices'):
            diff=gate.difference(a[key],b[key],key,roundoff)
            if diff is not None:differences['trace_'+key]=diff
        upper={k:v for k,v in right['upper'].items() if k not in ('budget_retention','budget_observation')}
        diff=gate.difference(left['upper'],upper,'upper',roundoff)
        if diff is not None:differences['upper']=diff
        saved_old=gate.raw(old/'raw/upper_transitions.pkl.gz')
        saved_new=gate.raw(new/'raw/upper_transitions.pkl.gz')
        transitions=[]
        for row in saved_new['transitions']:
            item={k:v for k,v in row.items() if k not in ('budget_retention','budget_observation')}
            item['state']=item['state'][:180];item['next_state']=item['next_state'][:180]
            transitions.append(item)
        diff=gate.difference(saved_old['transitions'],transitions,'transitions',roundoff)
        if diff is not None:differences['transitions']=diff
        checks[str(rho)]=dict(passed=not differences,differences=differences)
    passed=all(v['passed'] for v in checks.values())
    charged=sum(1 for folder in ('engineering','engineering_retry01')
                for _ in (args.root/folder).glob('*/started.json'))
    result=dict(completed=True,passed=passed,checks=checks,charged_engineering_paths=charged,
        engineering_directory=args.engineering_dir,
        allowed_diagnostic_roundoff=len(roundoff),roundoff_examples=roundoff[:2],
        physical_starts=0,training_updates=0,main_policy_changed=False)
    write(args.root/'engineering_gate/summary/results.json',result)
    if passed:write(args.root/'engineering_gate/summary/admitted.json',result)
    print('V401_ENGINEERING_GATE',passed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--engineering-dir',choices=('engineering','engineering_retry01'),default='engineering')
    args=parser.parse_args()
    try:run(args)
    except Exception:
        name='failure_retry01.json' if args.engineering_dir=='engineering_retry01' else 'failure.json'
        write(args.root/'engineering_gate/summary'/name,dict(completed=False,
            error=traceback.format_exc()[-6500:],physical_starts=0));raise
