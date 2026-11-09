"""Frozen-runtime admission and old-day engineering equivalence; zero new rollouts."""

import argparse
from pathlib import Path
import traceback

import numpy as np
import torch

import basis_worker as frozen
import basis_law
from evaluation import read, write, helper, bind, policy, validate_case, load_modules, HERE


def runtime_check(args):
    load_modules()
    plan = read(HERE/'manifest.json')
    traffic = bind(args.assets)
    assert Path(frozen.__file__).parent == args.assets/'basis'
    assert frozen.integration.LearnedManager.__bases__ == (basis_law.SlowManager,)
    assert frozen.base.SlowManager is frozen.law.SlowManager
    expected = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v4-worktree/FreqDuet/experiments/v314_dynamic_upper/source03')
    assert Path(frozen.base.__file__).parent == Path(frozen.law.__file__).parent == expected
    runtime = traffic/'v400_import_check'
    runtime.mkdir(exist_ok=False)
    joint = frozen.base.fixed.pilot.joint
    _, _, load_config, _ = joint.optimized_runtime(runtime)
    load_config(str(joint.MODEL_ROOT/'configs_freqduet/F_freqduet_protocol_v6_confirmed_main_hiro.yaml'))
    for name in ('passenger_OD.xlsx','stop_news.xlsx','route_news.xlsx','time_table.xlsx'):
        assert (joint.MODEL_ROOT/'env/data'/name).is_file(), name
    configs, trainers = {}, {}
    for k in (0,4):
        configs[k], trainers[k] = policy(args.assets,k)
    for key in ('reward_units','state_divisors','train_batches','trainer_settings','action_values'):
        assert configs[0][key] == configs[4][key], key
    assert configs[0]['upper_fourier_k'] == 0
    gate = helper(args.assets,'gate_default')
    old = gate.raw(args.assets/'reference/K4_r050/raw/path.pkl.gz')
    saved = gate.raw(args.assets/'reference/K4_r050/raw/upper_transitions.pkl.gz')
    max_error = 0.
    for choice, transition in zip(old['upper_choices'],saved['transitions']):
        state = torch.as_tensor(transition['state'],dtype=torch.float32,device=trainers[4].device)
        with torch.no_grad():
            pi = trainers[4].policy_net.dist_info(state[None])[0][0].cpu().numpy()
        max_error = max(max_error,float(np.max(abs(pi-np.asarray(choice['pi'])))))
    assert len(old['upper_choices']) == len(saved['transitions']) > 0
    assert max_error == 0., max_error
    for arm in plan['arms']:
        for day in plan['days']:
            validate_case(plan,arm['id'],day,False)
    physical = helper(args.assets,'physical')
    for arm in ('K4_r050','manual_r050'):
        row = read(args.assets/'reference'/arm/'summary/results.json')
        trace = gate.raw(args.assets/'reference'/arm/'raw/path.pkl.gz')
        assert helper(args.assets,'v326').physical_account(physical,row,trace)['passed']
        assert helper(args.assets,'budget_account').quota_account(trace,.5)['passed']
    result = dict(completed=True, passed=True, physical_starts=0, native_runner_constructions=0,
        environment_steps=0, training_updates=0, actor_calls=len(old['upper_choices']),
        actor_probability_max_abs_error=max_error, heldout_cases_validated=80,
        new_days_not_consumed=True, main_policy_changed=False)
    write(args.root/'runtime_check/summary/results.json',result)
    write(args.root/'runtime_check/summary/ready.json',result)
    print('V400_RUNTIME_ADMITTED',flush=True)


def engineering_check(args):
    metadata = read(args.root.parent/'preflight01/summary/results.json')
    assert metadata['completed'] and metadata['freshness_verified_across_both_servers']
    gate = helper(args.assets,'gate_default')
    checks, tolerances = {}, []
    for arm in ('K4_r050','manual_r050'):
        folder = args.root/'engineering'/f'{arm}_d2077002'
        if not (folder/'summary/results.json').exists():
            checks[arm] = {'passed':False,'reason':'charged engineering path did not complete'}
            continue
        old = read(args.assets/'reference'/arm/'summary/results.json')
        new = read(folder/'summary/results.json')
        left = gate.raw(args.assets/'reference'/arm/'raw/path.pkl.gz')
        right = gate.raw(folder/'raw/path.pkl.gz')
        differences = {}
        for key in ('results','contract','coverage','execution','action_chain','slow','quota_probes'):
            diff = gate.difference(old[key],new[key],key,tolerances)
            if diff is not None:
                differences['summary_'+key] = diff
        for key in ('holds','dispatches','arrivals','departures','queries','uniforms','returns',
                    'slow_events','publish_ticks','public_clocks'):
            diff = gate.difference(left[key],right[key],key,tolerances)
            if diff is not None:
                differences['trace_'+key] = diff
        if arm == 'K4_r050':
            for key in ('upper','action_semantics'):
                diff = gate.difference(old[key],new[key],key,tolerances)
                if diff is not None:
                    differences[key] = diff
            old_saved = gate.raw(args.assets/'reference'/arm/'raw/upper_transitions.pkl.gz')
            new_saved = gate.raw(folder/'raw/upper_transitions.pkl.gz')
            for key,a,b in [('transitions',old_saved['transitions'],new_saved['transitions']),
                            ('upper_choices',left['upper_choices'],right['upper_choices'])]:
                diff = gate.difference(a,b,key,tolerances)
                if diff is not None:
                    differences[key] = diff
        checks[arm] = {'passed':not differences,'differences':differences}
    passed = all(c['passed'] for c in checks.values())
    result = dict(completed=True,passed=passed,checks=checks,
        allowed_diagnostic_roundoff=len(tolerances),roundoff_examples=tolerances[:2],
        physical_starts=0,charged_engineering_paths=2,training_updates=0,main_policy_changed=False)
    write(args.root/'engineering_gate/summary/results.json',result)
    if passed:
        write(args.root/'engineering_gate/summary/admitted.json',result)
    print('V400_ENGINEERING_GATE',passed,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('runtime','engineering'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    args=parser.parse_args()
    try:
        (runtime_check if args.phase=='runtime' else engineering_check)(args)
    except Exception:
        destination='runtime_check' if args.phase=='runtime' else 'engineering_gate'
        write(args.root/destination/'summary/failure.json',dict(completed=False,passed=False,
              physical_starts=0,training_updates=0,error=traceback.format_exc()[-5000:]))
        raise
