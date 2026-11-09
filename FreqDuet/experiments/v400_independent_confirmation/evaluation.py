"""Frozen single-case evaluation using the unmodified V350/V314 control classes."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent


def load_modules():
    global np, torch, frozen, basis_law
    import numpy as np
    import torch
    import basis_worker as frozen
    import basis_law


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False,
                  default=lambda v:v.item())
        stream.write('\n')


def helper(assets, name):
    path = assets/'support'/(name+'.py')
    spec = importlib.util.spec_from_file_location('v400_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bind(assets):
    runtime = assets/'runtime'
    joint = frozen.base.fixed.pilot.joint
    joint.MODEL_ROOT = runtime/'model'
    joint.TRAIN_ROOT = runtime/'native_training'
    joint.optimized_runtime.__globals__['OPTIMIZED_ROOT'] = Path(
        '/home/zhengliang01/scheduleurm_work/FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet')
    import runner_v3
    runner_v3.SCRIPT_DIR = runtime/'model'
    return runtime/'traffic'


def policy(assets, k):
    folder = assets/f'K{k}'
    config = read(folder/'config.json')
    # Keep the original checkpoint protocol intact; new-day eligibility belongs
    # to V400, not to the historical worker's four-day CLI restriction.
    trainer, envelope = frozen.load_policy(folder, config, 314011)
    assert envelope['completed_batches'] == 8 and envelope['successful_updates'] == 1536
    assert config['fast_actor'] == 12011 and config['fast_sampler'] == 27011001
    assert config['candidate_seed'] == 314011 and config['action_values'] == [2, 3]
    assert config['reward_units']['cv_weight'] == 0
    return config, trainer


def validate_case(plan, arm, day, engineering):
    arms = {a['id']:a for a in plan['arms']}
    assert arm in arms
    if engineering:
        assert day == 2077002 and arm in ('K4_r050', 'manual_r050')
    else:
        assert day in plan['days']
    return arms[arm]


def run(args):
    plan = read(HERE/'manifest.json')
    authorization = read(HERE/'authorization.json')
    assert authorization['approved_physical_start_cap'] == 82
    assert read(args.root/'runtime_check/summary/ready.json')['passed']
    if not args.engineering:
        assert read(args.root/'engineering_gate/summary/admitted.json')['passed']
    spec = validate_case(plan, args.arm, args.day, args.engineering)
    out = args.root/('engineering' if args.engineering else 'cases')/f'{args.arm}_d{args.day}'
    out.mkdir(parents=True, exist_ok=False)
    identity = dict(protocol_id=plan['protocol_id'], day=args.day, arm=args.arm,
        split='engineering' if args.engineering else 'confirmation', learner_seed=314011,
        budget_retain_fraction=spec['rho'], upper_fourier_k=spec['K'],
        checkpoint=str(args.assets/f'K{spec["K"]}') if spec['upper']=='learned' else None,
        physical_starts=1, training_updates=0, main_policy_changed=False,
        confirmation_expanded=False, automatic_retry=False)
    # One case attempt is charged before any controller or native runner setup.
    write(out/'started.json', identity)
    start = time.perf_counter()
    try:
        load_modules()
        traffic = bind(args.assets)
        if spec['upper'] == 'learned':
            config, trainer = policy(args.assets, spec['K'])
            sample_seed = int(np.random.SeedSequence([314011,args.day]).generate_state(1,dtype=np.uint64)[0])
            generator = torch.Generator(device=trainer.device)
            generator.manual_seed(sample_seed)
            trainer.policy_net.set_sampling_state(generator.get_state())
            before = copy.deepcopy(trainer.policy_net.state_dict())
            basis_law.RETAIN_FRACTION = spec['rho']
            credit = frozen.base.fixed.pilot.CreditObserver()
            audit = frozen.LearnedAudit(credit, trainer, frozen.RewardUnits(**config['reward_units']),
                config['state_divisors'], True, 0., upper_fourier_k=spec['K'])
            label = f'development_s314011_bNone_d{args.day}'
            if spec['K'] == 4:
                label = 'engineering_k4_'+label
        else:
            config = read(args.assets/'K4/config.json')
            frozen.law.RETAIN_FRACTION = spec['rho']
            credit = frozen.base.fixed.pilot.CreditObserver()
            audit = frozen.base.Audit('joint', credit)
            label = f'v314_d{args.day}_joint'
        runtime = traffic/f'v400_{args.arm}_d{args.day}'
        runtime.mkdir(exist_ok=False)
        row, uniforms, _, holds = frozen.base.fixed.pilot.rollout(runtime, 12011,
            args.day, 'stochastic', 27011001, [0.]*4, label,
            probability_rule=audit, trace_observer=audit, credit_observer=credit)
        assert audit.manager is audit.bridge.manager
        row.update(waiting_person_s=float(credit.stream.total[0]),
            onboard_person_s=float(credit.stream.total[1]), backlog_trip_s=float(credit.stream.total[4]),
            holding_person_s=audit.holding_person_s, **audit.dispatch, **audit.delays)
        frozen.base.previous.label_resources(audit.bridge, audit.dispatches, holds)
        trace = dict(holds=holds, dispatches=audit.dispatches, arrivals=audit.arrivals,
            departures=audit.departures, queries=audit.decisions, uniforms=uniforms,
            returns=credit.stream.returns, graph_queries=audit.bridge.queries,
            resource_predictions=audit.bridge.predictions, slow_events=audit.manager.events,
            publish_ticks=audit.publish_ticks, public_clocks=audit.public_clocks)
        result = dict(identity, completed=True, results=row, contract=audit.contract,
            coverage=audit.coverage, execution=audit.execution, action_chain=audit.chain,
            graph=audit.bridge.summary(), slow=audit.manager.summary(),
            actor=12011, sampler=27011001, controller_reference='same-plan budget-conditioned native',
            status='complete_upper_day' if spec['upper']=='learned' else 'complete_development_path')
        probes = [q['quota_probe'] for q in audit.bridge.queries if 'quota_probe' in q]
        result['quota_probes'] = dict(attempted=len(probes), supported=sum(p['supported'] is True for p in probes),
            changed=sum(p['changed'] is True for p in probes), extra_evaluations=sum(p['evals'] for p in probes))
        if spec['upper'] == 'learned':
            assert trainer.successful_updates == 0 and not len(trainer.replay_buffer)
            assert all(torch.equal(before[k],v) for k,v in trainer.policy_net.state_dict().items())
            assert audit.manager.upper_provider.k == spec['K'] and basis_law.RETAIN_FRACTION == spec['rho']
            semantics = frozen.action_semantics(audit.manager)
            trace['upper_choices'] = audit.manager.choices
            saved = dict(identity=identity, protocol=config, upper=audit.manager.upper_receipt,
                         transitions=audit.manager.transitions, action_semantics=semantics)
            frozen.dump(out/'raw/upper_transitions.pkl.gz', saved)
            result.update(upper=audit.manager.upper_receipt, action_semantics=semantics,
                          native_actors_unchanged=True, independent_upper_unchanged=True)
        else:
            assert type(audit.manager) is frozen.law.SlowManager and frozen.law.RETAIN_FRACTION == spec['rho']
        frozen.dump(out/'raw/path.pkl.gz', trace)
        physical = helper(args.assets,'physical')
        account = helper(args.assets,'v326').physical_account(physical, result, trace)
        assert account['passed'], account
        quota = helper(args.assets,'budget_account').quota_account(trace, spec['rho'])
        if spec['upper'] == 'learned':
            actual = helper(args.assets,'v326').actual_action_account(frozen, result, trace, saved)
            credit_check = helper(args.assets,'credit').credit_check(result,trace,saved,config)
            assert actual['passed'] and credit_check['passed']
        result.update(physical_account=account, quota_account=quota,
            endpoints=helper(args.assets,'v324').endpoints(result,len(trace['dispatches']),config['reward_units']),
            wall_s=time.perf_counter()-start)
        write(out/'summary/results.json', result)
        print('V400_CASE_COMPLETE',args.arm,args.day,flush=True)
    except Exception:
        write(out/'summary/failure.json', dict(identity, completed=False, error=traceback.format_exc()[-5000:]))
        raise
    finally:
        write(out/'summary/finished.json', dict(identity,
            completed=(out/'summary/results.json').exists(), wall_s=time.perf_counter()-start))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--arm', required=True)
    parser.add_argument('--day', type=int, required=True)
    parser.add_argument('--engineering', action='store_true')
    run(parser.parse_args())
