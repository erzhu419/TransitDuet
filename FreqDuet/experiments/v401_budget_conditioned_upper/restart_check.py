"""Check replacement admission and distinct paths without constructing a runner."""

import argparse
from types import SimpleNamespace

import worker


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=worker.Path,required=True)
    parser.add_argument('--assets',type=worker.Path,required=True)
    parser.add_argument('--out',type=worker.Path,required=True)
    args=parser.parse_args()
    amendment=worker.read(worker.HERE/'engineering_restart_authorization.json')
    assert amendment['effective_physical_start_cap']==90+2
    assert amendment['training_paths']==64 and amendment['development_paths']==24
    assert amendment['candidate_training_update_cap']==3072
    assert not list(args.root.glob('train/*/*/*/started.json'))
    assert not list(args.root.glob('development/*/*/*/started.json'))
    assert not (args.root/'engineering_gate/summary/admitted.json').exists()
    for rho in (.25,.5):
        case=SimpleNamespace(**vars(args),command='path',split='engineering',arm='implicit',
                             day=2400001,rho=rho,engineering_attempt=2)
        worker.configuration(case)
        old=args.root/'engineering'/f'r{round(rho*100):03d}'
        assert worker.read(old/'started.json')['physical_starts']==1
        assert not worker.read(old/'summary/finished.json')['completed']
        assert (old/'summary/failure.json').is_file()
        replacement=worker.case_location(case)
        assert replacement==args.root/'engineering_retry01'/old.name
        assert not replacement.exists()
        runtime=args.root/'runtime'/f'engineering_implicit_r{round(rho*100):03d}_d2400001_retry01'
        assert not runtime.exists()
        for name in ('benchmark_summary.json','training_equivalence/summary.json'):
            assert (runtime.parent/'runtime_optimization_20261014'/name).is_file()
        case.engineering_attempt=1
        assert worker.case_location(case)==old
    for name in ('worker.py','engineering_gate.py','development.py'):
        compile((worker.HERE/name).read_bytes(),name,'exec')
    result=dict(completed=True,passed=True,original_failed_starts_preserved=2,
                approved_total_start_cap=92,physical_starts=0,native_runner_constructions=0,
                optimizer_updates=0,main_policy_changed=False)
    worker.write(args.out/'results.json',result);worker.write(args.out/'ready.json',result)
    print('V401_ENGINEERING_RESTART_READY',flush=True)
