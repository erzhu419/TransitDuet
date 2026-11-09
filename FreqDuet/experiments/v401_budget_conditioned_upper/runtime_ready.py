"""Bind existing runtime proofs at the new output root; construct no runner."""

import argparse
from pathlib import Path
import shutil

import worker


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    for name in ('benchmark_summary.json','training_equivalence/summary.json'):
        source=args.assets/'runtime/traffic/runtime_optimization_20261014'/name
        target=args.root/'runtime/runtime_optimization_20261014'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert worker.read(target)==worker.read(source)
        else:shutil.copyfile(source,target)
    worker.modules();worker.bind(args)
    joint=worker.frozen.base.fixed.pilot.joint
    _,_,load_config,_=joint.optimized_runtime(args.root/'runtime/import_check')
    load_config(str(joint.MODEL_ROOT/'configs_freqduet/F_freqduet_protocol_v6_confirmed_main_hiro.yaml'))
    for name in ('passenger_OD.xlsx','stop_news.xlsx','route_news.xlsx','time_table.xlsx'):
        assert (joint.MODEL_ROOT/'env/data'/name).is_file()
    result=dict(completed=True,passed=True,existing_proof_files_copied=2,
        physical_starts=0,native_runner_constructions=0,optimizer_updates=0,
        main_policy_changed=False)
    worker.write(args.out/'results.json',result);worker.write(args.out/'ready.json',result)
