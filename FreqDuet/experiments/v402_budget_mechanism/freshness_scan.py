"""Inspect existing experiment metadata; never load trajectories or checkpoints."""

import argparse
import json
import os
from pathlib import Path

DAY_KEYS={'day','days','scenario_seed','scenario_seeds','demand_seed','demand_days',
          'evaluation_days','confirmation_days','reserved_days','train_days',
          'train_batches','scenario_day','development_days','development_days_proposed'}
SKIP={'raw','checkpoints','logs','tb_logs','site-packages','node_modules','.git',
      '__pycache__','tape','replay_buffer','native_train'}
NAMES={'results.json','started.json','config.json','manifest.json','protocol.json',
       'completion.json','CURRENT.json','PIPELINE.json'}
PAYLOADS={
    'protocol_v6_v263_evaluation_bank_collection': 'records',
    'protocol_v6_v259_selection_bank_collection': 'records',
    'protocol_v6_v75_resource_value': 'cells',
}


def hits(value,days):
    found=[]
    if isinstance(value,dict):
        for key,child in value.items():
            if key in DAY_KEYS:
                pending=[child]
                while pending:
                    item=pending.pop()
                    if isinstance(item,list): pending.extend(item)
                    elif isinstance(item,int) and item in days: found.append(item)
            if key=='by_day' and isinstance(child,dict):
                found.extend(int(k) for k in child if str(k).isdigit() and int(k) in days)
            found.extend(hits(child,days))
    elif isinstance(value,list):
        for item in value:
            if isinstance(item,(dict,list)):found.extend(hits(item,days))
    return found


def audit(roots,days,resolved=()):
    resolved=set(resolved)
    result=dict(json_files_read=0,collisions=[],read_errors=[],resolved_readonly_artifacts=[],
                excluded_payload_directories=[])
    for root in map(Path,roots):
        assert root.is_dir(),str(root)
        for directory,folders,files in os.walk(root):
            # S/E records are compressed payloads; V75 cells are redundant native
            # runs of its fixed 1750001--1751012 input grid. Their parent metadata
            # remains in the scan, without descending into native runtime trees.
            payload=PAYLOADS.get(Path(directory).name)
            if payload in folders:
                result['excluded_payload_directories'].append(str(Path(directory)/payload))
            folders[:]=[n for n in folders if n not in SKIP and not n.startswith(
                ('source','runtime_source','v401_','protocol_v6_v401'))
                and n!=payload]
            for name in files:
                if name not in NAMES and not (name.startswith('p') and name[1:-5].isdigit() and name.endswith('.json')):
                    continue
                path=Path(directory)/name
                if str(path) in resolved:
                    result['resolved_readonly_artifacts'].append(str(path));continue
                try:value=json.loads(path.read_text())
                except (OSError,UnicodeError,json.JSONDecodeError) as error:
                    result['read_errors'].append(dict(path=str(path),error=type(error).__name__));continue
                result['json_files_read']+=1
                found=hits(value,set(days))
                if found:result['collisions'].append(dict(path=str(path),days=sorted(set(found))))
                if result['json_files_read']%500==0:
                    print('V401_METADATA_FILES',result['json_files_read'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--resolved',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.protocol.read_text());days=plan['development_days_proposed']
    resolution=json.loads(args.resolved.read_text())
    assert resolution['passed'] and not set(days)&set(resolution['source_days'])
    roots=[Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v4-worktree/FreqDuet')/p
           for p in ('freqduet/results_freqduet','experiments')]
    hpc=audit(roots,days,[e['path'] for e in resolution['resolved_read_errors']])
    gpu=json.loads(args.gpu.read_text());queue=json.loads(args.queue.read_text())
    assert gpu['days']==queue['days']==days
    passed=all(v['json_files_read']>0 and not v['collisions'] and not v['read_errors'] for v in (hpc,gpu['metadata'])) and not queue['collisions']
    result=dict(completed=True,passed=passed,days=days,hpc=hpc,gpu=gpu,queue=queue,
                physical_starts=0,training_updates=0,main_policy_changed=False)
    args.out.mkdir(parents=True,exist_ok=True)
    for name in ('results.json','ready.json') if passed else ('results.json',):
        with (args.out/name).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('V401_FRESHNESS',passed,flush=True)
