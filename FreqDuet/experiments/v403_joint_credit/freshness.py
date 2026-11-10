"""Reserve only four new development days using bounded producer metadata."""

import argparse
from pathlib import Path

from worker import HERE,read,write
from freshness_scan import audit,hits


def main(args):
    days=read(HERE/'protocol.json')['development_days']
    old=read(args.root.parent.parent/'FreqDuet-v401-budget-conditioned-upper/metadata03/summary/results.json')
    assert old['passed']
    base=args.root.parent.parent/'FreqDuet-v4-worktree/FreqDuet'
    observed=audit([base/'freqduet/results_freqduet',base/'experiments'],days,
                   old['hpc']['resolved_readonly_artifacts'])
    assert not observed['collisions'] and not observed['read_errors'],observed
    extra=[]
    for project in args.root.parent.parent.glob('FreqDuet-*'):
        if not project.is_dir() or project==args.root.parent:continue
        for pattern in ('source*/protocol.json','source*/manifest.json','source*/config.json'):
            for path in project.glob(pattern):
                found=hits(read(path),set(days))
                if found:extra.append(dict(path=str(path),days=found))
    inventory=read(HERE/'queue_inventory.json')
    assert inventory['development_days']==days and not inventory['collisions'] and not extra
    output=dict(completed=True,passed=True,development_days=days,
        hpc_metadata_files=observed['json_files_read'],hpc_collisions=[],
        separate_runtime_manifest_collisions=extra,queue_inventory=inventory,
        previous_gpu_inventory=old['gpu'],physical_starts=0,training_updates=0)
    write(args.root/'freshness/summary/results.json',output)
    write(args.root/'freshness/summary/ready.json',dict(passed=True,days=days))
    print('V403_FRESH_DAYS_RESERVED',days,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
