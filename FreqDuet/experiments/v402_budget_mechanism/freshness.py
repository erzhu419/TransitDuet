"""New simulation-day admission using producer metadata and scheduler inventory."""

import argparse
from pathlib import Path
from worker import HERE,read,write
from freshness_scan import audit,hits


def main(args):
    days=read(HERE/'protocol.json')['confirmation_days']
    old=read(args.root.parent.parent/'FreqDuet-v401-budget-conditioned-upper/metadata03/summary/results.json')
    assert old['passed']
    base=args.root.parent.parent/'FreqDuet-v4-worktree/FreqDuet'
    observed=audit([base/'freqduet/results_freqduet',base/'experiments'],days,
                   old['hpc']['resolved_readonly_artifacts'])
    assert not observed['collisions'] and not observed['read_errors'],observed
    current=args.root.parent
    extra=[]
    for project in args.root.parent.parent.glob('FreqDuet-*'):
        if not project.is_dir() or project==current:continue
        for pattern in ('source*/protocol.json','source*/manifest.json','source*/config.json'):
            for path in project.glob(pattern):
                found=hits(read(path),set(days))
                if found:extra.append(dict(path=str(path),days=found))
    queue=read(HERE/'queue_inventory.json')
    assert queue['confirmation_days']==days and not queue['collisions']
    assert not extra,extra
    result=dict(completed=True,passed=True,confirmation_days=days,
        hpc_metadata_files=observed['json_files_read'],unresolved_errors=0,
        hpc_collisions=[],separate_runtime_manifest_collisions=extra,
        queue_inventory=queue,previous_gpu_inventory=old['gpu'],
        gpu_scope='Reuse settled V401 inventory; subsequent registered FreqDuet cases checked in scheduler inventory.',
        physical_starts=0,training_updates=0,main_policy_changed=False)
    write(args.root/'freshness/summary/results.json',result)
    write(args.root/'freshness/summary/ready.json',dict(completed=True,passed=True,days=days))
    print('V402_FRESH_DAYS_RESERVED',days,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args())
