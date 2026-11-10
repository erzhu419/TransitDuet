"""Eight bounded sequential shards; a failed start is never retried."""

import argparse
import itertools
from pathlib import Path
import subprocess
import sys

from worker import HERE,read,write,grid,selected_cases


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--phase',choices=('calibration','confirmation'),required=True)
    p.add_argument('--shard',type=int,choices=range(8),required=True);a=p.parse_args()
    plan=read(HERE/'protocol.json')
    cases=grid(plan) if a.phase=='calibration' else selected_cases(a.root,plan)
    days=plan['calibration_days'] if a.phase=='calibration' else plan['confirmation_days']
    cells=list(itertools.product(days,cases))[a.shard::8];completed=[];rc=0
    try:
        for day,case in cells:
            command=[sys.executable,'-u',str(HERE/'worker.py'),'--root',str(a.root),'--assets',str(a.assets),
                     '--phase',a.phase,'--arm',case['arm'],'--day',str(day)]
            rc=subprocess.run(command).returncode
            if rc:break
            completed.append(dict(day=day,arm=case['arm']))
    except Exception:
        rc=1;raise
    finally:write(a.root/f'{a.phase}_shard_{a.shard}/summary/finished.json',dict(
        completed=rc==0 and len(completed)==len(cells),cases=completed,returncode=rc,automatic_retry=False))
    raise SystemExit(rc)
