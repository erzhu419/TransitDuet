"""Approved 32-way partition; child cases still use the frozen source01 worker."""

import argparse
import itertools
from pathlib import Path
import subprocess
import sys

from worker import HERE,read,write,selected_cases


def partition(days,cases,rank):
    return list(itertools.product(days,cases))[rank::32]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--phase',choices=('confirmation',),required=True)
    p.add_argument('--shard',type=int,choices=range(32),required=True);a=p.parse_args()
    assert read(a.root/'partition_check/summary/ready.json')['passed']
    plan=read(HERE/'protocol.json');cases=selected_cases(a.root,plan)
    cells=partition(plan['confirmation_days'],cases,a.shard);completed=[];rc=0
    try:
        for day,case in cells:
            command=[sys.executable,'-u',str(HERE/'worker.py'),'--root',str(a.root),'--assets',str(a.assets),
                     '--phase','confirmation','--arm',case['arm'],'--day',str(day)]
            rc=subprocess.run(command).returncode
            if rc:break
            completed.append(dict(day=day,arm=case['arm']))
    except Exception:
        rc=1;raise
    finally:write(a.root/f'confirmation_shard_{a.shard}/summary/finished.json',dict(
        completed=rc==0 and len(completed)==len(cells),cases=completed,returncode=rc,automatic_retry=False))
    raise SystemExit(rc)
