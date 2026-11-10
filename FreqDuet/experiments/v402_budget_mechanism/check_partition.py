"""Verify exact case coverage for the approved partition, without traffic."""

import argparse
import itertools
from pathlib import Path

from confirmation32 import partition
from worker import HERE,read,write


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    approval=read(Path(__file__).parent/'concurrency_amendment.json')
    assert approval['confirmation_parallel_paths']==32 and approval['physical_start_cap']==150
    assert HERE==a.root.parent/'source01'
    checks={}
    for n in (3,4,5):
        days=tuple(range(16));cases=tuple(range(n))
        cells=[cell for rank in range(32) for cell in partition(days,cases,rank)]
        assert len(cells)==len(set(cells))==16*n
        assert set(cells)==set(itertools.product(days,cases))
        checks[str(16*n)]=dict(unique_cases=len(cells),duplicate_cases=0,missing_cases=0)
    result=dict(completed=True,passed=True,checks=checks,physics_worker=str(HERE/'worker.py'),
        physical_starts=0,training_updates=0,main_policy_changed=False)
    write(a.root/'partition_check/summary/results.json',result)
    write(a.root/'partition_check/summary/ready.json',result)
    print('V402_PARTITION32_PASSED',checks,flush=True)
