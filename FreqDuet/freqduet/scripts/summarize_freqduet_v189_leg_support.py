#!/usr/bin/env python3
"""Correct V189 observed-key counts from saved support cells, without path rereads."""

import argparse
from itertools import product
import os
from pathlib import Path
import subprocess
import sys

import preregister_freqduet_v189_leg_innovation as stage


def run(args):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v189_*.py','-v'],check=True)
    old = stage.read(args.v189_root/'summary/status.json')
    plan = stage.read(args.v189_root/'protocol.json')
    stage.require(old['status'] == 'v189_leg_innovation_support_complete' and old['source'] == '92c59ee242' and
        old['paths'] == 1536 and plan == old['protocol'] and plan['calibration_days'] == list(stage.DAYS),
        'requires complete original V189 support extraction')
    actors = [r['id'] for r in old['results'][0]['actor_margins']]
    stage.require(len(actors) == len(set(actors)) == 12, 'actor grid differs')
    rows = []
    for actor,mode in product(actors,('old','handoff')):
        b = stage.read(args.v189_root/'cells'/f'{actor}_{mode}.json')
        stage.require(b['source'] == old['source'] and len(b['rows']) == 64 and
            all((r['actor'],r['mode']) == (actor,mode) for r in b['rows']), 'support bucket differs')
        rows.extend(b['rows'])
    results = [stage.summarize([r for r in rows if r['mode'] == mode],actors,mode,scope)
        for mode,scope in product(('old','handoff'),stage.SCOPES)]
    for before,after in zip(old['results'],results):
        stage.require(all(before[k] == after[k] for k in before if k != 'keys_by_count'),
            'count-only correction changed other support metrics')
    result = dict(old,source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],results=results,
        original_v189_source=old['source'],observed_key_count_corrected=True,
        correction='Exclude zero-support keys from observed counts; label count columns 1/9, not vector indices 1/2.')
    stage.write(args.out_root/'summary/status.json',result)
    print('V189_SAVED_CELL_COUNT_CORRECTION_COMPLETE',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--v189-root',type=Path,required=True)
    run(p.parse_args())
