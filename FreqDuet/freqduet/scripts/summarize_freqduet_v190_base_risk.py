#!/usr/bin/env python3
"""Preregister a baseline-adjusted diagnostic before reading V190 risk results."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import sys

import audit_freqduet_v190_leg_risk as stage


def run(args):
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).resolve().parents[1]/'tests'),
        '-p','test_v190_*.py','-v'],check=True)
    parent = stage.read(args.v190_root/'protocol.json')
    stage.require(parent['source'] == 'a7d126966a' and parent['calibration_days'] == list(stage.DAYS) and
        parent['support']['minimum_distinct_days'] == 19,'requires original frozen V190 protocol')
    plan = dict(source=os.environ['FREQDUET_DIAGNOSTIC_COMMIT'],parent_protocol=parent,
        supplement='Add95% baseline-adjusted per-signal diagnostics, without conditioning on the other signal; '
            'distinguish shared signal information from unique conditional information. Original primary gate unchanged.',
        preinspection='No parent risk summaries/details opened; seal before reading saved moment cells',
        interval_fit=False,predictor_fit=False,minimum_days=19,new_replays=0,new_training=0)
    stage.require(not (args.out_root/'protocol.json').exists(),'do not overwrite supplement')
    stage.write(args.out_root/'protocol.json',plan)
    print('V190_BASE_DIAGNOSTIC_SEALED_BEFORE_RISK_RESULT_INSPECTION',flush=True)
    model = stage.read(args.v172_root/'model.json')
    stage.require(model['execution_source'] == 'a5b45df520' and model['status'] == 'sealed' and
        model['calibration_days'] == list(stage.DAYS) and len(model['actors']) == 12,'frozen V172 metadata differs')
    with ProcessPoolExecutor(max_workers=8,mp_context=get_context('spawn')) as pool:
        results = []
        for job in as_completed([pool.submit(stage.reduce_scope,args,model,m,s,args.v190_root) for m,s in
                product(('old','handoff'),stage.SCOPES)]):
            results.extend(job.result())
    stage.write(args.out_root/'summary/status.json',dict(status='v190_baseline_and_unique_information_complete',
        source=plan['source'],protocol=plan,paths=1536,
        results=[{k:v for k,v in r.items() if k not in ('actor_margins','day_margins')} for r in results],
        evidence=stage.evidence(results),interval_fit=False,predictor_fit=False,map_updates=0,
        confirmation_outcomes_read=False,confirmation_started=False,new_replays=0,new_training=0,
        main_policy_changed=False,parent_risk_results_opened=False))
    print('V190_BASE_AND_UNIQUE_INFORMATION_COMPLETE_NO_FIT',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--v190-root',type=Path,required=True)
    p.add_argument('--v172-root',type=Path,required=True)
    run(p.parse_args())
