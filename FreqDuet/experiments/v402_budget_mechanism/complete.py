"""Orchestrate the approved stages; never run simulation or statistics locally."""

import json
from pathlib import Path
import time

from launch import HERE,ROOT,deployment,submit_many,dispatch,scheduler


def wait(tasks,label):
    pending={t['id'] for t in tasks};last=None
    while pending:
        state={t['id']:t for t in scheduler.load_state()['tasks']}
        failed=[i for i in pending if state[i]['status'] in ('failed','cancelled')]
        assert not failed,(label,failed)
        for task_id in list(pending):
            task=state[task_id]
            if task['status']=='done':
                ok,error=scheduler._sync_one_result(task)
                assert ok,(task_id,error)
                pending.remove(task_id)
        progress=len(tasks)-len(pending)
        if progress!=last:print('V402_STAGE_PROGRESS',label,progress,'/',len(tasks),flush=True);last=progress
        if pending:time.sleep(90)


def register(stage,specs):
    path=HERE/f'submission_{stage}.json'
    if stage=='confirmation' and (HERE/'submission_confirmation32.json').exists():
        path=HERE/'submission_confirmation32.json'
    if path.exists():
        wait(json.loads(path.read_text())['tasks'],stage);return
    existing={t.get('signature'):t for t in scheduler.load_state()['tasks']}
    missing=[spec for spec in specs if 'FreqDuet/v402/'+spec[0] not in existing]
    if missing:
        existing.update({t['signature']:t for t in submit_many(missing)})
    tasks=[existing['FreqDuet/v402/'+spec[0]] for spec in specs]
    with path.open('x') as stream:json.dump(dict(stage=stage,tasks=[dict(id=t['id'],signature=t['signature'])
                                                           for t in tasks]),stream,indent=2)
    print('V402_STAGE_REGISTERED',stage,[t['id'] for t in tasks],flush=True)
    dispatch(tasks);wait(tasks,stage)


def require(label):
    result=json.loads((HERE/'review'/label/'results.json').read_text())
    assert result['completed'] and result['passed'],(label,result)


def main():
    pre=json.loads((HERE/'submission_preflight.json').read_text())['tasks']
    wait(pre,'preflight');require('checks');require('freshness')
    assets=deployment()['assets'];common=['--root',ROOT,'--assets',assets]
    specs=[]
    for arm in ('pressure','native_plan','cap_45'):
        waits=[HERE/'review/checks/ready.json',HERE/'review/freshness/ready.json']
        if arm=='cap_45':waits.append(HERE/'review/engineering_native_plan/finished.json')
        specs.append((f'engineering_{arm}','worker.py',[*common,'--phase','engineering','--arm',arm,'--day',2410001],
            ROOT/'engineering'/arm/'d2410001/summary',waits))
    register('engineering',specs)
    register('engineering_gate',[('engineering_gate','analysis.py',['engineering',*common],
                                  ROOT/'engineering_gate/summary',[])])
    require('engineering_gate')
    specs=[(f'calibration_shard_{i}','shard.py',[*common,'--phase','calibration','--shard',i],
            ROOT/f'calibration_shard_{i}/summary',[]) for i in range(8)]
    register('calibration',specs)
    for i in range(8):
        assert json.loads((HERE/f'review/calibration_shard_{i}/finished.json').read_text())['completed']
    register('selection',[('selection','analysis.py',['selection',*common],ROOT/'selection/summary',[])])
    result=json.loads((HERE/'review/selection/results.json').read_text())
    if not result['passed']:
        print('V402_CLOSED_CALIBRATION_OUTSIDE_SUPPORT',flush=True);return
    specs=[(f'confirmation_shard_{i}','shard.py',[*common,'--phase','confirmation','--shard',i],
            ROOT/f'confirmation_shard_{i}/summary',[]) for i in range(8)]
    register('confirmation',specs)
    for i in range(8):
        assert json.loads((HERE/f'review/confirmation_shard_{i}/finished.json').read_text())['completed']
    register('aggregate',[('aggregate','analysis.py',['aggregate',*common],ROOT/'aggregate/summary',[])])
    print('V402_ALL_STAGES_COMPLETE',flush=True)


if __name__=='__main__':main()
