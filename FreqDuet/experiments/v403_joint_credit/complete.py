"""Local orchestration only; no local simulation or statistics."""

import json
import time

from launch import HERE,ROOT,deployment,submit_many,dispatch,scheduler,preflight


def stage(name,specs):
    receipt=HERE/f'submission_{name}.json'
    if receipt.exists():
        tasks=json.loads(receipt.read_text())['tasks']
    else:
        tasks=submit_many(specs)
        tasks=[dict(id=t['id'],signature=t['signature']) for t in tasks]
        with receipt.open('x') as stream:json.dump(dict(tasks=tasks),stream,indent=2)
        print('V403_REGISTERED',name,[t['id'] for t in tasks],flush=True)
        dispatch(tasks)
    pending={t['id'] for t in tasks};previous=None
    while pending:
        state={t['id']:t for t in scheduler.load_state()['tasks']}
        for task_id in list(pending):
            task=state[task_id]
            assert task['status'] not in ('failed','cancelled'),(task_id,task.get('last_error'))
            if task['status']=='done':
                ok,error=scheduler._sync_one_result(task);assert ok,(task_id,error)
                pending.remove(task_id)
        progress=len(tasks)-len(pending)
        if progress!=previous:print('V403_PROGRESS',name,progress,'/',len(tasks),flush=True);previous=progress
        if pending:time.sleep(60)


def passed(name):
    r=json.loads((HERE/'review'/name/'results.json').read_text())
    assert r['completed'] and r['passed'],(name,r)


def main():
    if not (HERE/'submission_preflight.json').exists():preflight()
    name='preflight_repaired' if (HERE/'submission_preflight_repaired.json').exists() else 'preflight'
    stage(name,[]);passed('checks');passed('freshness')
    plan=json.loads((HERE/'protocol.json').read_text())
    common=['--root',ROOT,'--assets',deployment()['assets']]
    stage('equivalent_paths',[(f'engineering_{arm}','worker.py',[*common,'--phase','engineering',
        '--arm',arm,'--day',plan['engineering_day']],ROOT/'engineering'/arm/f'd{plan["engineering_day"]}/summary',[])
        for arm in ('pressure','candidate_noop')])
    stage('equivalence',[('equivalence','analysis.py',['equivalence',*common],ROOT/'equivalence/summary',[])])
    passed('equivalence')
    stage('candidate_path',[('engineering_candidate','worker.py',[*common,'--phase','engineering',
        '--arm','candidate','--day',plan['engineering_day']],ROOT/'engineering/candidate'/f'd{plan["engineering_day"]}/summary',[])])
    stage('engineering_gate',[('engineering_gate','analysis.py',['engineering',*common],ROOT/'engineering_gate/summary',[])])
    passed('engineering_gate')
    stage('development',[(f'development_{arm}_d{day}','worker.py',[*common,'--phase','development',
        '--arm',arm,'--day',day],ROOT/'development'/arm/f'd{day}/summary',[])
        for day in plan['development_days'] for arm in plan['arms']])
    stage('aggregate',[('aggregate','analysis.py',['aggregate',*common],ROOT/'aggregate/summary',[])])
    print('V403_ALL_STAGES_COMPLETE',flush=True)


if __name__=='__main__':main()
