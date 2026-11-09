"""Register the approved gated workflow; only high/manual scheduler jobs execute it."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
SCHEDULER=Path('/home/erzhu419/mine_code/scheduleurm/skill')
sys.path.insert(0,str(SCHEDULER))
import scheduler


def main():
    deployment=json.loads((HERE/'deployment.json').read_text())
    authorization=json.loads((HERE/'authorization.json').read_text())
    assert authorization['approved_physical_start_cap']==82 and authorization['max_concurrent_paths']==8
    root,source,assets=map(Path,(deployment['root'],deployment['source'],deployment['assets']))
    assert not (HERE/'matrix_submission.json').exists()
    specs=[]

    def add(label,entry,args,waits):
        specs.append(dict(description='FreqDuet V400 '+label,project='FreqDuet',
            signature='FreqDuet/v400/source02/'+label,priority='high',require_node='node004',
            cpu=1,ram_mb=4096,vram=0,no_auto_requeue=True,skip_launch_staging=True,
            cwd=str(source),env=deployment['env'],
            cmd=shlex.join([deployment['python'],'-u',str(source/entry),*map(str,args)]),
            result_dir=str(root/label/'summary'),local_result_dir=str(HERE/'review01'/label),
            wait_for_files=list(map(str,waits))))

    common=['--root',root,'--assets',assets]
    add('runtime_check','checks.py',['runtime',*common],[])
    ready=HERE/'review01/runtime_check/ready.json'
    for arm in ('K4_r050','manual_r050'):
        label=f'engineering/{arm}_d2077002'
        add(label,'evaluation.py',[*common,'--engineering','--arm',arm,'--day',2077002],[ready])
    add('engineering_gate','checks.py',['engineering',*common],
        [HERE/'review01/results.json',
         *[HERE/f'review01/engineering/{a}_d2077002/finished.json' for a in ('K4_r050','manual_r050')]])
    admitted=HERE/'review01/engineering_gate/admitted.json'
    for i in range(8):
        add(f'shard_{i}','shard.py',[*common,'--shard',i],[admitted])
    add('aggregate','aggregate.py',common,[HERE/f'review01/shard_{i}/finished.json' for i in range(8)])
    queued={r.get('signature') for r in json.loads(scheduler.QUEUE_FILE.read_text())['tasks']}
    assert not any(s['signature'] in queued for s in specs)
    process=subprocess.run([sys.executable,str(SCHEDULER/'scheduler.py'),'submit-jsonl',
        '--stdin','--trusted','--json','--lock-timeout','120'],input=json.dumps(specs),
        text=True,capture_output=True,check=True)
    submitted=json.loads(process.stdout)['submitted']
    assert len(submitted)==13
    tasks=[{'id':r['id'],'signature':r['signature']} for r in submitted]
    with (HERE/'matrix_submission.json').open('x') as f:
        json.dump({'tasks':tasks,'priority':'high','node':'node004','physical_start_cap':82,
                   'confirmation_release':'two old-day reproductions must pass','automatic_retry':False},f,indent=2)
    command=[sys.executable,str(SCHEDULER/'scheduler.py'),'dispatch','--lock-timeout','120']
    for task in tasks:
        command+=['--task-id',task['id']]
    print('V400_REGISTERED',tasks,flush=True)
    subprocess.run(command,check=True)


if __name__=='__main__':
    main()
