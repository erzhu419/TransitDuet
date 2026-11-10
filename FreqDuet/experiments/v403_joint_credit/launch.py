"""Stage V403; execution remains scheduler high on Linux CPU nodes."""

import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/erzhu419/mine_code/scheduleurm/skill')
import scheduler

ROOT=Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v403-joint-credit/run01')


def inventory(plan):
    days=plan['development_days'];collisions=[];count=0;recovered=0
    def check(task,origin):
        nonlocal count
        if 'freqduet' not in str(task.get('project','')).lower() and 'freqduet' not in str(task.get('signature','')).lower():return
        count+=1
        text=' '.join(str(task.get(k,'')) for k in ('cmd','signature','description'))
        found=[d for d in days if str(d) in text]
        if found:collisions.append(dict(task=task['id'],origin=origin,days=found))
    for t in scheduler.load_state()['tasks']:check(t,'queue')
    archive=Path.home()/'.claude/scheduler/queue_archive.jsonl'
    if archive.exists():
        with archive.open('rb') as stream:
            for line in stream:
                if b'FreqDuet' in line or b'freqduet' in line:
                    if line.startswith(b'\x00'):
                        line=line.lstrip(b'\x00');recovered+=1
                    check(json.loads(line.decode('utf-8')),'archive')
    result=dict(development_days=days,registered_tasks_checked=count,collisions=collisions,
                recovered_null_prefixed_archive_records=recovered)
    assert not collisions,collisions
    with (HERE/'queue_inventory.json').open('x') as stream:json.dump(result,stream,indent=2)


def deployment():return json.loads((HERE.parent/'v400_independent_confirmation/deployment.json').read_text())


def make_spec(label,script,args,result,waits=()):
    d=deployment();source=ROOT.parent/'source03'
    old=ROOT.parent.parent/'FreqDuet-v402-budget-mechanism/source01'
    env=dict(d['env']);env['PYTHONPATH']=str(source)+':'+str(old)+':'+env['PYTHONPATH']
    spec=dict(description='FreqDuet V403 '+label,project='FreqDuet',signature='FreqDuet/v403/'+label,
        priority='high',allowed_nodes=['node001','node002','node003','node004','node005','node006'],
        cpu=1,ram_mb=4096,vram=0,allow_cpu_training=True,
        cpu_training_justification='CPU-only transit simulation and JSON analysis in the frozen dedicated Linux environment; no learner updates.',
        no_auto_requeue=True,skip_launch_staging=True,cwd=str(source),env=env,
        cmd=shlex.join([d['python'],'-u',str(source/script),*map(str,args)]),
        result_dir=str(result),local_result_dir=str(HERE/'review'/label),wait_for_files=list(map(str,waits)))
    return spec


def submit_many(requests):
    specs=[make_spec(*request) for request in requests]
    proc=subprocess.run([sys.executable,scheduler.__file__,'submit-jsonl','--stdin','--trusted','--json',
                         '--lock-timeout','600'],input=json.dumps(specs),text=True,capture_output=True)
    if proc.returncode:raise RuntimeError(proc.stdout[-2000:]+'\n'+proc.stderr[-4000:])
    return json.loads(proc.stdout)['submitted']


def dispatch(tasks):
    command=[sys.executable,scheduler.__file__,'dispatch','--lock-timeout','120']
    for t in tasks:command+=['--task-id',t['id']]
    result=subprocess.run(command,text=True,capture_output=True)
    print(result.stdout[-6500:],result.stderr[-1500:],flush=True)
    assert result.returncode==0


def preflight():
    assert not (HERE/'submission_preflight.json').exists()
    plan=json.loads((HERE/'protocol.json').read_text());inventory(plan)
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for path in HERE.iterdir():
            if path.suffix not in ('.py','.json','.md') or path.name=='launch.py':continue
            if path.suffix=='.py':compile(path.read_bytes(),path.name,'exec')
            archive.add(path,arcname=path.name)
    code='import base64,io,tarfile\nfrom pathlib import Path\n'
    code+=f'p=Path({str(ROOT.parent/"source03")!r});p.mkdir(parents=True,exist_ok=False)\n'
    code+=f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({base64.b64encode(buffer.getvalue()).decode()!r}))).extractall(p)\n'
    d=deployment();rc,out,err=scheduler.run_on('node003',shlex.join([d['python'],'-c',code]),timeout=60,check=False)
    assert rc==0,(out,err[-2000:])
    tasks=submit_many([('checks','checks.py',['--root',ROOT,'--assets',d['assets']],ROOT/'checks/summary'),
           ('freshness','freshness.py',['--root',ROOT],ROOT/'freshness/summary')])
    with (HERE/'submission_preflight.json').open('x') as stream:
        json.dump(dict(tasks=[dict(id=t['id'],signature=t['signature']) for t in tasks],physical_starts=0,
                       source=str(ROOT.parent/'source03'),root=str(ROOT)),stream,indent=2)
    print('V403_PREFLIGHT',[(t['id'],t['signature']) for t in tasks],flush=True)
    dispatch(tasks)


if __name__=='__main__':preflight()
