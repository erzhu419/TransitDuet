"""Register the two explicitly approved engineering replacements, not a new matrix."""

import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE=Path(__file__).parent.resolve()
sys.path.insert(0,'/home/erzhu419/mine_code/scheduleurm/skill')
import scheduler


def main():
    d=json.loads((HERE.parent/'v400_independent_confirmation/deployment.json').read_text())
    original=json.loads((HERE/'submission_pipeline01.json').read_text())
    amendment=json.loads((HERE/'engineering_restart_authorization.json').read_text())
    assert amendment['effective_physical_start_cap']==92
    receipt=HERE/'submission_engineering_restart01.json'
    assert not receipt.exists()
    tasks={t['id']:t for t in scheduler.load_state()['tasks']}
    for item in original['tasks']:
        task=tasks[item['id']]
        if '/engineering/' in item['signature'] or item['signature'].endswith('/engineering_gate'):
            assert task['status']=='failed'
        elif not item['signature'].endswith('/entry01'):
            assert task['status']=='queued', (task['id'],task['status'])
    root=Path(original['root']).parent;run=root/'run01';source=root/'source03';assets=Path(d['assets'])
    old_source=root/'source02'
    files=('conditioned_law.py','conditioned_integration.py','recipe.py','worker.py','engineering_gate.py',
           'development.py','restart_check.py','protocol.json','authorization.json',
           'engineering_restart_authorization.json','PROTOCOL.md')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name in files:
            if name.endswith('.py'):compile((HERE/name).read_bytes(),name,'exec')
            archive.add(HERE/name,arcname=name)
    encoded=base64.b64encode(buffer.getvalue()).decode()
    previous=subprocess.check_output(['git','show','HEAD:FreqDuet/experiments/v401_budget_conditioned_upper/development.py'],
                                     cwd=HERE,text=True)
    code='import base64,io,tarfile,shutil\nfrom pathlib import Path\n'
    code+=f'p=Path({str(source)!r});old=Path({str(old_source)!r})\n'
    code+=f'assert (old/"development.py").read_text()=={previous!r}\n'
    code+='p.mkdir(exist_ok=False)\n'
    code+=f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r}))).extractall(p)\n'
    code+='for name in ("conditioned_law.py","conditioned_integration.py","recipe.py","protocol.json","authorization.json"):\n'
    code+=' assert (p/name).read_bytes()==(old/name).read_bytes(),name\n'
    code+='shutil.copyfile(old/"development.py",old/"development_before_restart01.py")\n'
    code+='for name in ("development.py","engineering_restart_authorization.json"):\n shutil.copyfile(p/name,old/name)\n'
    rc,out,err=scheduler.run_on('node004',shlex.join([d['python'],'-c',code]),timeout=60,check=False)
    assert rc==0,(out[-500:],err[-2000:])
    env=dict(d['env']);env['PYTHONPATH']=str(source)+':'+env['PYTHONPATH']
    local=HERE/'review02';specs=[]

    def add(label,script,args,result,waits,local_result):
        specs.append(dict(description='FreqDuet V401 approved '+label,project='FreqDuet',
            signature='FreqDuet/v401/source03/'+label,priority='high',require_node='node004',
            cpu=1,ram_mb=4096,vram=0,no_auto_requeue=True,skip_launch_staging=True,
            cwd=str(source),env=env,cmd=shlex.join([d['python'],'-u',str(source/script),*map(str,args)]),
            result_dir=str(result),local_result_dir=str(local_result),wait_for_files=list(map(str,waits))))

    common=['--root',run,'--assets',assets]
    add('restart_check','restart_check.py',[*common,'--out',root/'restart_check01/summary'],
        root/'restart_check01/summary',[],local/'restart_check01')
    for rho in (.25,.5):
        name=f'engineering_retry01/r{round(rho*100):03d}'
        add(name,'worker.py',['path',*common,'--arm','implicit','--split','engineering',
            '--day',2400001,'--rho',rho,'--engineering-attempt',2],run/name/'summary',
            [local/'restart_check01/ready.json'],local/name)
    add('engineering_gate_retry01','engineering_gate.py',[*common,'--reference',d['root'],
        '--engineering-dir','engineering_retry01'],run/'engineering_gate/summary',
        [local/f'engineering_retry01/r{round(rho*100):03d}/finished.json' for rho in (.25,.5)],
        local/'engineering_gate')
    proc=subprocess.run([sys.executable,str(Path(scheduler.__file__)),'submit-jsonl',
        '--stdin','--trusted','--json','--lock-timeout','120'],input=json.dumps(specs),
        text=True,capture_output=True,check=True)
    submitted=json.loads(proc.stdout)['submitted']
    with receipt.open('x') as stream:
        json.dump(dict(tasks=[dict(id=t['id'],signature=t['signature']) for t in submitted],
            source=str(source),physical_start_cap=92,additional_physical_starts=2,
            original_failures=['t213966','t213967'],priority='high',node='node004',
            automatic_retry=False),stream,indent=2)
    command=[sys.executable,str(Path(scheduler.__file__)),'dispatch','--lock-timeout','120']
    for task in submitted:command+=['--task-id',task['id']]
    print('V401_ENGINEERING_RESTART_REGISTERED',[(t['id'],t['signature']) for t in submitted],flush=True)
    dispatched=subprocess.run(command,text=True,capture_output=True)
    print(dispatched.stdout[-5000:],flush=True)
    if dispatched.returncode:print(dispatched.stderr[-2000:],flush=True)
    raise SystemExit(dispatched.returncode)


if __name__=='__main__':main()
