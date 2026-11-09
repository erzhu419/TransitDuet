"""Stage V401 and register its approved, file-gated 90-path workflow."""

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
    plan=json.loads((HERE/'protocol.json').read_text())
    authorization=json.loads((HERE/'authorization.json').read_text())
    assert authorization['approved_physical_start_cap']==90
    assert not (HERE/'submission_pipeline01.json').exists()
    root=Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v401-budget-conditioned-upper')
    source=root/'source02';run=root/'run01';assets=Path(d['assets'])
    files=('conditioned_law.py','conditioned_integration.py','recipe.py','worker.py',
           'entry_check.py','runtime_ready.py','engineering_gate.py','development.py','protocol.json','authorization.json','PROTOCOL.md')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name in files:
            if name.endswith('.py'):compile((HERE/name).read_bytes(),name,'exec')
            archive.add(HERE/name,arcname=name)
    encoded=base64.b64encode(buffer.getvalue()).decode()
    code='import base64,io,tarfile\nfrom pathlib import Path\n'
    code+=f'p=Path({str(source)!r});p.mkdir(exist_ok=False)\n'
    code+=f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r}))).extractall(p)\n'
    rc,out,err=scheduler.run_on('node004',shlex.join([d['python'],'-c',code]),timeout=60,check=False)
    assert rc==0,(out[-500:],err[-2000:])
    env=dict(d['env']);env['PYTHONPATH']=str(source)+':'+env['PYTHONPATH']
    specs=[]
    local=HERE/'review02'

    def add(label,script,args,result,waits):
        specs.append(dict(description='FreqDuet V401 '+label,project='FreqDuet',
            signature='FreqDuet/v401/source02/'+label,priority='high',require_node='node004',
            cpu=1,ram_mb=4096,vram=0,no_auto_requeue=True,skip_launch_staging=True,
            cwd=str(source),env=env,
            cmd=shlex.join([d['python'],'-u',str(source/script),*map(str,args)]),
            result_dir=str(result),local_result_dir=str(local/label),wait_for_files=list(map(str,waits))))

    common=['--root',run,'--assets',assets]
    add('entry01','entry_check.py',['--assets',assets,'--out',root/'entry01/summary'],
        root/'entry01/summary',[])
    add('runtime_assets','runtime_ready.py',[*common,'--out',root/'runtime_assets/summary'],
        root/'runtime_assets/summary',[])
    engineering_wait=[local/'entry01/ready.json',local/'metadata03/ready.json',local/'runtime_assets/ready.json']
    for rho in plan['retentions']:
        name=f'engineering/r{round(rho*100):03d}'
        add(name,'worker.py',['path',*common,'--arm','implicit','--split','engineering',
            '--day',2400001,'--rho',rho],run/name/'summary',engineering_wait)
    add('engineering_gate','engineering_gate.py',[*common,'--reference',d['root']],
        run/'engineering_gate/summary',[local/f'engineering/r{round(r*100):03d}/finished.json' for r in plan['retentions']])
    gate=local/'engineering_gate/admitted.json'
    for arm in plan['arms']:
        checkpoint=run/'checkpoints'/arm/'b0'
        add(f'checkpoints/{arm}/b0','worker.py',['initialize',*common,'--arm',arm,'--out',checkpoint],
            checkpoint/'summary',[gate])
    for batch in range(8):
        waits=[local/f'checkpoints/{arm}/b{batch}/ready.json' for arm in plan['arms']]
        for arm in plan['arms']:
            checkpoint=run/'checkpoints'/arm/f'b{batch}'
            paths=[];finished=[]
            for position,rho in enumerate(plan['rho_by_batch_position']):
                day=plan['train_day_start']+4*batch+position
                name=f'train/{arm}/b{batch}/d{day}'
                paths.append(run/name);finished.append(local/name/'finished.json')
                add(name,'worker.py',['path',*common,'--arm',arm,'--split','train','--batch',batch,
                    '--day',day,'--rho',rho,'--checkpoint',checkpoint],run/name/'summary',waits)
            destination=run/'checkpoints'/arm/f'b{batch+1}'
            add(f'checkpoints/{arm}/b{batch+1}','worker.py',['update',*common,'--arm',arm,
                '--batch',batch,'--checkpoint',checkpoint,'--out',destination,'--paths',*paths],
                destination/'summary',finished)
    terminal=[local/f'checkpoints/{arm}/b8/ready.json' for arm in plan['arms']]
    for i in range(8):
        add(f'development_shard_{i}','development.py',['shard',*common,'--shard',i],
            run/f'development_shard_{i}/summary',terminal)
    add('aggregate','development.py',['aggregate',*common],run/'summary',
        [local/f'development_shard_{i}/finished.json' for i in range(8)])
    assert len(specs)==96
    existing={t.get('signature') for t in scheduler.load_state()['tasks']}
    assert not any(s['signature'] in existing for s in specs)
    proc=subprocess.run([sys.executable,str(Path(scheduler.__file__)),'submit-jsonl',
        '--stdin','--trusted','--json','--lock-timeout','120'],input=json.dumps(specs),
        text=True,capture_output=True,check=True)
    tasks=json.loads(proc.stdout)['submitted']
    with (HERE/'submission_pipeline01.json').open('x') as stream:
        json.dump(dict(tasks=[dict(id=t['id'],signature=t['signature']) for t in tasks],
            node='node004',priority='high',physical_start_cap=90,max_concurrent_paths=8,
            source=str(source),root=str(run),assets=str(assets),automatic_retry=False),stream,indent=2)
    command=[sys.executable,str(Path(scheduler.__file__)),'dispatch','--lock-timeout','120']
    for task in tasks:command+=['--task-id',task['id']]
    print('V401_REGISTERED',len(tasks),'entry',tasks[0]['id'],flush=True)
    dispatched=subprocess.run(command,text=True,capture_output=True)
    print('V401_DISPATCH',dispatched.returncode,
          [line for line in dispatched.stdout.splitlines() if 'LAUNCH on ' in line],flush=True)
    if dispatched.returncode:print(dispatched.stderr[-2000:],flush=True)
    raise SystemExit(dispatched.returncode)


if __name__=='__main__':main()
