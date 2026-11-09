"""Stage only the new copy and manually dispatch its zero-traffic contracts."""

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


def main():
    deployment=json.loads((HERE.parent/'v400_independent_confirmation/deployment.json').read_text())
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert protocol['authorized_physical_starts']==protocol['authorized_candidate_training_updates']==0
    root=Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v401-budget-conditioned-upper')
    source=root/'source01'
    signature='FreqDuet/v401/source01/zero-traffic-contracts'
    assert not any(t.get('signature')==signature for t in scheduler.load_state()['tasks'])
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name in ('conditioned_law.py','conditioned_integration.py','recipe.py',
                     'test_conditioned.py','protocol.json','PROTOCOL.md'):
            if name.endswith('.py'):
                compile((HERE/name).read_bytes(),name,'exec')
            archive.add(HERE/name,arcname=name)
    encoded=base64.b64encode(buffer.getvalue()).decode()
    code='import base64,io,tarfile\nfrom pathlib import Path\n'
    code+=f'p=Path({str(source)!r});p.mkdir(parents=True,exist_ok=False)\n'
    code+=f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r}))).extractall(p)\n'
    rc,out,err=scheduler.run_on('node004',shlex.join([deployment['python'],'-c',code]),
                               timeout=60,check=False)
    assert rc==0,(out[-500:],err[-2000:])
    env=dict(deployment['env'])
    env['PYTHONPATH']=str(source)+':'+env['PYTHONPATH']
    spec=dict(description='FreqDuet V401 budget-conditioned upper synthetic contracts; zero traffic/updates',
        project='FreqDuet',signature=signature,priority='high',require_node='node004',
        cpu=1,ram_mb=4096,vram=0,no_auto_requeue=True,skip_launch_staging=True,
        cwd=str(source),env=env,
        cmd=shlex.join([deployment['python'],'-u',str(source/'test_conditioned.py'),
            '--base-config',str(Path(deployment['assets'])/'K4/config.json'),
            '--out',str(root/'checks01/summary')]),
        result_dir=str(root/'checks01/summary'),local_result_dir=str(HERE/'review01/checks01'))
    queued=subprocess.run([sys.executable,str(Path(scheduler.__file__)),'submit-jsonl',
        '--stdin','--trusted','--json','--lock-timeout','120'],
        input=json.dumps([spec]),text=True,capture_output=True,check=True)
    task=json.loads(queued.stdout)['submitted'][0]['id']
    with (HERE/'submission_checks01.json').open('x') as stream:
        json.dump(dict(task=task,node='node004',priority='high',source=str(source),
            physical_starts=0,candidate_training_updates=0,automatic_retry=False),stream,indent=2)
    print('V401_CHECK_TASK',task,flush=True)
    subprocess.run([sys.executable,str(Path(scheduler.__file__)),'dispatch',
                    '--task-id',task,'--lock-timeout','120'],check=True)


if __name__=='__main__':
    main()
