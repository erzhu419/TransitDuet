"""Orchestrate a small code upload and direct GPU-to-HPC frozen asset transfer."""

import base64
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/home/erzhu419/mine_code/scheduleurm/skill')
import scheduler

REMOTE=Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v400-independent-confirmation')
PYTHON='/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python'


def remote(node,python,code,timeout=120):
    rc,out,err=scheduler.run_on(node,shlex.join([python,'-c',code]),timeout=timeout,check=False)
    if rc:
        raise RuntimeError((node,rc,err[-2500:],out[-1000:]))
    return out


def save(name,value):
    with (HERE/name).open('x') as f:
        json.dump(value,f,indent=2);f.write('\n')


def main():
    environment=json.loads((HERE.parent/'v347_initial_queue_prequential/gpu01/review01/environment.json').read_text())['result']['analysis_spec']['env']
    gpu_stage='/home/erzhu419/scheduleurm_work/FreqDuet-v400-prepare01'
    files={'package_assets.py':(HERE/'package_assets.py').read_bytes(),
           'environment.json':json.dumps(environment).encode()}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for name,data in files.items():
            item=tarfile.TarInfo(name);item.size=len(data)
            archive.addfile(item,io.BytesIO(data))
    encoded=base64.b64encode(buf.getvalue()).decode()
    code='import base64,io,tarfile\nfrom pathlib import Path\n'
    code+=f'p=Path({gpu_stage!r});p.mkdir(exist_ok=False)\n'
    code+=f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r}))).extractall(p)\n'
    remote('jtl110gpu2','/home/erzhu419/.conda/envs/freqduet-paper/bin/python',code)
    code='import subprocess,sys\n'
    code+=f'subprocess.run([sys.executable,{gpu_stage+"/package_assets.py"!r},"--environment",{gpu_stage+"/environment.json"!r}],check=True)'
    transferred=json.loads(remote('jtl110gpu2','/home/erzhu419/.conda/envs/freqduet-paper/bin/python',code,300))
    save('asset_transport.json',transferred)

    code='import json,tarfile\nfrom pathlib import Path\n'
    code+=f'root=Path({str(REMOTE)!r});assets=root/"assets01";assets.mkdir(exist_ok=False)\n'
    code+='tarfile.open(root/"transport/assets.tar.gz","r:gz").extractall(assets)\n'
    code+='env=json.loads((assets/"environment.json").read_text())\n'
    code+='old="/home/erzhu419/scheduleurm_work/FreqDuet-v347-runtime/hpc"\n'
    code+='paths=[str(assets/"basis"),str(assets/"history")]\n'
    code+='paths += [p.replace(old,"/home/zhengliang01/scheduleurm_work") for p in env["PYTHONPATH"].split(":") if p!="/home/erzhu419/scheduleurm_work/FreqDuet-v347-runtime/run01"]\n'
    code+='paths += [str(assets/"support")]\nenv["PYTHONPATH"]=":".join(paths)\n'
    code+=f'value={{"node":"node004","python":{PYTHON!r},"root":str(root/"run01"),"source":str(root/"source02"),"assets":str(assets),"env":env,"missing_search_paths":[p for p in paths if not Path(p).is_dir()]}}\n'
    code+='(root/"runtime_deployment.json").write_text(json.dumps(value))\nprint(json.dumps(value))\n'
    deployment=json.loads(remote('node004',PYTHON,code))
    save('deployment.json',deployment)

    names=['manifest.json','authorization.json','PROTOCOL.md','evaluation.py','checks.py','shard.py','aggregate.py']
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for name in names:
            if name.endswith('.py'):
                compile((HERE/name).read_bytes(),name,'exec')
            archive.add(HERE/name,arcname=name)
    encoded=base64.b64encode(buf.getvalue()).decode()
    code='import base64,io,tarfile\nfrom pathlib import Path\n'
    code+=f'p=Path({deployment["source"]!r});p.mkdir(exist_ok=False)\n'
    code+=f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r}))).extractall(p)\n'
    remote('node004',PYTHON,code)
    print('V400_RUNTIME_STAGED',transferred,flush=True)


if __name__=='__main__':
    main()
