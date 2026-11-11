"""Stage immutable sources and manually dispatch the bounded CPU study."""

import argparse
import base64
import importlib.util
import io
from pathlib import Path
import shlex
import tarfile

HERE = Path(__file__).resolve().parent
module = importlib.util.spec_from_file_location(
    'v404_launcher', HERE.parent / 'v404_publication_closure/launch.py')
shared = importlib.util.module_from_spec(module)
module.loader.exec_module(shared)
shared.HERE = HERE
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v407-service-continuation/run03')
shared.SOURCE = shared.ROOT.parent / 'source03'


def stage():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in ('frozen_v406.py', 'service_kernel.py', 'control.py', 'tests.py',
                     'run.py', 'readout.py', 'protocol.json'):
            path = HERE / name
            if path.suffix == '.py':
                compile(path.read_bytes(), name, 'exec')
            archive.add(path, arcname=name)
    payload = base64.b64encode(buf.getvalue()).decode()
    code = ('import base64,io,tarfile\nfrom pathlib import Path\n'
        f'p=Path({str(shared.SOURCE)!r});p.mkdir(parents=True,exist_ok=False)\n'
        f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))).extractall(p)\n')
    rc,out,err = shared.scheduler.run_on('node003',shlex.join([
        shared.deployment()['python'],'-c',code]),timeout=90,check=False)
    assert rc == 0,(out,err[-1000:])


def spec(label, command, *args, result=None):
    job = shared.spec(label,'run.py',[],result=result)
    job.update(description='FreqDuet V407 '+label,signature='FreqDuet/v407/'+label)
    job['env'].pop('PYTHONPATH',None)
    job['cmd'] = shlex.join([shared.deployment()['python'],'-B','-I','-u',
        str(shared.SOURCE/'run.py'),command,'--root',str(shared.ROOT),*map(str,args)])
    return job


def submit_traffic():
    assert shared.read(HERE/'review/checks03/results.json')['passed']
    plan = shared.read(HERE/'protocol.json')
    jobs = [spec(f'{r}_{d}_{arm}','case','--regime',r,'--day',d,'--arm',arm,
        result=shared.ROOT/'cases'/r/str(d)/arm/'summary')
        for r,days in plan['days'].items() for d in days for arm in plan['arms']]
    assert len(jobs) == plan['physical_start_cap'] <= plan['max_concurrent_paths']
    return shared.submit('traffic',jobs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=('checks','traffic','readout'))
    a = p.parse_args()
    if a.phase == 'checks':
        if not (HERE/'submission_checks03.json').exists():
            stage()
        shared.wait(shared.submit('checks03',[spec('checks03','checks',
            result=shared.ROOT/'checks/summary')]))
    elif a.phase == 'traffic':
        shared.wait(submit_traffic())
    else:
        job = spec('readout','checks',result=shared.ROOT/'readout/summary')
        job['cmd'] = shlex.join([shared.deployment()['python'],'-B','-I','-u',
            str(shared.SOURCE/'readout.py'),'--root',str(shared.ROOT)])
        shared.wait(shared.submit('readout',[job]))


if __name__ == '__main__':
    main()
