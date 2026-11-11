"""Stage and manually dispatch the bounded study on the established CPU pool."""

import argparse
import base64
import io
import importlib.util
from pathlib import Path
import shlex
import sys
import tarfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'v404_publication_closure'))
module = importlib.util.spec_from_file_location(
    'v404_launcher', HERE.parent / 'v404_publication_closure/launch.py')
shared = importlib.util.module_from_spec(module)
module.loader.exec_module(shared)

shared.HERE = HERE
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v405-causal-coordination/run03')
shared.SOURCE = shared.ROOT.parent / 'source03'


def stage():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in ('coordination.py', 'run.py', 'tests.py', 'protocol.json'):
            path = HERE / name
            if path.suffix == '.py':
                compile(path.read_bytes(), name, 'exec')
            archive.add(path, arcname=name)
    payload = base64.b64encode(buf.getvalue()).decode()
    code = ('import base64,io,tarfile\nfrom pathlib import Path\n'
            f'p=Path({str(shared.SOURCE)!r});p.mkdir(parents=True,exist_ok=True)\n'
            f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))).extractall(p)\n')
    rc, out, err = shared.scheduler.run_on('node003', shlex.join([shared.deployment()['python'],
        '-c', code]), timeout=90, check=False)
    assert rc == 0, (out, err[-1500:])


def spec(label, command, *args, result=None):
    job = shared.spec(label, 'run.py', [], result=result)
    job.update(description='FreqDuet V405 ' + label, signature='FreqDuet/v405/' + label)
    job['env'].pop('PYTHONPATH', None)
    job['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
        str(shared.SOURCE / 'run.py'), command, '--root', str(shared.ROOT), *map(str, args)])
    return job


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('checks', 'repair', 'aggregate'))
    p.add_argument('--regime')
    p.add_argument('--day', type=int)
    a = p.parse_args()
    if a.phase == 'checks':
        stage()
        shared.wait(shared.submit('checks_recording', [spec('checks_recording', 'checks',
            result=shared.ROOT / 'checks/summary')]))
        return
    if a.phase == 'aggregate':
        shared.wait(shared.submit('aggregate', [spec('aggregate', 'aggregate')]))
        return
    plan = shared.read(HERE / 'protocol.json')
    assert shared.read(HERE / 'review/checks_recording/results.json')['passed']
    assert a.day in plan['days'][a.regime]
    label = f'repair_{a.regime}_{a.day}'
    path = shared.ROOT / 'cases' / a.regime / str(a.day) / 'coordinated/summary'
    shared.wait(shared.submit(label, [spec(label, 'case', '--regime', a.regime,
        '--day', a.day, '--arm', 'coordinated', result=path)]))


if __name__ == '__main__':
    main()
