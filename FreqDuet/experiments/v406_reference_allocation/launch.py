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
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v406-reference-allocation/run01')
shared.SOURCE = shared.ROOT.parent / 'source01'


def stage(protocol='protocol.json'):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in ('coordination.py', 'run.py', 'tests.py', 'protocol.json', 'readout.py'):
            path = HERE / (protocol if name == 'protocol.json' else name)
            if path.suffix == '.py':
                compile(path.read_bytes(), name, 'exec')
            archive.add(path, arcname=name)
    payload = base64.b64encode(buf.getvalue()).decode()
    code = ('import base64,io,tarfile\nfrom pathlib import Path\n'
            f'p=Path({str(shared.SOURCE)!r});p.mkdir(parents=True,exist_ok=False)\n'
            f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))).extractall(p)\n')
    rc, out, err = shared.scheduler.run_on('node003', shlex.join([shared.deployment()['python'],
        '-c', code]), timeout=90, check=False)
    assert rc == 0, (out, err[-1500:])


def spec(label, command, *args, result=None):
    job = shared.spec(label, 'run.py', [], result=result)
    job.update(description='FreqDuet V406 ' + label, signature='FreqDuet/v406/' + label)
    job['env'].pop('PYTHONPATH', None)
    job['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
        str(shared.SOURCE / 'run.py'), command, '--root', str(shared.ROOT), *map(str, args)])
    return job


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('checks', 'traffic', 'readout',
                                   'unit_checks', 'unit_traffic', 'unit_readout'))
    a = p.parse_args()
    unit = a.phase.startswith('unit_')
    phase = a.phase.removeprefix('unit_')
    prefix = 'unit_' if unit else ''
    if unit:
        shared.ROOT = shared.ROOT.parent / 'run02'
        shared.SOURCE = shared.ROOT.parent / 'source02'
    protocol = 'unit_repair_protocol.json' if unit else 'protocol.json'
    if phase == 'checks':
        if not (HERE / ('submission_'+prefix+'checks.json')).exists():
            stage(protocol)
        shared.wait(shared.submit(prefix+'checks', [spec(prefix+'checks', 'checks',
            result=shared.ROOT / 'checks/summary')]))
        return
    if phase == 'readout':
        job = spec(prefix+'readout', 'checks', result=shared.ROOT/'readout/summary')
        job['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
            str(shared.SOURCE/'readout.py'), '--root', str(shared.ROOT)])
        shared.wait(shared.submit(prefix+'readout', [job]))
        return
    plan = shared.read(HERE / protocol)
    assert shared.read(HERE / 'review' / (prefix+'checks') / 'results.json')['passed']
    jobs = [spec(f'{prefix}{r}_{d}_{arm}', 'case', '--regime', r, '--day', d, '--arm', arm,
        result=shared.ROOT/'cases'/r/str(d)/arm/'summary')
        for r, days in plan['days'].items() for d in days
        for arm in plan.get('collected_arms', plan['arms'])]
    assert len(jobs) == plan['physical_start_cap']
    shared.wait(shared.submit(prefix+'traffic', jobs))


if __name__ == '__main__':
    main()
