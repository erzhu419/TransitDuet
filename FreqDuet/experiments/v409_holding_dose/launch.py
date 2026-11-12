"""Submit the entire gated CPU pipeline once; no local scientific work."""

import argparse
import base64
import importlib.util
import io
from pathlib import Path
import shlex
import tarfile

HERE = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location(
    'v404_launcher', HERE.parent / 'v404_publication_closure/launch.py')
shared = importlib.util.module_from_spec(loader)
loader.loader.exec_module(shared)
shared.HERE = HERE
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v409-holding-dose/run01')
shared.SOURCE = shared.ROOT.parent / 'source02'
PLAN = shared.read(HERE / 'protocol.json')


def stage():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in ('study409.py', 'protocol.json', 'PROTOCOL.md'):
            path = HERE / name
            if path.suffix == '.py':
                compile(path.read_bytes(), name, 'exec')
            archive.add(path, arcname=name)
    payload = base64.b64encode(buf.getvalue()).decode()
    code = ('import base64,io,tarfile\nfrom pathlib import Path\n'
            f'p=Path({str(shared.SOURCE)!r});p.mkdir(parents=True,exist_ok=False)\n'
            f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))).extractall(p)\n')
    rc, out, err = shared.scheduler.run_on('node004', shlex.join([
        shared.deployment()['python'], '-c', code]), timeout=90, check=False)
    assert rc == 0, (out, err[-1500:])


def job(label, phase, result, *, day=None, waits=()):
    item = shared.spec(label, 'study409.py', [], result=shared.ROOT / result / 'summary')
    item.update(description='FreqDuet V409 ' + label, signature='FreqDuet/v409/' + label,
                wait_for_files=[str(HERE / 'review' / f) for f in waits])
    item['env'].pop('PYTHONPATH', None)
    argv = [shared.deployment()['python'], '-B', '-I', '-u',
            str(shared.SOURCE / 'study409.py'), phase, '--root', str(shared.ROOT)]
    if day is not None:
        argv += ['--day', str(day)]
    item['cmd'] = shlex.join(argv)
    return item


def pipeline():
    if not (HERE / 'submission_preflight02.json').exists():
        stage()
    shared.wait(shared.submit('preflight02', [job('preflight02', 'preflight', 'preflight')]))
    gate = shared.read(HERE / 'review/preflight02/results.json')
    assert gate['passed'] and gate['physical_starts'] == 0
    dev, days = PLAN['development_days'], PLAN['days']['heavy']
    assert gate['days'] == days and gate['development_days'] == dev
    tasks = [job(f'dev_{d}', 'development', f'calibration_days/{d}', day=d) for d in dev]
    tasks += [job('calibration', 'calibrate', 'calibration',
                  waits=[f'dev_{d}/results.json' for d in dev])]
    tasks += [job(f'heavy_{d}', 'confirmation', f'pairs/{d}', day=d,
                  waits=['calibration/ready.json']) for d in days]
    tasks += [job('aggregate', 'aggregate', 'aggregate',
                  waits=[f'heavy_{d}/results.json' for d in days])]
    print(shared.submit('pipeline', tasks), flush=True)


def placement():
    assert shared.read(HERE / 'review/aggregate/results.json')['complete']
    path = shared.ROOT.parent / 'analysis01/placement.py'
    if not (HERE / 'submission_placement.json').exists():
        source = (HERE / 'placement.py').read_bytes()
        compile(source, 'placement.py', 'exec')
        payload = base64.b64encode(source).decode()
        code = ('import base64\nfrom pathlib import Path\n'
                f'p=Path({str(path)!r});p.parent.mkdir(exist_ok=True)\n'
                f'with p.open("xb") as f: f.write(base64.b64decode({payload!r}))\n')
        rc, out, err = shared.scheduler.run_on('node004', shlex.join([
            shared.deployment()['python'], '-c', code]), timeout=30, check=False)
        assert rc == 0, (out, err[-1500:])
    spec = job('placement', 'aggregate', 'placement')
    spec['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
                             str(path), str(shared.ROOT)])
    shared.wait(shared.submit('placement', [spec]))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', nargs='?', choices=('pipeline', 'placement'), default='pipeline')
    globals()[p.parse_args().phase]()
