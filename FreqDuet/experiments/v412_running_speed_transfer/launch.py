"""Stage the frozen road-speed study and submit its dependent jobs."""

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
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v412-running-speed-transfer/run01')
shared.SOURCE = shared.ROOT.parent / 'source02'
PLAN = shared.read(HERE / 'protocol.json')


def stage():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in ('study412.py', 'protocol.json', 'PROTOCOL.md'):
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


def job(label, phase, result, *, case=None, waits=()):
    item = shared.spec(label, 'study412.py', [], result=shared.ROOT / result / 'summary')
    item.update(description='FreqDuet V412 ' + label, signature='FreqDuet/v412/' + label,
                wait_for_files=[str(HERE / 'review' / f) for f in waits])
    item['env'].pop('PYTHONPATH', None)
    argv = [shared.deployment()['python'], '-B', '-I', '-u',
            str(shared.SOURCE / 'study412.py'), phase, '--root', str(shared.ROOT)]
    if case:
        argv += ['--regime', case[0], '--day', str(case[1])]
    item['cmd'] = shlex.join(argv)
    return item


def main():
    if not (HERE / 'submission_preflight02.json').exists():
        stage()
    shared.wait(shared.submit('preflight02', [job('preflight02', 'preflight', 'preflight')]))
    gate = shared.read(HERE / 'review/preflight02/results.json')
    assert gate['passed'] and gate['physical_starts'] == 0 and gate['days'] == PLAN['days']
    tasks = []
    all_files, first_files = [], []
    for regime, days in PLAN['days'].items():
        for i, day in enumerate(days):
            label = f'{regime}_{day}'
            file = f'{label}/results.json'
            all_files.append(file)
            if i == 0:
                first_files.append(file)
            tasks.append(job(label, 'case', f'pairs/{regime}/{day}', case=(regime, day),
                             waits=['engineering/ready.json'] if i else ()))
    tasks += [job('engineering', 'engineering', 'engineering', waits=first_files),
              job('aggregate', 'aggregate', 'aggregate', waits=all_files)]
    print(shared.submit('pipeline', tasks), flush=True)


if __name__ == '__main__':
    main()
