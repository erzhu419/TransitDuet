"""Stage and submit the bounded confirmation on the scheduler CPU pool."""

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
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v408-heavy-confirmation/run01')
shared.SOURCE = shared.ROOT.parent / 'source01'
PLAN = shared.read(HERE / 'protocol.json')


def stage():
    if (HERE / 'submission_preflight.json').exists():
        return
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in ('confirm.py', 'protocol.json', 'PROTOCOL.md'):
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


def spec(label, phase, *args, result):
    job = shared.spec(label, 'confirm.py', [], result=result)
    job.update(description='FreqDuet V408 ' + label, signature='FreqDuet/v408/' + label)
    job['env'].pop('PYTHONPATH', None)
    job['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
        str(shared.SOURCE / 'confirm.py'), phase, '--root', str(shared.ROOT), *map(str, args)])
    return job


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('preflight', 'traffic', 'aggregate', 'placement'))
    a = p.parse_args()
    if a.phase == 'preflight':
        stage()
        shared.wait(shared.submit('preflight', [spec('preflight', 'preflight',
            result=shared.ROOT / 'preflight/summary')]))
    elif a.phase == 'traffic':
        gate = shared.read(HERE / 'review/preflight/results.json')
        days = PLAN['days']['heavy']
        assert gate['passed'] and gate['physical_starts'] == 0
        assert gate['days'] == days and gate['arms'] == PLAN['arms']
        assert len(days) <= PLAN['max_concurrent_days']
        jobs = [spec('heavy_' + str(d), 'day', '--day', d,
                     result=shared.ROOT / 'paired' / str(d) / 'summary') for d in days]
        print(shared.submit('traffic', jobs), flush=True)
    elif a.phase == 'aggregate':
        tasks = shared.read(HERE / 'submission_traffic.json')['tasks']
        wanted = {t['id'] for t in tasks}
        statuses = {t['status'] for t in shared.scheduler.load_state()['tasks']
                    if t['id'] in wanted}
        assert statuses and statuses <= {'done', 'failed', 'cancelled'}, statuses
        shared.wait(shared.submit('aggregate', [spec('aggregate', 'aggregate',
            result=shared.ROOT / 'aggregate/summary')]))
    else:
        assert shared.read(HERE / 'review/aggregate/results.json')['complete']
        script = shared.ROOT.parent / 'analysis01/placement.py'
        receipt = HERE / 'submission_placement_file.json'
        if not receipt.exists():
            source = (HERE / 'placement.py').read_bytes()
            compile(source, 'placement.py', 'exec')
            payload = base64.b64encode(source).decode()
            code = ('import base64\nfrom pathlib import Path\n'
                    f'p=Path({str(script)!r});p.parent.mkdir(exist_ok=True)\n'
                    f'with p.open("xb") as f: f.write(base64.b64decode({payload!r}))\n')
            rc, out, err = shared.scheduler.run_on('node004', shlex.join([
                shared.deployment()['python'], '-c', code]), timeout=30, check=False)
            assert rc == 0, (out, err[-1500:])
        job = spec('placement_file', 'aggregate', result=shared.ROOT / 'placement/summary')
        job['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
                                 str(script), str(shared.ROOT)])
        shared.wait(shared.submit('placement_file', [job]))


if __name__ == '__main__':
    main()
