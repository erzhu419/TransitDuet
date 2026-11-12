"""Submit the zero-traffic preflight or one frozen V412 parity path."""

import argparse
import base64
import importlib.util
import io
from pathlib import Path
import shlex
import tarfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('shared', HERE.parent / 'v404_publication_closure/launch.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
shared.HERE = HERE
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v413-public-slot-observer/run01')
shared.SOURCE = shared.ROOT.parent / 'source02'
PREFLIGHT = 'preflight02'


def stage():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tar:
        for name in ('observer.py', 'slot_checks.py', 'run.py', 'protocol.json', 'PROTOCOL.md'):
            path = HERE / name
            if path.suffix == '.py':
                compile(path.read_bytes(), name, 'exec')
            tar.add(path, arcname=name)
    code = ('import base64,io,tarfile\nfrom pathlib import Path\n'
            f'p=Path({str(shared.SOURCE)!r});p.mkdir(parents=True,exist_ok=False)\n'
            f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({base64.b64encode(buf.getvalue()).decode()!r}))).extractall(p)\n')
    rc, out, err = shared.scheduler.run_on('node004', shlex.join([
        shared.deployment()['python'], '-c', code]), timeout=60, check=False)
    assert rc == 0, (out, err[-1500:])


def job(label, phase):
    item = shared.spec(label, 'run.py', [], result=shared.ROOT / phase / 'summary')
    item.update(description=f'FreqDuet V413 public-slot {phase}', signature=f'FreqDuet/v413/{label}')
    item['env'].pop('PYTHONPATH', None)
    item['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
                             str(shared.SOURCE / 'run.py'), phase, '--root', str(shared.ROOT)])
    return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', nargs='?', choices=('preflight', 'parity'), default='preflight')
    if parser.parse_args().phase == 'preflight':
        if not (HERE / f'submission_{PREFLIGHT}.json').exists():
            stage()
        shared.wait(shared.submit(PREFLIGHT, [job(PREFLIGHT, 'preflight')]))
    else:
        protocol = shared.read(HERE / 'protocol.json')
        gate = shared.read(HERE / 'review' / PREFLIGHT / 'results.json')
        assert gate['passed'] and gate['physical_starts'] == 0
        assert gate['parity_day'] == protocol['engineering_day']
        assert protocol['planned_parity_paths'] == 1 and not protocol['automatic_retry']
        print(shared.submit('parity', [job('parity', 'parity')]), flush=True)
