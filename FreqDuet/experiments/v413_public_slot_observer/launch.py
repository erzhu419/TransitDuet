"""Only the zero-traffic preflight can be submitted from this stage."""

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
LABEL = 'preflight02'


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


if __name__ == '__main__':
    if not (HERE / f'submission_{LABEL}.json').exists():
        stage()
    item = shared.spec(LABEL, 'run.py', [], result=shared.ROOT / 'preflight/summary')
    item.update(description='FreqDuet V413 public-slot preflight', signature=f'FreqDuet/v413/{LABEL}')
    item['env'].pop('PYTHONPATH', None)
    item['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
                             str(shared.SOURCE / 'run.py'), 'preflight', '--root', str(shared.ROOT)])
    shared.wait(shared.submit(LABEL, [item]))
