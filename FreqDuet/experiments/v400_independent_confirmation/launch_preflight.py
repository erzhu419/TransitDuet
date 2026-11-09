"""Stage only metadata checks, submit high on node004, and explicitly dispatch."""

import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile


HERE = Path(__file__).resolve().parent
SCHEDULER = Path('/home/erzhu419/mine_code/scheduleurm/skill')
sys.path.insert(0, str(SCHEDULER))
import scheduler

REMOTE = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v400-independent-confirmation')
SOURCE = REMOTE/'source01'
PYTHON = '/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    plan = json.loads((HERE/'manifest.json').read_text())
    signature = 'FreqDuet/v400/zero-traffic-preflight/source01'
    tasks = json.loads(scheduler.QUEUE_FILE.read_text())['tasks']
    assert not any(t.get('signature') == signature for t in tasks), 'Already submitted; inspect its receipt.'
    collisions, examined = [], 0
    for task in tasks:
        if task.get('project') != 'FreqDuet':
            continue
        examined += 1
        argv = shlex.split(task.get('cmd', ''))
        for i, token in enumerate(argv):
            value = argv[i+1] if token == '--day' and i+1 < len(argv) else (
                token.split('=', 1)[1] if token.startswith('--day=') else '')
            if value.isdigit() and int(value) in plan['days']:
                collisions.append({'task': task['id'], 'status': task['status'], 'day': int(value)})
    queue = {'days': plan['days'], 'tasks_examined': examined, 'collisions': collisions}

    # This is a read-only directory/JSON query, not a rollout or model evaluation.
    script = (HERE/'preflight.py').read_text()
    query = 'namespace={"__name__":"inventory_only"}\n'
    query += f'exec({script!r},namespace)\n'
    query += f'root=namespace["Path"]({plan["gpu_metadata_root"]!r})\n'
    query += 'roots=sorted(str(p) for p in root.glob("FreqDuet*") if p.is_dir() and p.name not in '
    query += '("FreqDuet-v347-runtime","FreqDuet-v347-deploy","FreqDuet-v400-independent-confirmation"))\n'
    query += f'metadata=namespace["inspect_metadata"](roots,set({plan["days"]!r}))\n'
    query += f'print(namespace["json"].dumps({{"days":{plan["days"]!r},"metadata":metadata}}))\n'
    rc, out, err = scheduler.run_on('jtl110gpu2', shlex.join([
        '/home/erzhu419/.conda/envs/freqduet-paper/bin/python', '-c', query]), timeout=120, check=False)
    if rc:
        raise RuntimeError(f'GPU metadata query failed: {err[-1200:]}')
    gpu = json.loads(out)
    gpu['queried_at_utc'] = datetime.now(timezone.utc).isoformat()
    save(HERE/'preflight_inputs/gpu_inventory.json', gpu)
    save(HERE/'preflight_inputs/queue_inventory.json', queue)
    dispatch_preflight()


def dispatch_preflight():
    signature = 'FreqDuet/v400/zero-traffic-preflight/source01'
    names = ['manifest.json', 'PROTOCOL.md', 'preflight.py',
             'preflight_inputs/gpu_inventory.json', 'preflight_inputs/queue_inventory.json']
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in names:
            archive.add(HERE/name, arcname=name)
    encoded = base64.b64encode(buf.getvalue()).decode()
    deploy = 'import base64,io,tarfile\nfrom pathlib import Path\n'
    deploy += f'p=Path({str(SOURCE)!r});p.mkdir(parents=True,exist_ok=False)\n'
    deploy += f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r}))).extractall(p)\n'
    rc, out, err = scheduler.run_on('node004', shlex.join([PYTHON, '-c', deploy]), timeout=60, check=False)
    if rc:
        raise RuntimeError(f'Deployment failed: {err[-1200:]}')
    result_dir = REMOTE/'preflight01/summary'
    spec = dict(description='FreqDuet V400 frozen plan and metadata preflight; zero traffic',
                signature=signature, project='FreqDuet', priority='high', require_node='node004',
                cpu=1, ram_mb=1024, vram=0, no_auto_requeue=True, skip_launch_staging=True,
                cwd=str(SOURCE), env={'PYTHONDONTWRITEBYTECODE':'1'},
                cmd=shlex.join([PYTHON, '-u', str(SOURCE/'preflight.py'),
                    '--manifest', str(SOURCE/'manifest.json'),
                    '--gpu-inventory', str(SOURCE/'preflight_inputs/gpu_inventory.json'),
                    '--queue-inventory', str(SOURCE/'preflight_inputs/queue_inventory.json'),
                    '--out', str(result_dir/'results.json')]),
                result_dir=str(result_dir), local_result_dir=str(HERE/'review01'))
    proc = subprocess.run([sys.executable, str(SCHEDULER/'scheduler.py'), 'submit-jsonl',
        '--stdin', '--trusted', '--json', '--lock-timeout', '120'],
        input=json.dumps([spec]), capture_output=True, text=True, check=True)
    task = json.loads(proc.stdout)['submitted'][0]
    save(HERE/'submission.json', {'task':task['id'], 'priority':'high', 'node':'node004',
         'signature':signature, 'remote_result':str(result_dir/'results.json'),
         'physical_starts':0, 'training_updates':0})
    print('V400_REGISTERED', task['id'], flush=True)
    subprocess.run([sys.executable, str(SCHEDULER/'scheduler.py'), 'dispatch',
                    '--task-id', task['id'], '--lock-timeout', '120'], check=True)


if __name__ == '__main__':
    main()
