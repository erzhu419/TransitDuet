"""Stage a new current-action engineering check and dispatch one server path."""

import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

SKILL = Path('/home/erzhu419/mine_code/scheduleurm/skill')
sys.path.insert(0, str(SKILL))
import scheduler

HERE = Path(__file__).resolve().parent
REMOTE = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v4-worktree/FreqDuet')
PYTHON = '/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python'
SIGNATURE = 'FreqDuet/v277/detached-current-exact/candidate01'


def launch():
    stage = REMOTE/'optimizations/v277_current_exact/candidate01'
    root = REMOTE/'freqduet/results_freqduet/protocol_v6_v277_current_exact'
    state = json.loads(scheduler.QUEUE_FILE.read_text())
    template = next(t for t in state['tasks'] if t['id'] == 't212103')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path in (HERE/'run_current.py', HERE/'protocol.json',
                     HERE.parent/'v277_exact/fast_continuation.py',
                     HERE.parent/'v277_exact/test_fast_continuation.py',
                     HERE.parent/'v277_full_day/run_full_day.py'):
            archive.add(path, arcname=path.name)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    code = ('import base64,io,pathlib,tarfile;'
        f'p=pathlib.Path({str(stage)!r});p.mkdir(parents=True,exist_ok=False);'
        f't=tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r})));t.extractall(p);print(p)')
    rc, out, err = scheduler.run_on('node004', shlex.join([PYTHON, '-c', code]), timeout=60, check=False)
    print(out, err, flush=True)
    if rc:
        raise RuntimeError('New-only staging failed; no traffic submitted')
    env = dict(template['extra_env'])
    env['PYTHONPATH'] = str(stage)+':'+env['PYTHONPATH']
    command = shlex.join([PYTHON, '-u', str(stage/'run_current.py'), '--root', str(root), '--results', str(root.parent)])
    args = [sys.executable, str(SKILL/'scheduler.py'), 'submit', '--description',
        'FreqDuet one detached current-action engineering replay of archived p17; exact parity; no retry',
        '--cmd', command, '--cwd', str(stage), '--signature', SIGNATURE, '--project', 'FreqDuet',
        '--priority', 'high', '--require-node', 'node004', '--vram', '0', '--ram-mb', '4096', '--cpu', '1',
        '--no-auto-requeue', '--skip-launch-staging', '--result-dir', str(root/'summary'),
        '--local-result-dir', str(HERE/'results/summary'), '--env']
    args.extend(k+'='+v for k,v in env.items())
    result = subprocess.run(args, capture_output=True, text=True)
    print(result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    state = json.loads(scheduler.QUEUE_FILE.read_text())
    task = next(t for t in state['tasks'] if t['signature'] == SIGNATURE)
    with (HERE/'submission.json').open('x') as stream:
        json.dump(dict(task=task['id'], node='node004', priority='high', reference_day=2077002,
            physical_start_cap=1, automatic_retry=False, remote_source=str(stage), remote_results=str(root),
            old_tasks_modified=False), stream, indent=2)
        stream.write('\n')
    subprocess.run([sys.executable, str(SKILL/'scheduler.py'), 'dispatch', '--task-id', task['id'],
                    '--lock-timeout', '60'], check=True)
    print('CURRENT_EXACT_TASK', task['id'], flush=True)


if __name__ == '__main__':
    launch()
