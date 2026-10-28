"""Stage only new candidate files and manually dispatch a server-only test."""

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
import scheduler as scheduler

HERE = Path(__file__).resolve().parent
REMOTE = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v4-worktree/FreqDuet')
PYTHON = '/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python'


def launch():
    stage = REMOTE/'optimizations/v277_exact/candidate01'
    output = REMOTE/'optimizations/v277_exact/run01'
    previous = next(t for t in json.loads(scheduler.QUEUE_FILE.read_text())['tasks'] if t['id'] == 't212088')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name in ('fast_continuation.py', 'test_fast_continuation.py', 'benchmark.py'):
            archive.add(HERE/name, arcname=name)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    # Copy into a new exclusive directory; never sync or overwrite live sources.
    code = ('import base64,io,pathlib,tarfile;'
        f'p=pathlib.Path({str(stage)!r});p.mkdir(parents=True,exist_ok=False);'
        f'b=base64.b64decode({encoded!r});'
        't=tarfile.open(fileobj=io.BytesIO(b));t.extractall(p);print(p)')
    rc, out, err = scheduler.run_on('node004', shlex.join([PYTHON, '-c', code]), timeout=60, check=False)
    print(out, err, flush=True)
    if rc:
        raise RuntimeError('New-only source staging failed')
    env = dict(previous['extra_env'])
    env['PYTHONPATH'] = str(stage) + ':' + env['PYTHONPATH']
    frozen = REMOTE/'freqduet/results_freqduet/protocol_v6_v277_metric_alignment/source_b9909ddc03/FreqDuet/freqduet/scripts/freqduet_flow_controller.py'
    command = shlex.join([PYTHON, '-u', str(stage/'benchmark.py'), '--out', str(output),
        '--source', 'v277-exact-candidate01', '--frozen-flow', str(frozen)])
    args = [sys.executable, str(SKILL/'scheduler.py'), 'submit', '--description',
        'FreqDuet detached sparse-AVL exact-output tests and profile; zero traffic; old code untouched',
        '--cmd', command, '--cwd', str(stage), '--signature', 'FreqDuet/v277/detached-exact/candidate01',
        '--project', 'FreqDuet', '--priority', 'high', '--require-node', 'node004',
        '--vram', '0', '--ram-mb', '4096', '--cpu', '1', '--no-auto-requeue',
        '--skip-launch-staging', '--result-dir', str(output/'summary'),
        '--local-result-dir', str(HERE/'results/run01/summary'), '--env']
    args.extend(k+'='+v for k, v in env.items())
    submitted = subprocess.run(args, check=True, capture_output=True, text=True)
    print(submitted.stdout, flush=True)
    state = json.loads(scheduler.QUEUE_FILE.read_text())
    task = next(t for t in state['tasks'] if t['signature'] == 'FreqDuet/v277/detached-exact/candidate01')
    subprocess.run([sys.executable, str(SKILL/'scheduler.py'), 'dispatch', '--task-id', task['id'],
                    '--lock-timeout', '60'], check=True)
    print('DETACHED_TASK', task['id'], flush=True)


if __name__ == '__main__':
    launch()
