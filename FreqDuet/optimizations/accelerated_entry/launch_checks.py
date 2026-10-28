"""Provision the independent entry and manually dispatch zero-traffic checks."""

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


def launch():
    stage = REMOTE/'optimizations/accelerated_entry/source01'
    output = REMOTE/'optimizations/accelerated_entry/check01/summary'
    template = next(t for t in json.loads(scheduler.QUEUE_FILE.read_text())['tasks'] if t['id'] == 't212120')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for path in (HERE/'controllers.py', HERE/'run.py', HERE/'test_entry.py', HERE/'verify.py',
                     HERE/'final_acceptance.py', HERE.parent/'v277_exact/fast_continuation.py',
                     HERE.parent/'v277_exact/test_fast_continuation.py'):
            archive.add(path, arcname=path.name)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    code = ('import base64,io,pathlib,tarfile;'
        f'p=pathlib.Path({str(stage)!r});p.mkdir(parents=True,exist_ok=False);'
        f't=tarfile.open(fileobj=io.BytesIO(base64.b64decode({encoded!r})));t.extractall(p);print(p)')
    rc, out, err = scheduler.run_on('node004', shlex.join([PYTHON, '-c', code]), timeout=60, check=False)
    print(out, err, flush=True)
    if rc:
        raise RuntimeError('New-only entry staging failed')
    env = dict(template['extra_env'])
    env['PYTHONPATH'] = str(stage)+':'+env['PYTHONPATH']
    command = shlex.join([PYTHON, '-u', str(stage/'verify.py'), '--out', str(output/'results.json'),
        '--probe-root', str(REMOTE/'freqduet/results_freqduet/accelerated_entry_metadata_probe')])
    signature = 'FreqDuet/accelerated-entry/source01/checks'
    args = [sys.executable, str(SKILL/'scheduler.py'), 'submit', '--description',
        'FreqDuet generic accelerated entry synthetic and metadata checks; zero traffic',
        '--cmd', command, '--cwd', str(stage), '--signature', signature, '--project', 'FreqDuet',
        '--priority', 'high', '--require-node', 'node004', '--vram', '0', '--ram-mb', '4096', '--cpu', '1',
        '--no-auto-requeue', '--skip-launch-staging', '--result-dir', str(output),
        '--local-result-dir', str(HERE/'checks/summary'), '--env']
    args.extend(k+'='+v for k,v in env.items())
    result = subprocess.run(args, capture_output=True, text=True)
    print(result.stdout, result.stderr, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    task = next(t for t in json.loads(scheduler.QUEUE_FILE.read_text())['tasks'] if t['signature'] == signature)
    with (HERE/'deployment.json').open('x') as stream:
        json.dump(dict(node='node004', python=PYTHON, source_dir=str(stage), env=env,
            results_root=str(REMOTE/'freqduet/results_freqduet'), check_task=task['id'],
            check_result=str(output/'results.json'), kernel_source='cbf336f036'), stream, indent=2)
        stream.write('\n')
    subprocess.run([sys.executable, str(SKILL/'scheduler.py'), 'dispatch', '--task-id', task['id'],
                    '--lock-timeout', '60'], check=True)
    print('ACCELERATED_ENTRY_CHECK_TASK', task['id'], flush=True)


if __name__ == '__main__':
    launch()
