"""Orchestrate the bounded publication closure on scheduler CPU nodes only."""

import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v404-publication-closure/run01')
SOURCE = ROOT.parent / 'source04'
sys.path.insert(0, '/home/erzhu419/mine_code/scheduleurm/skill')
import scheduler


def read(path):
    return json.loads(Path(path).read_text())


def deployment():
    return read(HERE.parent / 'v400_independent_confirmation/deployment.json')


def spec(label, script, args, *, result=None):
    d = deployment()
    env = dict(d['env'])
    env['PYTHONPATH'] = ':'.join((str(SOURCE), str(ROOT.parent.parent /
        'FreqDuet-v402-budget-mechanism/source01'), env['PYTHONPATH']))
    return dict(description='FreqDuet V404 ' + label, project='FreqDuet',
        signature='FreqDuet/v404/' + label, priority='high',
        allowed_nodes=['node001', 'node002', 'node003', 'node004', 'node005', 'node006'],
        cpu=1, ram_mb=4096, vram=0, allow_cpu_training=True,
        cpu_training_justification='Frozen CPU transit evaluation and manuscript evidence; no learning.',
        no_auto_requeue=True, skip_launch_staging=True, cwd=str(SOURCE), env=env,
        cmd=shlex.join([d['python'], '-u', str(SOURCE / script), '--root', str(ROOT),
                       '--assets', d['assets'], *map(str, args)]),
        result_dir=str(result or ROOT / label / 'summary'),
        local_result_dir=str(HERE / 'review' / label))


def submit(label, requests):
    receipt = HERE / ('submission_' + label + '.json')
    if receipt.exists():
        return read(receipt)['tasks']
    p = subprocess.run([sys.executable, scheduler.__file__, 'submit-jsonl', '--stdin',
        '--trusted', '--json', '--lock-timeout', '600'], input=json.dumps(requests),
        text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-3000:])
    tasks = [dict(id=t['id'], signature=t['signature']) for t in json.loads(p.stdout)['submitted']]
    receipt.write_text(json.dumps(dict(tasks=tasks), indent=2) + '\n')
    cmd = [sys.executable, scheduler.__file__, 'dispatch', '--lock-timeout', '120']
    for t in tasks:
        cmd += ['--task-id', t['id']]
    p = subprocess.run(cmd, text=True, capture_output=True)
    print(p.stdout[-4500:], p.stderr[-1000:], flush=True)
    if p.returncode:
        raise RuntimeError('manual dispatch failed; existing IDs remain registered')
    return tasks


def stage():
    if (HERE / 'staged04.json').exists():
        return
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for p in HERE.iterdir():
            if p.suffix in ('.py', '.json', '.md') and p.name != 'launch.py':
                if p.suffix == '.py':
                    compile(p.read_bytes(), p.name, 'exec')
                archive.add(p, arcname=p.name)
    code = 'import base64,io,tarfile\nfrom pathlib import Path\n'
    code += f'p=Path({str(SOURCE)!r});p.mkdir(parents=True,exist_ok=False)\n'
    code += f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({base64.b64encode(buf.getvalue()).decode()!r}))).extractall(p)\n'
    rc, out, err = scheduler.run_on('node003', shlex.join([deployment()['python'], '-c', code]),
                                   timeout=90, check=False)
    assert rc == 0, (out, err[-2000:])
    (HERE / 'staged04.json').write_text(json.dumps(dict(source=str(SOURCE), root=str(ROOT))) + '\n')


def stage_extra(names):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in names:
            path = HERE / name
            compile(path.read_bytes(), name, 'exec')
            archive.add(path, arcname=name)
    code = 'import base64,io,tarfile\n'
    code += f'tarfile.open(fileobj=io.BytesIO(base64.b64decode({base64.b64encode(buf.getvalue()).decode()!r}))).extractall({str(SOURCE)!r})\n'
    rc, out, err = scheduler.run_on('node003', shlex.join([deployment()['python'], '-c', code]),
                                   timeout=90, check=False)
    assert rc == 0, (out, err[-2000:])


def wait(tasks):
    command = [sys.executable, scheduler.__file__, 'wait-for', '--refresh', '--poll',
               '45', '--timeout', '0', '--verbose', '--task-id'] + [t['id'] for t in tasks]
    result = subprocess.run(command, text=True, capture_output=True)
    print(result.stdout[-1800:], result.stderr[-800:], flush=True)
    if result.returncode:
        raise RuntimeError('A registered stage failed; no traffic path will be retried.')
    pending = {t['id'] for t in tasks}
    previous = None
    while pending:
        state = {t['id']: t for t in scheduler.load_state()['tasks']}
        for task_id in list(pending):
            task = state[task_id]
            if task['status'] in ('failed', 'cancelled'):
                raise RuntimeError((task_id, task['status'], task.get('last_error')))
            if task['status'] == 'done':
                ok, err = scheduler._sync_one_result(task)
                assert ok, (task_id, err)
                pending.remove(task_id)
        if len(pending) != previous:
            print('V404_STAGE', len(tasks) - len(pending), '/', len(tasks), flush=True)
            previous = len(pending)
        if pending:
            time.sleep(45)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('preflight', 'all'))
    a = p.parse_args()
    stage()
    wait(submit('preflight02', [spec('preflight02', 'study.py', ['preflight'])]))
    if a.command == 'all':
        plan = read(HERE / 'protocol.json')
        assert read(HERE / 'review/preflight02/results.json')['passed']
        for label in ('observer_check','pressure_engineering','package_runtime'):
            actual='package_runtime_repair' if label=='package_runtime' and (HERE/'submission_package_runtime_repair.json').exists() else label
            wait(submit(actual,[spec(actual,'study.py',[label],result=ROOT/label/'summary')]))
            assert read(HERE/'review'/actual/'results.json')['passed']
        shared=plan['shared_native_case']
        wait(submit('shared_native',[spec('shared_native','study.py',['portable_case','--regime',shared['regime'],
            '--day',shared['day'],'--arm',shared['arm']],result=ROOT/'cases'/shared['regime']/
            str(shared['day'])/shared['arm']/'summary')]))
        wait(submit('native_gate',[spec('native_gate','study.py',['native_gate'])]))
        assert read(HERE/'review/native_gate/results.json')['passed']
        cases = [(r, d, arm) for r in plan['regimes'] for d in plan['days'][r] for arm in plan['arms']]
        cases=[c for c in cases if c!=(shared['regime'],shared['day'],shared['arm'])]
        for i in range(0, len(cases), plan['max_concurrent_paths']):
            requests = [spec(f'{r}_{d}_{arm}', 'study.py', ['portable_case', '--regime', r,
                '--day', d, '--arm', arm], result=ROOT / 'cases' / r / str(d) / arm / 'summary')
                for r, d, arm in cases[i:i + plan['max_concurrent_paths']]]
            wait(submit('batch_' + str(i), requests))
        if (HERE/'submission_aggregate_accept.json').exists():
            wait(read(HERE/'submission_aggregate_accept.json')['tasks'])
            (HERE/'review/aggregate').mkdir(exist_ok=True)
            shutil.copyfile(HERE/'review/aggregate_accept/results.json',HERE/'review/aggregate/results.json')
        else:
            wait(submit('aggregate', [spec('aggregate', 'study.py', ['aggregate'])]))
        print('V404_ALL_COMPLETE', flush=True)
