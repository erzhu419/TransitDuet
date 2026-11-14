"""Submit each V416 stage with its frozen release dependencies."""

import argparse
import importlib.util
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_launcher',
    HERE.parent / 'v413_public_slot_observer/launch.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
shared = prior.shared
shared.HERE = prior.HERE = HERE
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v416-dispatch-spacing/run01')
shared.SOURCE = shared.ROOT.parent / 'source01'


def main():
    receipt = HERE / 'submission_preflight.json'
    if not receipt.exists():
        prior.stage(shared.SOURCE, ('checks.py', 'spacing.py', 'PROTOCOL.md'))
    task = shared.spec('preflight', 'checks.py', [], result=shared.ROOT/'preflight/summary')
    task.update(description='FreqDuet V416 spacing preflight', signature='FreqDuet/v416/preflight')
    task['env'].pop('PYTHONPATH', None)
    task['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
                             str(shared.SOURCE/'checks.py'), '--root', str(shared.ROOT)])
    shared.wait(shared.submit('preflight', [task]))


def traffic():
    gate = shared.read(HERE / 'review/preflight/results.json')
    assert gate['passed'] and gate['physical_starts'] == 0
    plan = shared.read(HERE / 'traffic_protocol.json')
    source = shared.ROOT.parent / 'traffic01'

    def job(label, phase, output, day=None, waits=()):
        task = shared.spec(label, 'traffic.py', [], result=shared.ROOT / output / 'summary')
        task.update(description=f'FreqDuet V416 {label}', signature=f'FreqDuet/v416/{label}',
                    cwd=str(source), wait_for_files=[str(HERE / 'review' / p) for p in waits])
        task['env'].pop('PYTHONPATH', None)
        cmd = [shared.deployment()['python'], '-B', '-I', '-u', str(source / 'traffic.py'),
               phase, '--root', str(shared.ROOT)]
        if day is not None:
            cmd += ['--day', str(day)]
        task['cmd'] = shlex.join(cmd)
        return task

    if not (HERE / 'submission_traffic_preflight.json').exists():
        prior.stage(source, ('traffic.py', 'traffic_protocol.json'))
    label = 'traffic_preflight'
    shared.wait(shared.submit(label, [job(label, 'preflight', label)]))
    checked = shared.read(HERE / f'review/{label}/results.json')
    days = plan['development_days']
    assert checked['passed'] and checked['physical_starts'] == 0 and checked['days'] == days
    assert checked['engineering_day'] == plan['engineering_day'] and checked['arms'] == plan['arms']
    requests = [job('engineering', 'engineering', 'engineering')]
    requests += [job(f'triple_{day}', 'day_case', f'triples/{day}', day,
                     waits=('engineering/results.json',)) for day in days]
    requests += [job('aggregate', 'aggregate', 'aggregate',
                     waits=tuple(f'triple_{day}/results.json' for day in days))]
    if not (HERE / 'submission_traffic.json').exists():
        signatures = {task['signature'] for task in requests}
        assert not any(t.get('signature') in signatures for t in shared.scheduler.load_state()['tasks'])
    print(shared.submit('traffic', requests), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', nargs='?', choices=('preflight', 'traffic'), default='preflight')
    phase = parser.parse_args().phase
    if phase == 'preflight':
        main()
    else:
        traffic()
