"""Stage the independent panel and submit it only after preflight."""

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
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v418-spacing-confirmation/run01')
shared.SOURCE = shared.ROOT.parent / 'source01'


def job(label, phase, output, day=None, waits=()):
    task = shared.spec(label, 'study.py', [], result=shared.ROOT / output / 'summary')
    task.update(description=f'FreqDuet V418 {label}', signature=f'FreqDuet/v418/{label}',
                wait_for_files=[str(HERE / 'review' / name) for name in waits])
    task['env'].pop('PYTHONPATH', None)
    command = [shared.deployment()['python'], '-B', '-I', '-u', str(shared.SOURCE / 'study.py'),
               phase, '--root', str(shared.ROOT)]
    if day is not None:
        command += ['--day', str(day)]
    task['cmd'] = shlex.join(command)
    return task


def preflight():
    paid = shared.read(HERE.parent / 'v417_pooled_spacing/review/aggregate/results.json')
    assert paid['complete'] and paid['physical_starts'] == 4 and paid['reused_paths'] == 12
    if not (HERE / 'submission_preflight.json').exists():
        prior.stage(shared.SOURCE, ('study.py', 'protocol.json', 'PROTOCOL.md'))
    shared.wait(shared.submit('preflight', [job('preflight', 'preflight', 'preflight')]))


def confirmation():
    plan = shared.read(HERE / 'protocol.json')
    gate = shared.read(HERE / 'review/preflight/results.json')
    days = plan['confirmation_days']
    assert gate['passed'] and gate['physical_starts'] == 0
    assert gate['days'] == days and gate['arms'] == plan['arms']
    tasks = [job(f'day_{day}', 'case', f'days/{day}', day) for day in days]
    tasks.append(job('aggregate', 'aggregate', 'aggregate',
                     waits=tuple(f'day_{day}/results.json' for day in days)))
    if not (HERE / 'submission_confirmation.json').exists():
        signatures = {t['signature'] for t in tasks}
        assert not any(t.get('signature') in signatures for t in shared.scheduler.load_state()['tasks'])
    print(shared.submit('confirmation', tasks), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'confirmation'))
    {'preflight': preflight, 'confirmation': confirmation}[parser.parse_args().phase]()
