"""Submit the missing arm and its readout behind a zero-traffic gate."""

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
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v417-pooled-spacing/run01')
shared.SOURCE = shared.ROOT.parent / 'source01'


def job(label, phase, output, day=None, waits=()):
    task = shared.spec(label, 'study.py', [], result=shared.ROOT / output / 'summary')
    task.update(description=f'FreqDuet V417 {label}', signature=f'FreqDuet/v417/{label}',
                wait_for_files=[str(HERE / 'review' / name) for name in waits])
    task['env'].pop('PYTHONPATH', None)
    cmd = [shared.deployment()['python'], '-B', '-I', '-u', str(shared.SOURCE / 'study.py'),
           phase, '--root', str(shared.ROOT)]
    if day is not None:
        cmd += ['--day', str(day)]
    task['cmd'] = shlex.join(cmd)
    return task


def main():
    plan = shared.read(HERE / 'protocol.json')
    paid = shared.read(HERE.parent / 'v416_dispatch_spacing/review/aggregate/results.json')
    assert paid['complete'] and not paid['development_screen_passed'] and paid['physical_starts'] == 13
    if not (HERE / 'submission_preflight.json').exists():
        prior.stage(shared.SOURCE, ('study.py', 'protocol.json', 'PROTOCOL.md'))
    shared.wait(shared.submit('preflight', [job('preflight', 'preflight', 'preflight')]))
    gate = shared.read(HERE / 'review/preflight/results.json')
    days = plan['development_days']
    assert gate['passed'] and gate['physical_starts'] == 0 and gate['reused_paths'] == 12
    assert gate['days'] == days and gate['arm'] == plan['new_arm']
    tasks = [job(f'day_{d}', 'case', f'days/{d}', d) for d in days]
    tasks += [job('aggregate', 'aggregate', 'aggregate',
                  waits=tuple(f'day_{d}/results.json' for d in days))]
    if not (HERE / 'submission_factorial.json').exists():
        signatures = {t['signature'] for t in tasks}
        assert not any(t.get('signature') in signatures for t in shared.scheduler.load_state()['tasks'])
    print(shared.submit('factorial', tasks), flush=True)


if __name__ == '__main__':
    main()
