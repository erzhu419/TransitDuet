"""Submit the frozen V415 comparison or its read-only divergence audit."""

import argparse
import importlib.util
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v413_launch', HERE.parent / 'v413_public_slot_observer/launch.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
shared = prior.shared
shared.HERE = prior.HERE = HERE
shared.ROOT = Path('/home/zhengliang01/scheduleurm_work/FreqDuet-v415-running-closed-loop/run01')
shared.SOURCE = shared.ROOT.parent / 'source01'
PLAN = shared.read(HERE / 'protocol.json')


def job(label, phase, output, *, day=None, waits=()):
    item = shared.spec(label, 'study415.py', [], result=shared.ROOT / output / 'summary')
    item.update(description=f'FreqDuet V415 {label}', signature=f'FreqDuet/v415/{label}',
                wait_for_files=[str(HERE / 'review' / path) for path in waits])
    item['env'].pop('PYTHONPATH', None)
    cmd = [shared.deployment()['python'], '-B', '-I', '-u', str(shared.SOURCE / 'study415.py'),
           phase, '--root', str(shared.ROOT)]
    if day is not None:
        cmd.extend(('--day', str(day)))
    item['cmd'] = shlex.join(cmd)
    return item


def main():
    source = shared.read(HERE.parent / 'v413_public_slot_observer/review/running_shadow_readout/results.json')
    assert source['primary_gate']['passed'] and source['paid_confirmation_paths'] == 8
    if not (HERE / 'submission_preflight.json').exists():
        prior.stage(shared.SOURCE, ('study415.py', 'protocol.json', 'PROTOCOL.md'))
    shared.wait(shared.submit('preflight', [job('preflight', 'preflight', 'preflight')]))
    gate = shared.read(HERE / 'review/preflight/results.json')
    days = PLAN['days']['slow']
    assert gate['passed'] and gate['physical_starts'] == 0 and gate['days'] == days
    requests = [job('engineering', 'engineering', 'engineering')]
    requests.extend(job(f'pair_{day}', 'pair', f'pairs/{day}', day=day,
                        waits=('engineering/results.json',)) for day in days)
    requests.append(job('aggregate', 'aggregate', 'aggregate',
                        waits=tuple(f'pair_{day}/results.json' for day in days)))
    if not (HERE / 'submission_pipeline.json').exists():
        signatures = {item['signature'] for item in requests}
        assert not any(task.get('signature') in signatures for task in shared.scheduler.load_state()['tasks'])
    print(shared.submit('pipeline', requests), flush=True)


def divergence():
    result = shared.read(HERE / 'review/aggregate/results.json')
    assert result['complete'] and not result['passed'] and result['physical_starts'] == 17
    label = 'divergence'
    source = shared.ROOT.parent / 'divergence01'
    if not (HERE / f'submission_{label}.json').exists():
        prior.stage(source, ('divergence.py',))
    item = job(label, label, label)
    item['cwd'] = str(source)
    item['cmd'] = shlex.join([shared.deployment()['python'], '-B', '-I', '-u',
                             str(source / 'divergence.py'), '--root', str(shared.ROOT)])
    shared.wait(shared.submit(label, [item]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', nargs='?', choices=('pipeline', 'divergence'), default='pipeline')
    {'pipeline': main, 'divergence': divergence}[parser.parse_args().phase]()
