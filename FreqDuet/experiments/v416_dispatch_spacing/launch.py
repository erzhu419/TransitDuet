"""Submit only the V416 zero-traffic stage."""

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


if __name__ == '__main__':
    main()
