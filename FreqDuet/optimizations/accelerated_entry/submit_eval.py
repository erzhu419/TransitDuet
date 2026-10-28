"""Submit one explicitly requested new accelerated evaluation, high/manual."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--actor', type=int, required=True)
    parser.add_argument('--day', type=int, required=True)
    parser.add_argument('--sampler', type=int, required=True)
    parser.add_argument('--controller', choices=('native_beam', 'arrival_beam', 'arrival_current'), required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    deployment = json.loads((here/'deployment.json').read_text())
    checks = json.loads((here/'checks_receipt.json').read_text())
    assert checks['status'] == 'ready_for_new_frozen_evaluations'
    remote = Path(deployment['results_root'])/args.name
    stage = Path(deployment['source_dir'])
    command = shlex.join([deployment['python'], '-u', str(stage/'run.py'), 'run', '--root', str(remote),
        '--actor', str(args.actor), '--day', str(args.day), '--sampler', str(args.sampler),
        '--controller', args.controller])
    scheduler = '/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py'
    signature = 'FreqDuet/accelerated-evaluation/'+args.name
    cmd = [sys.executable, scheduler, 'submit', '--description', 'FreqDuet accelerated frozen evaluation '+args.name,
        '--cmd', command, '--cwd', str(stage), '--signature', signature, '--project', 'FreqDuet',
        '--priority', 'high', '--require-node', deployment['node'], '--vram', '0', '--ram-mb', '4096',
        '--cpu', '1', '--no-auto-requeue', '--skip-launch-staging', '--result-dir', str(remote/'summary'),
        '--local-result-dir', str(here/'evaluations'/args.name/'summary'), '--env']
    cmd.extend(k+'='+v for k,v in deployment['env'].items())
    submitted = subprocess.run(cmd, capture_output=True, text=True)
    print(submitted.stdout, submitted.stderr, flush=True)
    if submitted.returncode:
        raise SystemExit(submitted.returncode)
    sys.path.insert(0, '/home/erzhu419/mine_code/scheduleurm/skill')
    import scheduler as api
    task = next(t for t in json.loads(api.QUEUE_FILE.read_text())['tasks'] if t['signature'] == signature)
    subprocess.run([sys.executable, scheduler, 'dispatch', '--task-id', task['id'], '--lock-timeout', '60'], check=True)


if __name__ == '__main__':
    main()
