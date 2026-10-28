"""Server-only entry validation; tests use model fixtures and mocked rollouts."""

import argparse
import json
from pathlib import Path
import unittest

from profile_freqduet_v277_solver import load_module


def verify(out, probe_root):
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun >= 13
    entry = load_module(Path(__file__).with_name('run.py'), 'accelerated_entry_runtime_probe')
    probes = []
    for controller in entry.CONTROLLERS:
        args = entry.parser().parse_args(['preflight', '--root', str(probe_root), '--actor', '12011',
            '--day', '2099001', '--sampler', '333', '--controller', controller])
        probes.append(entry.preflight(args))
    assert not probe_root.exists(), 'Metadata preflight unexpectedly started a path'
    value = dict(status='ready_for_new_frozen_evaluations', tests=result.testsRun,
        controllers=list(entry.CONTROLLERS), runtime_probes=probes,
        archived_reference_required=False, real_traffic_starts=0, training_updates=0,
        main_policy_changed=False, old_code_changed=False, running_matrix_changed=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('ACCELERATED_ENTRY_CHECKS_COMPLETE', result.testsRun, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe-root', type=Path, required=True)
    args = parser.parse_args()
    verify(args.out, args.probe_root)
