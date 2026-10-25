#!/usr/bin/env python3
"""Import/native-hook and synthetic bridge checks only: no physical episode."""

import argparse
import inspect
import json
from pathlib import Path
import unittest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    import run_freqduet_v267_bilateral_factorial as fixed
    from run_freqduet_v271_native_event_graph import SEED, DAY, FUTURE
    assert (SEED, DAY) == (fixed.SEEDS[0], fixed.DAYS[0])
    assert FUTURE == 27011001
    # Resolve the same optimized modules used by rollout, without constructing
    # a Runner, loading a checkpoint, resetting an episode or advancing traffic.
    _, Runner, _, _ = fixed.pilot.joint.optimized_runtime(a.out_root.parent)
    from env.bus import Bus
    for name in ('exchange_passengers', 'drive', '_start_dwelling', '_process_dwelling'):
        assert callable(getattr(Bus, name))
    assert 'current_time' in inspect.signature(Bus.exchange_passengers).parameters
    assert 'current_time' in inspect.signature(Bus.drive).parameters
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                               pattern='test_v271_native_event_graph.py')
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    assert tests.testsRun > 0 and tests.wasSuccessful()
    value = dict(source=a.source, status='native_interfaces_and_synthetic_bridge_passed',
        tests=tests.testsRun, runtime_runner=inspect.getfile(Runner), native_bus=inspect.getfile(Bus),
        demand_day=DAY, actor_anchor=SEED, future=FUTURE, physical_starts=0,
        training_updates=0, main_policy_changed=False)
    path = a.out_root/'checks/status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print('V271_SYNTHETIC_BRIDGE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
