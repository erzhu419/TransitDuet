import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import run_freqduet_v277_arrival_metric as trial


class MatrixTests(unittest.TestCase):
    def test_nested_matrix_uses_outer_root_for_frozen_runtime_proof(self):
        args = SimpleNamespace(root=Path('/results/V277/matrix_repaired'))
        self.assertEqual(trial.rollout_root(args),Path('/results/V277'))
        self.assertEqual(trial.rollout_root(args).parent/'runtime_optimization_20261014',
                         Path('/results/runtime_optimization_20261014'))

    def test_twenty_paths_eight_shards_without_duplicate_futures(self):
        indices = [i for shard in range(8) for i in range(shard,20,8)]
        self.assertEqual(sorted(indices),list(range(20)))
        self.assertEqual(len(set(trial.ITEMS)),20)
        self.assertEqual(len(trial.DAYS),4)
        self.assertEqual(len(trial.ARMS),5)

    def test_frozen_arms_do_not_inherit_new_control_modules(self):
        with TemporaryDirectory() as tmp:
            env = Path(tmp)/'env.json'
            env.write_text(json.dumps(dict(PYTHONPATH='/frozen/4a/scripts:/native/runtime')))
            args = SimpleNamespace(env_file=env,results_root=Path('/results'))
            self.assertEqual(trial.arm_env(args,trial.ARMS[2])['PYTHONPATH'],
                             '/frozen/4a/scripts:/native/runtime')
            self.assertTrue(trial.arm_env(args,trial.ARMS[1])['PYTHONPATH'].startswith(
                '/results/protocol_v6_v275_flow_controller/source_894ad14707/'))
            self.assertTrue(trial.arm_env(args,trial.ARMS[3])['PYTHONPATH'].endswith(
                ':/frozen/4a/scripts:/native/runtime'))

    def stats(self):
        return {m:dict(ci95=[-2.,-1.] if not m.startswith('trips_') else [0.,0.],
            by_day={str(d):0. for d in trial.DAYS}) for m in trial.METRICS}

    def test_historical_gates_and_each_day_trip_nonharm_are_preserved(self):
        stats = self.stats()
        self.assertTrue(all(trial.paired_gate(stats,base) for base in trial.ARMS[:3]))
        stats['cv_squared']['ci95'] = [-.01,.001]
        self.assertFalse(any(trial.paired_gate(stats,base) for base in trial.ARMS[:3]))
        stats = self.stats()
        stats['passenger_person_s']['ci95'] = [-1.,0.]
        self.assertFalse(trial.paired_gate(stats,trial.ARMS[0]))
        self.assertTrue(trial.paired_gate(stats,trial.ARMS[1]))
        stats = self.stats()
        stats['trips_incomplete']['by_day'][str(trial.DAYS[0])] = 1.
        self.assertFalse(any(trial.paired_gate(stats,base) for base in trial.ARMS[:3]))


if __name__ == '__main__':
    unittest.main()
