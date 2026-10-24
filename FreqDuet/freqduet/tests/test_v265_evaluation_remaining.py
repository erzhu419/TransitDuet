import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import freqduet_v263_evaluation_bank as bank
import run_freqduet_v265_evaluation_remaining as runner


class Admission(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.a = NS(results_root=root, source='test', collection=root/bank.COLLECTION,
            admission=root/'v265/summary/e0_admission.json', shard=0)
        self.locks = []
        for c in range(4):
            branch = 'b2' if c == 1 else 'native'
            lock = dict(context=c, contrast_id=f'c{c}:{branch}_vs_native_b1', native_bin=1,
                chosen_branch=branch, chosen_bin=2 if c == 1 else 1,
                plan=bank.action_plan(1, list(range(7))))
            self.locks.append(lock)
            unit = bank.evaluation_unit(c, 0)
            key = bank.unit_key(unit)
            paths = {}
            for b in lock['plan']['physical']:
                index = 1 if b == 'native' else int(b[1:])
                path = dict(passed=True, plan=lock['plan'], locked_contrast=lock,
                    actual_anchor_holding_s=bank.core.ACTIONS[index], actual_execution_complete=True,
                    queries=5200, metrics=dict(trips_unlaunched=0, trips_incomplete=0,
                        passenger_person_s=1e9, cv_squared=1000),
                    delta_vs_native=dict(passenger_person_s=1e9),
                    coupling=dict(uniform_mismatches=0, common_events=5200, native_only=0, alternative_only=0))
                paths[b] = path
                runner.write(self.a.collection/'checks'/key/(b+'.json'),
                    dict(path, source=runner.COLLECTOR_SOURCE, unit=unit, branch=b, retry_authorized=False))
                runner.write(self.a.collection/'started'/key/(b+'.json'),
                    dict(source=runner.COLLECTOR_SOURCE, unit=unit, branch=b,
                        locked_contrast=lock, retry_authorized=False))
            runner.write(self.a.collection/'units'/(key+'.json'), dict(source=runner.COLLECTOR_SOURCE,
                unit=unit, passed=True, locked_contrast=lock, plan=lock['plan'], paths=paths,
                primary_branch=branch, primary_contrast_id=lock['contrast_id'],
                all_arms_retained=True, selection_recomputed=False))

    def run_audit(self):
        with patch.object(runner, 'frozen_locks', return_value=copy.deepcopy(self.locks)), redirect_stdout(io.StringIO()):
            runner.audit_e0(self.a, bank)

    def change_unit(self, change):
        path = self.a.collection/'units/c0_evaluation_f0.json'
        row = runner.read(path)
        change(row)
        runner.write(path, row)

    def test_complete_engineering_admits_even_harmful_outcomes(self):
        self.run_audit()
        gate = runner.read(self.a.admission)
        self.assertEqual(gate['physical_starts'], 28)
        self.assertFalse(gate['admission_uses_outcome_quality'])
        self.assertEqual(sorted(u for s in runner.SHARDS for u in s), list(runner.REMAINING))
        self.assertEqual([len(s) for s in runner.SHARDS], [4,4,4,4,3,3,3,3])

    def test_missing_legal_bin_rejects(self):
        self.change_unit(lambda row: row['paths'].pop('b6'))
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_uniform_mismatch_rejects(self):
        self.change_unit(lambda row: row['paths']['b2']['coupling'].update(uniform_mismatches=1))
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_changed_S_primary_rejects(self):
        self.change_unit(lambda row: row.update(primary_branch='b2'))
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_unexecuted_request_rejects(self):
        self.change_unit(lambda row: row['paths']['b2'].update(actual_execution_complete=False))
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_failed_child_is_not_retried_or_replaced(self):
        self.run_audit()
        with patch.object(runner, 'frozen_locks', return_value=self.locks), \
                patch.object(runner.subprocess, 'run', return_value=NS(returncode=1)) as calls, \
                redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            runner.collect_shard(self.a, bank, Path('/frozen_E'))
        self.assertEqual(calls.call_count, 4)
        units = []
        for call in calls.call_args_list:
            command = call.args[0]
            units.append((int(command[command.index('--context')+1]), int(command[command.index('--future')+1])))
            self.assertEqual(command[command.index('--source')+1], runner.COLLECTOR_SOURCE)
        self.assertEqual(units, list(runner.SHARDS[0]))
        saved = runner.read(self.a.admission.parent.parent/'shards/shard0.json')
        self.assertEqual(saved['status'], 'failed_units_retained')
        self.assertFalse(saved['automatic_retry'])


if __name__ == '__main__':
    unittest.main()
