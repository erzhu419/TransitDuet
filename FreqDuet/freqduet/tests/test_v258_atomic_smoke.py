import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from freqduet_v257_atomic_smoke import smoke_plan
import run_freqduet_v258_atomic_smoke as driver


class Slots(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/driver.COLLECTION
        self.source = 'synthetic'

    def pass_branch(self, branch, plan=None):
        driver.write(driver.check_file(self.root,branch),dict(source=self.source,branch=branch,
            passed=True,plan=plan or smoke_plan(1,[0,1,2,3,4,5,6])))

    def test_native_debit_and_no_retry(self):
        driver.claim_slot(self.root,'native',self.source)
        with self.assertRaises(AssertionError):
            driver.claim_slot(self.root,'native',self.source)
        self.assertEqual(len(list((self.root/'started').glob('*.json'))),1)

    def test_failed_native_stops_all_remaining(self):
        driver.claim_slot(self.root,'native',self.source)
        driver.write(driver.check_file(self.root,'native'),dict(source=self.source,branch='native',passed=False))
        with self.assertRaises(AssertionError):
            driver.claim_slot(self.root,'forced_native',self.source)
        self.assertEqual(len(list((self.root/'started').glob('*.json'))),1)

    def test_alias_and_mask_never_create_a_slot(self):
        driver.claim_slot(self.root,'native',self.source)
        self.pass_branch('native',smoke_plan(1,[0,1,3]))
        for b in ('b1','b2','b4','b5','b6'):
            with self.subTest(b=b),self.assertRaises(AssertionError):
                driver.claim_slot(self.root,b,self.source)
        self.assertEqual(len(list((self.root/'started').glob('*.json'))),1)

    def test_forced_native_precedes_fixed_bins_and_failure_does_not_expand(self):
        driver.claim_slot(self.root,'native',self.source)
        self.pass_branch('native')
        with self.assertRaises(FileNotFoundError):
            driver.claim_slot(self.root,'b0',self.source)
        driver.claim_slot(self.root,'forced_native',self.source)
        driver.write(driver.check_file(self.root,'forced_native'),dict(source=self.source,
            branch='forced_native',passed=False))
        with self.assertRaises(AssertionError):
            driver.claim_slot(self.root,'b0',self.source)
        self.assertEqual(len(list((self.root/'started').glob('*.json'))),2)

    def test_all_bins_use_exactly_eight_starts(self):
        plan = smoke_plan(1,list(range(7)))
        for b in plan['physical']:
            driver.claim_slot(self.root,b,self.source)
            self.pass_branch(b,plan)
        markers = list((self.root/'started').glob('*.json'))
        self.assertEqual(len(markers),8)
        self.assertTrue(all(json.loads(p.read_text())['future_root']==3540999999 for p in markers))
        with self.assertRaises(AssertionError):
            driver.claim_slot(self.root,'b6',self.source)

    def test_record_roundtrip_keeps_arrays_and_tuple_keys(self):
        import numpy as np
        value = dict(prefix={(1,2,False):np.arange(4)},native_bin=1)
        path = self.root/'records/example.pkl.gz'
        driver.save_record(path,value)
        loaded = driver.load_record(path)
        self.assertEqual(loaded['native_bin'],1)
        np.testing.assert_array_equal(loaded['prefix'][(1,2,False)],value['prefix'][(1,2,False)])


class Authority(unittest.TestCase):
    def values(self):
        fixed = dict(protocol_version='v6',clock=dict(end_s=50000),fleet_target=12,
            physical_vehicle_count=12,fleet_inventory_mode='finite',dt_s=1.)
        old = dict(physical_contract=dict(fixed,scenario_tape_id='v1:2151001',passengers_generated=1000))
        data = dict(contract=dict(fixed,scenario_tape_id='v1:3540999999',passengers_generated=1003),
            anchor=dict(scenario=dict(seed=2151001,version='v1')),future=dict(root=3540999999))
        return data,old

    def check(self,data,old):
        return driver.archived_authority(data,old,lambda a,b,label:self.assertEqual(a,b,label),
            lambda seed,version:SimpleNamespace(identifier=f'{version}:{seed}'))

    def test_new_future_identity_and_population_are_not_old_endpoint_equalities(self):
        result = self.check(*self.values())
        self.assertTrue(result['fixed_authority_exact'])
        self.assertEqual(result['new_future_passengers_generated'],1003)

    def test_fixed_authority_changes_are_still_rejected(self):
        data,old = self.values()
        data['contract']['physical_vehicle_count'] = 13
        with self.assertRaises(AssertionError):
            self.check(data,old)

    def test_wrong_future_identity_is_rejected(self):
        data,old = self.values()
        data['contract']['scenario_tape_id'] = 'v1:2151001'
        with self.assertRaises(AssertionError):
            self.check(data,old)


if __name__ == '__main__':
    unittest.main()
