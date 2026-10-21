import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import seal_freqduet_v172_temporal_intervals as seal


class TemporalIntervalTests(unittest.TestCase):
    def example(self):
        days = np.arange(1, 65)
        requests = np.array([[d, i, 11, 100., t, i, 2, 1]
            for i, d in enumerate(days) for t in (7200., 10800.)])
        vectors = np.column_stack((np.repeat(np.arange(64.), 2), np.zeros((128, 2))))
        return requests, vectors

    def test_split_key_recomputes_day_episode_query_weights(self):
        q, vectors = self.example()
        entries = [dict(key=[1, 'old'])]
        candidate, index, mass, records, error = seal.temporal_fit(entries, np.zeros(128, dtype=int),
            q, np.arange(128), vectors)
        self.assertEqual([e['key'][-1] for e in candidate], ['other', 'hour2'])
        np.testing.assert_array_equal(mass, np.full(128, 64))
        _, old_mass = seal.fit.fit_key(vectors, q[:, 0].astype(int), q[:, 1].astype(int))
        np.testing.assert_array_equal(old_mass, np.full(128, 128))
        self.assertLess(error, 1e-12)
        for entry in candidate:
            self.assertEqual((entry['qlow_s'], entry['qhigh_s']), (5., 58.))
            self.assertEqual((entry['ranks']['lower_rank'], entry['ranks']['upper_rank']), (6, 59))

    def test_day_episode_query_hierarchy_is_rebuilt_inside_stratum(self):
        q, vectors = self.example()
        q = np.vstack((q, q[0]))
        vectors = np.vstack((vectors, vectors[0]))
        _, index, mass, _, _ = seal.temporal_fit([dict(key=['key'])], np.zeros(len(q), dtype=int),
            q, np.arange(len(q)), vectors)
        self.assertEqual(mass[0], 128)
        self.assertEqual(mass[-1], 128)
        self.assertEqual(mass[1], 64)
        for key in np.unique(index):
            self.assertAlmostEqual(float((1./mass[index == key]).sum()), 1.)

    def test_strata_cannot_share_days_to_rescue_low_support(self):
        days = np.r_[np.arange(1, 19), np.arange(1, 20)]
        q = np.array([[d, i, 1, 100., 7200. if i < 18 else 46800., i, 2, 1]
            for i, d in enumerate(days)])
        entries, _, _, records, _ = seal.temporal_fit([dict(key=['key'])], np.zeros(len(q), dtype=int),
            q, np.arange(len(q)), np.zeros((len(q), 3)))
        self.assertEqual([r['days'] for r in records], [18, 19])
        self.assertEqual(entries[0]['status'], 'abstain_less_than_19_days')
        self.assertIsNone(seal.fit.interval(100., entries[0]))
        self.assertEqual(seal.fit.interval(100., entries[1]), (100., 100.))
        self.assertIsNone(seal.fit.interval(100., None))
        self.assertIsNone(seal.fit.interval(100., entries[1], native_supported=False))

    def test_bucket_preserves_joint_vectors_references_and_support(self):
        q, vectors = self.example()
        model = dict(calibration_days=list(range(1, 65)),
            frozen_protocol=dict(conditioning=dict(candidate_key=['native_stage_kind'])))
        table = dict(execution_source=seal.V167_SOURCE, actor=1, mode='old', candidate=[dict(key=['moving-launch'])],
            reference=[dict(key=['frozen'], status='finite', qlow_s=-2., qhigh_s=3.)],
            eligible_queries=128, joint_atoms=128, queries=128)
        days, eps = q[:, 0].astype(int), q[:, 1].astype(int)
        finite = np.ones(128, dtype=bool)
        masks = [(q[:, 2].astype(int) & (1 << i)) != 0 for i in range(5)]
        masks.append(masks[-1].copy())
        cells = [dict(actor=1, mode='old', **seal.plan.support_cell(d, s, mask, days, eps, finite, finite, finite))
            for d in model['calibration_days'] for s, mask in zip(seal.plan.SCOPES, masks)]
        support = dict(metadata=dict(unsupported_hour13_moving=0), cells=cells)
        records = [dict(old_key=0, stratum=s, days=64) for s in ('other', 'hour2')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bounds = np.tile(np.array([[1., 2.], [97., 102.]]), (128, 1, 1))
            np.savez_compressed(root/'parent.npz', requests=q, atom_request=np.arange(128), residual_rsd=vectors,
                candidate_key=np.zeros(128, dtype=int), reference_key=np.zeros(128, dtype=int),
                native_supported=finite, finite_candidate=finite, finite_reference=finite, bounds=bounds,
                mass_denominator=np.column_stack((np.full(128, 128), np.full(128, 777))),
                episode_keys=np.arange(64).reshape(-1, 1))
            stored = {'parent.json': table, '1_old.json': support}
            def read(path):
                return records if path.parent.name == 'key_support' else stored[path.name]
            args = SimpleNamespace(v167_root=root, v171_root=root/'protocol', out_root=root/'new')
            with patch.object(seal, 'read', side_effect=read), patch.dict('os.environ', FREQDUET_DIAGNOSTIC_COMMIT='test'):
                result = seal.build_bucket(args, model, dict(actor=1, mode='old', map_file='parent.json', atom_file='parent.npz'))
            self.assertEqual(result['reference'], table['reference'])
            self.assertEqual(result['changed_weight_atoms'], 128)
            self.assertEqual(result['cells'], cells)
            with np.load(args.out_root/result['atom_file']) as saved:
                np.testing.assert_array_equal(saved['residual_rsd'], vectors)
                np.testing.assert_array_equal(saved['mass_denominator'][:, 0], np.full(128, 64))
                np.testing.assert_array_equal(saved['mass_denominator'][:, 1], np.full(128, 777))
                np.testing.assert_array_equal(saved['bounds'][:, 1], bounds[:, 1])
                np.testing.assert_array_equal(saved['frozen_v167_bounds'], bounds[:, 0])
            self.assertEqual(json.loads((args.out_root/'maps/1_old.json').read_text())['candidate'][0]['day_count'], 64)

    def test_parent_admission_rejects_unsupported_or_modified_plan(self):
        frozen, support = dict(point='V151'), dict(minimum_calibration_days=19)
        reservation = dict(calibration_days=list(range(2951001, 2951065)), actors=[1],
            confirmation_days=list(seal.plan.CONFIRMATION),
            sampler_seeds={str(d): seal.plan.SAMPLERS[d] for d in seal.plan.CONFIRMATION})
        protocol = dict(execution_source=seal.V171_SOURCE, reservation=reservation, frozen=frozen, support=support,
            conditioning=dict(candidate_key=['key', 'query_time_stratum']),
            inference=dict(scopes=list(seal.plan.SCOPES)), execution=dict(confirmation_started=False))
        model = dict(status='sealed', execution_source=seal.V167_SOURCE, calibration_days=reservation['calibration_days'],
            actors=[1], frozen_protocol=dict(frozen=frozen, support=support, conditioning=dict(candidate_key=['key'])))
        admission = dict(status='v171_protocol_and_calibration_support_complete', execution_source=seal.V171_SOURCE,
            protocol=deepcopy(protocol), support=dict(decision='pass'))
        audit = dict(status='v166_readback_passed', audit_source='c990e3c99b', capture_source='20d914af03',
            passed_paths=1536, failed_paths=0, calibration_days=reservation['calibration_days'], counts={},
            query_metadata_complete=True, query_support_and_label_clocks_checked=True)
        seal.validate_parents(model, protocol, admission, audit)
        broken = deepcopy(admission)
        broken['support']['decision'] = 'insufficient_support'
        with self.assertRaises(AssertionError):
            seal.validate_parents(model, protocol, broken, audit)
        bad = deepcopy(audit)
        bad['counts']['eligible_unresolved'] = 1
        with self.assertRaises(AssertionError):
            seal.validate_parents(model, protocol, admission, bad)

    def test_no_collection_confirmation_or_tuning_entry(self):
        tree = ast.parse(Path(seal.__file__).read_text())
        calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'collect', 'rollout', 'train', 'stream', 'percentile', 'quantile', 'registration'}))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('v168_root', 'v169_root', 'v170_root')
            for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
