from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import seal_freqduet_v167_joint_intervals as seal


class FreshJointIntervalTests(unittest.TestCase):
    def admission(self):
        reservation = dict(calibration_days=list(seal.entry.fresh.CALIBRATION), actors=[1], modes=['old', 'handoff'])
        audit = dict(status='v166_readback_passed', audit_source=seal.AUDIT_SOURCE,
            capture_source=seal.CAPTURE_SOURCE, protocol_source=seal.PROTOCOL_SOURCE,
            expected_paths=1536, passed_paths=1536, failed_paths=0, counts=dict(eligible_unresolved=0),
            calibration_input_eligible=True, query_metadata_complete=True, query_support_and_label_clocks_checked=True,
            interval_fit=False, confirmation_started=False, confirmation_outcomes_read=False, confirmation_paths=0,
            **deepcopy(reservation))
        return dict(reservation=reservation), audit

    def test_only_new_64_day_admitted_calibration_can_be_used(self):
        plan, audit = self.admission()
        seal.validate_admission(plan, audit)
        for field, value in (('calibration_days', list(range(2851001, 2851033))),
                ('audit_source', 'older'), ('passed_paths', 1535), ('confirmation_started', True),
                ('query_metadata_complete', False), ('calibration_input_eligible', False)):
            broken = deepcopy(audit)
            broken[field] = value
            with self.assertRaises(AssertionError):
                seal.validate_admission(plan, broken)
        audit['counts']['eligible_unresolved'] = 1
        with self.assertRaisesRegex(AssertionError, 'labels'):
            seal.validate_admission(plan, audit)

    def test_64_day_ranks_keep_fixed_tail_and_observed_cdf(self):
        values = np.arange(64.)
        vectors = np.column_stack((values, -values, values))
        entry, denominator = seal.fit.fit_key(vectors, values.astype(int), values.astype(int))
        self.assertEqual((entry['ranks']['lower_rank'], entry['ranks']['upper_rank']), (6, 59))
        self.assertEqual((entry['qlow_s'], entry['qhigh_s']), (5., 58.))
        np.testing.assert_array_equal(denominator, np.full(64, 64))
        self.assertEqual(seal.fit.interval(100., entry), (42., 95.))
        self.assertIsNone(seal.fit.interval(100., None))

    def test_full_bucket_preserves_joint_atoms_weights_and_abstention_masks(self):
        days = list(seal.entry.fresh.CALIBRATION)
        plan = dict(reservation=dict(calibration_days=days,
            sampler_seeds={str(d): seal.entry.fresh.SAMPLERS[d] for d in days}),
            comparator=dict(key=['actor', 'mode', 'query_phase']))
        stored, streams = {}, {}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(capture_root=Path(directory)/'capture', out_root=Path(directory)/'seal')
            for i, day in enumerate(days):
                name, queries, residuals = f'1_{day}_old', [], []
                for j in range(4):
                    eligible, native = j < 3, j == 0 or (j == 1 and i < 18)
                    error = float(i) if j == 0 else 0.
                    queries.append(dict(actor=1, day=day, mode='old', query_key=[10+j, 2+j, True],
                        query_time_s=7200., point_s=100., denominator_eligible=eligible, native_supported=native,
                        episode=[10+j, 1, 2+j, True], candidate_key=[1, 'old', j] if native else None,
                        condition_values=dict(actor=1, mode='old', query_phase='active'), observable=dict(hour=2)))
                    residuals.append(dict(query_key=queries[-1]['query_key'], query_time_s=7200.,
                        label=dict(status='resolved' if eligible else 'no_valid_follower_evidence',
                            full_eta_s=100.-error if eligible else None),
                        residual_s=dict(running=float(i), service=-float(i), residence=error) if native else None))
                stored[name+'.json'] = dict(execution_source=seal.CAPTURE_SOURCE, protocol_source=seal.PROTOCOL_SOURCE,
                    seed=1, day=day, mode='old', sampler_seed=seal.entry.fresh.SAMPLERS[day],
                    query_record_count=4, counts=dict(queries=4, denominator_eligible=3, native_supported=2 if i < 18 else 1))
                streams['queries', name+'.jsonl.gz'] = queries
                streams['residuals', name+'.jsonl.gz'] = residuals
            with patch.object(seal, 'read', side_effect=lambda p: stored[p.name]), \
                    patch.object(seal.fit, 'stream', side_effect=lambda p: iter(streams[p.parent.name, p.name])), \
                    patch.dict(os.environ, FREQDUET_DIAGNOSTIC_COMMIT='test'):
                result = seal.build_bucket(args, plan, 1, 'old')
            common, rare = result['candidate']
            self.assertEqual((common['day_count'], common['qlow_s'], common['qhigh_s']), (64, 5., 58.))
            self.assertEqual((rare['day_count'], rare['status']), (18, 'abstain_less_than_19_days'))
            self.assertEqual((result['queries'], result['eligible_queries'], result['joint_atoms'],
                result['finite_queries'], result['native_abstentions'], result['key_abstentions']), (256, 192, 82, 64, 110, 18))
            with np.load(args.out_root/result['atom_file']) as atoms:
                self.assertEqual(int(atoms['finite_candidate'].sum()), 64)
                self.assertEqual(int(atoms['finite_reference'].sum()), 82)
                self.assertTrue(np.isnan(atoms['bounds'][~atoms['finite_candidate'], 0]).all())
                np.testing.assert_array_equal(atoms['residual_rsd'][:, 0], -atoms['residual_rsd'][:, 1])
                common_atoms = atoms['candidate_key'] == 0
                np.testing.assert_array_equal(atoms['mass_denominator'][common_atoms, 0], np.full(64, 64))
                np.testing.assert_array_equal(atoms['mass_denominator'][~common_atoms, 0], np.full(18, 18))
                self.assertTrue((atoms['mass_denominator'][:2, 1] == 128).all())

    def test_64_day_crossed_support_requires_every_cell_and_all_scopes(self):
        days = list(seal.entry.fresh.CALIBRATION)
        rows = [dict(actor=a, day=d, mode=m, scope=s, finite_fraction=.8,
            eligible_episodes=10, finite_episodes=8, eligible_queries=20, finite_queries=16,
            native_abstentions=2, key_abstentions=2, missing_reference=0)
            for a in (1, 2) for d in days for m in seal.fit.protocol.MODES for s in seal.fit.SCOPES]
        self.assertEqual(seal.fit.release(rows, [1, 2], days)['decision'], 'pass')
        rows[-1]['finite_fraction'] = .7
        self.assertIn('handoff/hour13', seal.fit.release(rows, [1, 2], days)['failed'])
        rows[-1]['finite_fraction'] = .8
        rows[-1]['finite_episodes'] = 0
        self.assertIn('handoff/hour13', seal.fit.release(rows, [1, 2], days)['failed'])


if __name__ == '__main__':
    unittest.main()
