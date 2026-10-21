import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v170_key_mixtures as audit


class KeyMixtureTests(unittest.TestCase):
    def entry(self):
        return dict(key=[1, 'old', 'clearance', 'clearance', True, 3, 'moving',
            'moving-launch', 'not_applicable', '[0,900]'], status='finite', day_count=64, qlow_s=-10.)

    def test_original_mixture_mass_not_raw_query_counts(self):
        entries = [self.entry()]
        before = deepcopy(entries)
        m = audit.mixtures(entries, np.array([0, 0, 0]), np.array([4, 4, 2]),
            np.array([2, 2, 3]), np.array([[-40., 0., 0.], [-40., 0., 0.], [10., 0., 0.]]),
            np.array([True, True, False]))[0]
        self.assertEqual(m['hour2_mass'], .5)
        self.assertEqual(m['pooled_bias_s'], -15.)
        self.assertEqual(m['hour2_bias_s'], -40.)
        self.assertEqual(m['other_hours_bias_s'], 10.)
        self.assertEqual(m['pooled_upper_rate'], .5)
        self.assertEqual(entries, before)

    def test_key_entirely_in_hour2_preserves_absent_other_hour_mean(self):
        m = audit.mixtures([self.entry()], np.array([0]), np.array([1]), np.array([2]),
            np.zeros((1, 3)), np.array([False]))[0]
        self.assertEqual(m['hour2_mass'], 1.)
        self.assertIsNone(m['other_hours_bias_s'])

    def test_episode_and_day_mass_are_not_pooled_queries(self):
        w = audit.day_weights(np.array([10, 10, 10, 11]), np.array([1, 1, 2, 3]))
        np.testing.assert_array_equal(w, [.25, .25, .5, 1.])

    def test_event_composition_preserves_zero_event_days(self):
        entry = self.entry()
        table = dict(candidate=[entry])
        reference = [dict(key=entry['key'][:3], qlow_s=-20.)]
        cells, keys = audit.hour13(1, 'old', 'confirmation32', np.array([10, 10, 11]),
            np.array([1, 1, 2]), np.array([0, 0, 0]), np.array([100.]*3),
            np.array([[-15., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
            np.array([[[90., 110.], [80., 120.]]]*3), table, reference, [10, 11])
        self.assertEqual(cells[0]['new'], .5)
        self.assertEqual(cells[1]['new'], 0.)
        result = audit.composition(keys, [1])
        new = next(r for r in result if r['event'] == 'new' and r['field'] == 'predicted_horizon_band')
        self.assertEqual(new['mass'], .25)
        self.assertEqual(new['event_share'], 1.)
        self.assertEqual(new['rate_within_matched_category'], .25)
        self.assertEqual(new['point_s'], 100.)
        for cell in cells:
            self.assertEqual(cell['new']+cell['repaired']+cell['both']+cell['neither'], 1.)
            self.assertEqual(cell['above']-cell['reference_above'], cell['gap'])
        json.dumps(result, allow_nan=False)

    def test_category_partition_and_actor_mass(self):
        rows = []
        for actor, mass, band in ((1, .1, '[0,900]'), (2, .3, '(900,1800]')):
            key = self.entry()['key']
            key[0], key[-1] = actor, band
            rows.append(dict(cohort='confirmation32', mode='old', event='new', key=key,
                mass=mass, sums=[100.*mass]*len(audit.FEATURES)))
            rows.append(dict(cohort='confirmation32', mode='old', event='all', key=key,
                mass=1., sums=[100.]*len(audit.FEATURES)))
        result = [r for r in audit.composition(rows, [1, 2]) if r['field'] == 'predicted_horizon_band' and r['event'] == 'new']
        self.assertAlmostEqual(sum(r['mass'] for r in result), .2)
        self.assertAlmostEqual(sum(r['event_share'] for r in result), 1.)
        self.assertTrue(all(r['point_s'] == 100. for r in result))

    def test_missing_day_is_not_a_zero_event(self):
        e = self.entry()
        with self.assertRaisesRegex(AssertionError, 'coverage missing'):
            audit.hour13(1, 'old', 'confirmation32', np.array([10]), np.array([1]), np.array([0]),
                np.array([100.]), np.zeros((1, 3)), np.array([[[90., 110.], [80., 120.]]]),
                dict(candidate=[e]), [dict(key=e['key'][:3], qlow_s=-20.)], [10, 11])

    def test_no_refit_rollout_or_training_entry_calls(self):
        tree = ast.parse(Path(audit.__file__).read_text())
        calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))}
        self.assertTrue(calls.isdisjoint({'fit_key', 'inverse_day_mass', 'rollout', 'collect', 'train', 'attach_labels'}))


if __name__ == '__main__':
    unittest.main()
