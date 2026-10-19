import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v121_independent_eta as run


class IndependentEtaTests(unittest.TestCase):
    def test_new_grid_is_fixed_and_disjoint_from_development(self):
        self.assertEqual(len(run.ITEMS), 16)
        self.assertEqual(len(set(run.ITEMS)), 16)
        self.assertFalse(set(run.DAYS) & set(run.frozen.DAYS))
        self.assertEqual({run.ITEMS[i][0] for i in run.PREFLIGHT}, set(run.SEEDS))
        self.assertEqual(run.model.WINDOW_S, 3600.)

    def receipts(self):
        return [dict(seed=s, day=d, index=i, model_used_for_actions=False,
            actor_updates=0, critic_updates=0, dual_updates=0) for i, (s, d) in enumerate(run.ITEMS)]

    def test_missing_reordered_or_updated_paths_cannot_be_scored(self):
        rows = self.receipts()
        run.assert_grid(rows)
        for changed in (rows[:-1], list(reversed(rows))):
            with self.assertRaises(AssertionError):
                run.assert_grid(changed)
        changed = copy.deepcopy(rows)
        changed[0]['model_used_for_actions'] = True
        with self.assertRaises(AssertionError):
            run.assert_grid(changed)

    def test_frozen_gate_requires_both_action_cohorts_not_just_mae(self):
        stats = dict(forecasts=dict(split_ready_mae_delta_s=dict(ci95=[-2., -1.])))
        for group in ('negative', 'quota'):
            stats[group] = dict(split_sign_delta=dict(ci95=[-.05, -.01]),
                split_departure_sign_delta=dict(mean=-.01))
        self.assertTrue(run.model.passes(stats))
        stats['quota']['split_sign_delta']['ci95'][1] = .01
        self.assertFalse(run.model.passes(stats))
        stats['quota']['split_sign_delta']['ci95'][1] = -.01
        stats['negative']['split_departure_sign_delta']['mean'] = .01
        self.assertFalse(run.model.passes(stats))


if __name__ == '__main__':
    unittest.main()
