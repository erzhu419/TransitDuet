import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_freqduet_v107_rule_confirmation as run


class ConfirmationTests(unittest.TestCase):
    def stats(self):
        return {k: dict(mean=-1., ci95=[-2., -0.1]) for k in
                ('passenger_person_s', 'cv_squared', 'readiness_delay_s', 'trips_unlaunched', 'trips_incomplete')}

    def test_grid_excludes_development_actors_and_days(self):
        self.assertEqual(len(run.SEEDS), 10)
        self.assertEqual(len(run.DAYS), 16)
        self.assertFalse(set(run.SEEDS) & set(run.prior.SEEDS))
        self.assertEqual(set(run.SEEDS), set(run.confirmation.TRAIN_SEEDS) - set(run.prior.SEEDS))
        self.assertFalse(set(run.DAYS) & set(run.prior.DAYS))
        self.assertEqual(len(run.ITEMS), 1440)
        self.assertEqual(len(set(run.ITEMS)), 1440)
        self.assertEqual(set(run.PRIMARY), {'forward025', 'bilateral025'})
        for method in run.PRIMARY:
            self.assertEqual(run.prior.RULES[method][1], .25)

    def test_stochastic_means_and_shared_deterministic_controls(self):
        for method in run.METHODS:
            rows = [r for r in run.ITEMS if r[2] == method]
            self.assertEqual(len(rows), 480 if method in ('native', 'preserve') else 160)
        self.assertEqual(run.CONTRASTS['forward025_vs_bilateral025'], ('forward025', 'bilateral025'))

    def test_zero_margin_rejects_any_positive_cv_upper_bound(self):
        stats = self.stats()
        self.assertTrue(run.joint_gate(stats))
        stats['cv_squared']['ci95'][1] = 1e-12
        self.assertFalse(run.joint_gate(stats))

    def test_passenger_benefit_is_strict_and_completion_is_required(self):
        stats = self.stats()
        stats['passenger_person_s']['ci95'][1] = 0.
        self.assertFalse(run.joint_gate(stats))
        stats = self.stats()
        stats['trips_incomplete']['ci95'][1] = 1.
        self.assertFalse(run.joint_gate(stats))
        stats['trips_incomplete']['ci95'][1] = 0.
        self.assertTrue(run.joint_gate(stats))

    def test_co_primary_rules_need_both_controls_without_winner_selection(self):
        stats = {m + '_vs_' + b: self.stats() for m, b in itertools.product(run.PRIMARY, ('native', 'preserve'))}
        stats['forward025_vs_preserve']['readiness_delay_s']['ci95'][1] = .01
        result = run.confirmation_gates(stats)
        self.assertFalse(result['forward025']['joint_both_controls'])
        self.assertTrue(result['forward025']['vs_native'])
        self.assertTrue(result['bilateral025']['joint_both_controls'])
        self.assertEqual(set(result), set(run.PRIMARY))

    def test_aggregate_averages_samplers_before_bootstrap_and_keeps_direct_pair(self):
        seeds, days = (1, 2), (3, 4)
        items = [(s, d, m, r) for s, d, m in itertools.product(seeds, days, run.METHODS)
                 for r in (range(3) if m in ('native', 'preserve') else (0,))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pairs').mkdir()
            for index, (seed, day, method, rep) in enumerate(items):
                value = 100. + run.METHODS.index(method) + rep
                row = dict(seed=seed, day=day, method=method, rep=rep,
                    results={k: value for k in run.METRICS}, contract=dict(day=day),
                    coverage=dict(decisions=1, missing_forward=0, bilateral=0, forward_fallback=0))
                (root / 'pairs' / f'p{index}.json').write_text(json.dumps(row))
            with patch.multiple(run, SEEDS=seeds, DAYS=days, ITEMS=items), \
                    patch.object(run.pilot, 'write_json') as write, patch('builtins.print'):
                run.aggregate(root)
            result = write.call_args.args[1]
        self.assertEqual(result['absolute']['native']['passenger_person_s']['mean'], 101.)
        self.assertEqual(result['effects']['zero_vs_preserve']['passenger_person_s']['mean'], 0.)
        self.assertEqual(result['effects']['forward025_vs_bilateral025']['passenger_person_s']['mean'], -1.)
        self.assertEqual(result['absolute']['zero']['cv_squared']['n_demand_days'], 2)
        self.assertEqual(result['new_rollouts'], 36)
        self.assertFalse(result['main_policy_changed'])


if __name__ == '__main__':
    unittest.main()
