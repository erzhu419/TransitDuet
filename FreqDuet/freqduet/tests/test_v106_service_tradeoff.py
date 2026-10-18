from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_freqduet_v106_service_tradeoff as run


class TradeoffTests(unittest.TestCase):
    def context(self, **overrides):
        values = dict(evidence_valid=True, follower_evidence_valid=True,
                      forward_headway_s=300., follower_departure_gap_s=420., target_headway_s=360.)
        return SimpleNamespace(**(values | overrides))

    def action(self, context, kind='forward', gain=1., feasible=None):
        mask = np.ones(7, bool) if feasible is None else feasible
        return run.pilot.step.ACTIONS[run.rule_action(context, kind, gain, mask)]

    def test_forward_uses_gain_and_same_command_cap(self):
        self.assertEqual(self.action(self.context(), gain=.25), 15.)
        self.assertEqual(self.action(self.context()), 45.)
        self.assertEqual(self.action(self.context(forward_headway_s=400.)), 0.)

    def test_bilateral_uses_observable_follower_and_falls_back_when_missing(self):
        context = self.context(forward_headway_s=350., follower_departure_gap_s=390.)
        self.assertEqual(self.action(context, 'bilateral'), 20.)
        context.follower_evidence_valid = False
        context.follower_departure_gap_s = None
        self.assertEqual(self.action(context, 'bilateral'), 10.)

    def test_missing_forward_is_zero_and_mask_is_respected(self):
        self.assertEqual(self.action(self.context(evidence_valid=False)), 0.)
        self.assertEqual(self.action(self.context(), feasible=np.asarray([True] + [False] * 6)), 0.)

    def test_quantization_ties_choose_shorter_bin(self):
        self.assertEqual(self.action(self.context(forward_headway_s=347.5)), 10.)

    def test_grid_keeps_deterministic_paths_shared_and_stochastic_repeats_separate(self):
        self.assertEqual(len(run.ITEMS), 104)
        self.assertEqual(len(set(run.ITEMS)), 104)
        for seed in run.SEEDS:
            for day in run.DAYS:
                for method in run.METHODS:
                    repeats = [r for s, d, m, r in run.ITEMS if (s, d, m) == (seed, day, method)]
                    self.assertEqual(repeats, [0, 1, 2] if method in ('native', 'preserve') else [0])

    def test_summary_averages_samplers_before_actor_day_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pairs').mkdir()
            for index, (seed, day, method, rep) in enumerate(run.ITEMS):
                value = 100. + run.METHODS.index(method) + rep
                row = dict(seed=seed, day=day, method=method, rep=rep,
                    results={k: value for k in run.METRICS}, contract=dict(day=day),
                    coverage=dict(decisions=1, missing_forward=0, bilateral=0, forward_fallback=0))
                (root / 'pairs' / f'p{index}.json').write_text(json.dumps(row))
            with patch.object(run.pilot, 'write_json') as write:
                run.aggregate(root)
            result = write.call_args.args[1]
        self.assertEqual(result['absolute']['native']['passenger_person_s'], 101.)
        self.assertEqual(result['absolute']['preserve']['passenger_person_s'], 102.)
        self.assertEqual(result['effects']['zero_vs_preserve']['passenger_person_s']['mean'], 0.)
        self.assertEqual(result['effects']['zero_vs_preserve']['passenger_person_s']['n_demand_days'], 4)
        self.assertEqual(result['new_rollouts'], 104)
        self.assertFalse(result['main_policy_changed'])


if __name__ == '__main__':
    unittest.main()
