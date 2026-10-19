import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from run_freqduet_v106_service_tradeoff import rule_action
import run_freqduet_v124_eta_factorial as experiment


class FactorialTests(unittest.TestCase):
    def context(self, **kw):
        return SimpleNamespace(**dict(dict(evidence_valid=True, follower_evidence_valid=True,
            target_headway_s=300., forward_headway_s=200., follower_departure_gap_s=400.), **kw))

    def test_raw_and_fallback_match_original_rule_with_masks(self):
        for context in (self.context(), self.context(evidence_valid=False),
                self.context(follower_evidence_valid=False), self.context(follower_departure_gap_s=100.)):
            for feasible in ([True]*7, [True, True, False, False, True, False, False]):
                self.assertEqual(experiment.bilateral_action(context, dict(component_used=False), feasible),
                    rule_action(context, 'bilateral', .25, feasible))

    def test_split_replaces_gap_without_innovation_or_quota(self):
        mask = [True]*7
        context = self.context()
        self.assertEqual(experiment.prior.ACTIONS[experiment.bilateral_action(context,
            dict(component_used=True, gap_s=440.), mask)], 30.)
        self.assertEqual(experiment.prior.ACTIONS[experiment.bilateral_action(context,
            dict(component_used=True, gap_s=100.), mask)], 0.)
        self.assertEqual(experiment.prior.ACTIONS[experiment.bilateral_action(context,
            dict(component_used=True, gap_s=300.), mask)], 10.)

    def test_shorter_tie_and_cap_keep_original_bins(self):
        mask = [True]*7
        self.assertEqual(experiment.prior.ACTIONS[experiment.bilateral_action(self.context(),
            dict(component_used=True, gap_s=300.), mask)], 10.)
        self.assertEqual(experiment.prior.ACTIONS[experiment.bilateral_action(self.context(),
            dict(component_used=True, gap_s=1000.), mask)], 45.)

    def test_factorial_interaction_direction(self):
        values = {experiment.prior.BILATERAL: 10., experiment.prior.RAW: 13.,
            experiment.PRIMARY: 8., experiment.prior.PRIMARY: 10.}
        levels = {m: {k: v for k in experiment.prior.METRICS} for m, v in values.items()}
        effects = experiment.factorial_effects(levels)
        for metric in experiment.prior.METRICS:
            self.assertEqual(effects[experiment.INTERACTION][metric], -1.)
            self.assertEqual(effects[experiment.PRIMARY+'_vs_'+experiment.prior.BILATERAL][metric], -2.)


if __name__ == '__main__':
    unittest.main()
