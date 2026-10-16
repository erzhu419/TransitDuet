from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v82_local_global_regularity as audit


class LocalGlobalTest(unittest.TestCase):
    def test_arrival_variance_between_groups(self):
        events = [dict(headway_s=h, group=g) for h, g in ((100., 0), (100., 0), (300., 1), (300., 1))]
        result = audit.arrival_decomposition(events, lambda e: e["group"])
        self.assertEqual(result["within_group_cv_squared"], 0.)
        self.assertEqual(result["between_group_cv_squared"], .25)
        self.assertEqual(sum(result["group_cv_contributions"].values()), .25)

    def test_arrival_variance_within_groups(self):
        events = [dict(headway_s=h, group=g) for h, g in ((100., 0), (300., 0), (100., 1), (300., 1))]
        result = audit.arrival_decomposition(events, lambda e: e["group"])
        self.assertEqual(result["within_group_cv_squared"], .25)
        self.assertEqual(result["between_group_cv_squared"], 0.)

    def test_unresolved_forecast_does_not_invent_future_labels(self):
        context = SimpleNamespace(evidence_valid=True, follower_evidence_valid=True)
        result = audit.local_effect(context, np.zeros((7, 4)), np.full(7, 1. / 7.), np.full(7, 1. / 7.), None)
        self.assertIsNone(result["completed_ready_gain_delta"])
        self.assertIsNone(result["completed_departure_gain_delta"])

    def test_instrumentation_preserves_each_probability_law(self):
        from lower.resac_lagrangian import CategoricalPolicy

        torch.set_num_threads(1)
        policy = CategoricalPolicy(3, audit.step.ACTIONS, hidden_dim=8)
        state = torch.tensor([[.2, .4, .6]])
        context = SimpleNamespace(target_headway_s=300., forward_headway_s=150.,
            follower_departure_gap_s=450., evidence_valid=True, follower_evidence_valid=True)
        features = audit.step.action_features(context, .5, 0, 2, 60.)
        for label in ("base", "raw", "projected"):
            theta = np.zeros(4) if label == "base" else np.array([0., .2, 0., .3])
            expected = (audit.RegularityProjection()(policy, state, features, theta) if label == "projected"
                        else audit.step.probabilities(policy, state, features, theta))
            observer = audit.LocalGlobalAudit(label)
            torch.testing.assert_close(observer(policy, state, features, theta), expected, rtol=0, atol=0)
            self.assertEqual(len(observer.laws), 1)


if __name__ == "__main__":
    unittest.main()
