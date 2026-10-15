from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v67_action_labels as labels


class ActionLabelTest(unittest.TestCase):
    def test_fixed_design_uses_new_days_complete_action_grids(self):
        old = set(labels.expanded.TRAIN_SEEDS) | set(labels.expanded.VALIDATION_SEEDS)
        old |= set(labels.residual.VALIDATION_SEEDS) | set(labels.shrink.VALIDATION_SEEDS)
        self.assertTrue(set(labels.DEMAND_SEEDS).isdisjoint(old))
        self.assertEqual((len(labels.REFERENCES), len(labels.CONTEXTS), len(labels.BRANCHES)), (32, 96, 672))
        self.assertEqual(len({labels.reference_spec(i)["sampler_seed"] for i in range(32)}), 32)
        for context in range(96):
            self.assertEqual([a for c, a in labels.BRANCHES if c == context], list(labels.joint.ACTIONS))

    def test_override_consumes_same_private_sampling_draw(self):
        class Policy:
            def __init__(self):
                self.rng = torch.Generator().manual_seed(670)

            def get_action(self, state, deterministic):
                return labels.joint.ACTIONS[int(torch.randint(7, (), generator=self.rng))]

        native, forced = Policy(), Policy()
        first = labels.draw_action(native, np.zeros(2))
        alternate = labels.draw_action(forced, np.zeros(2), 45)
        self.assertEqual(first[0], alternate[0])
        self.assertEqual(alternate[1], 45)
        self.assertTrue(torch.equal(native.rng.get_state(), forced.rng.get_state()))
        self.assertEqual(labels.draw_action(native, np.zeros(2)), labels.draw_action(forced, np.zeros(2)))

    def test_prefix_includes_private_sampler_and_ignores_intended_execution_change(self):
        first = {field: 0 for field in labels.PREFIX_FIELDS}
        first["policy_sampling"] = torch.tensor([1, 2])
        first["encoded_state"] = np.array([.1, .2])
        labels.same_prefix(dict(first, executed_action_s=0), dict(first, executed_action_s=45))
        for field in ("policy_sampling", "upper_sampling", "scenario_streams", "lower_prefix", "exposure"):
            changed = dict(first)
            changed[field] = torch.tensor([1, 3]) if field == "policy_sampling" else 1
            with self.assertRaises(AssertionError):
                labels.same_prefix(first, changed)

    def test_action_delta_removes_state_only_offset_and_shrinkage_does_not_rerank(self):
        class Linear(torch.nn.Module):
            def forward(self, states, actions):
                return states[:, :1] + actions / 1000

        capture = dict(encoded_state=np.ones(19, dtype=np.float32), prefix_cv_squared=.08)
        models = {name: Linear() for name in ("frozen", *labels.residual.CANDIDATES)}
        payload = dict(encoder=dict(local_dim=2, buses=1), prior=100.)
        values = labels.action_predictions(capture, models, payload, dict(linear_residual=.5, mlp_residual=.8))
        for name in values:
            self.assertEqual(values[name][0], 0.)
        np.testing.assert_allclose(values["shrunk_mlp"], values["mlp_residual"] * .8)
        np.testing.assert_array_equal(np.argsort(values["shrunk_linear"]), np.argsort(values["linear_residual"]))

    def test_metrics_do_not_pad_zero_action_or_count_truth_ties_as_rank_evidence(self):
        truth = np.array([0., 0., 1., 2., 3., 4., 5.])
        result = labels.predictor_metrics(np.zeros(7), truth)
        self.assertEqual(result["delta_mae"], 2.5)
        self.assertEqual(result["rank_pairs"], 20)
        self.assertEqual(result["tied_truth_pairs"], 1)
        self.assertEqual(result["rank_correct"], 0)
        exact = labels.predictor_metrics(truth, truth)
        self.assertEqual(exact["delta_mae"], 0.)
        self.assertEqual(exact["rank_correct"], exact["rank_pairs"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
