import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_option_residual as experiment
from freq_hrl.rl.optional_action_residual import OptionalActionResidual
from scripts import pointmaze_option_residual_stage111_spec as spec
from scripts.submit_pointmaze_option_residual_stage111_scheduleurm import task_specification, qualification_task
from test_pointmaze_optional_plan import OptionalPlanTest
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


class OptionResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sources(self):
        models, pred, _, cal = OptionalPlanTest().sources()
        return models, pred, spec.source_record(410011), cal

    def test_initial_exact_function_for_all_advice_and_only_branch_learns(self):
        model = self.sources()[0]["50"]
        before = copy.deepcopy(model.lower_actor.state_dict())
        branch = OptionalActionResidual(model.lower_actor, feedback_dim=392, advice_dim=4)
        states = torch.randn(32, 396)
        flat = branch.flat_input(states)
        expected = model.lower_actor.distribution(flat)
        actual = branch.distribution(states)
        torch.testing.assert_close(actual.mean, expected.mean, atol=0, rtol=0)
        torch.testing.assert_close(actual.stddev, expected.stddev, atol=0, rtol=0)
        torch.manual_seed(11)
        sampled = branch.forward_with_mean(states)
        torch.manual_seed(11)
        original = model.lower_actor.forward_with_mean(flat)
        torch.testing.assert_close(sampled, original, atol=0, rtol=0)
        actual.mean.square().sum().backward()
        self.assertGreater(float(branch.readout.weight.grad.norm()), 0.)
        self.assertGreater(float(branch.readout.bias.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in branch.base.parameters()))
        with torch.no_grad():
            branch.readout.weight -= .01*branch.readout.weight.grad
        self.assertFalse(torch.equal(branch.distribution(states).mean, expected.mean))
        torch.testing.assert_close(branch.distribution(states).stddev, expected.stddev, atol=0, rtol=0)
        torch.testing.assert_close(model.lower_actor.state_dict(), before, atol=0, rtol=0)
        torch.testing.assert_close(branch.base.state_dict(), before, atol=0, rtol=0)

    def test_final_blind_donor_metadata_and_std_freeze(self):
        originals, pred, _, cal = self.sources()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"result.json"
            manifest = {"groups": {}}
            for p in spec.PERIODS:
                donor = copy.deepcopy(originals[str(p)])
                with torch.no_grad():
                    donor.lower_actor.net[-1].bias.add_(.01)
                path = experiment.source.learning.final_checkpoint(donor, output, root=410011,
                    period=p, method="blind", updates=8, protocol=spec.source)
                manifest["groups"][str(p)] = {"trained": {"blind": {"checkpoint": path}}}
            experiment.write_json(output, manifest)
            path_for = lambda r, p: Path(directory)/"final_weights"/f"period_{p}_blind.pt"
            with patch.object(spec, "source_result", return_value=output), patch.object(spec, "donor_checkpoint", side_effect=path_for), \
                    patch.object(experiment.source, "qualify", side_effect=lambda c, **kw: c), \
                    patch.object(experiment.source, "load_source", side_effect=lambda r: (copy.deepcopy(originals), pred, {}, cal)):
                loaded, _, _, _ = experiment.load_source(410011)
                torch.testing.assert_close(loaded["50"].lower_actor.net[-1].bias,
                    originals["50"].lower_actor.net[-1].bias+.01, atol=0, rtol=0)
                path = path_for(410011, 50)
                saved = torch.load(path, weights_only=False)
                saved["updates"] = 7
                torch.save(saved, path)
                with self.assertRaisesRegex(ValueError, "final Stage107"):
                    experiment.load_source(410011)
                saved["updates"] = 8
                saved["weights"]["lower_actor"]["log_std"] += .01
                torch.save(saved, path)
                with self.assertRaises(AssertionError):
                    experiment.load_source(410011)

    def test_known_local_Q_and_replication_metric(self):
        panels = {name: {v: {"suffix_return": 10.} for v in spec.VARIANTS} for name in spec.PANELS}
        for rows in panels.values():
            for i, slope in enumerate((2., -3.)):
                rows[f"axis{i}_plus"]["suffix_return"] += slope*spec.EPSILON
                rows[f"axis{i}_minus"]["suffix_return"] -= slope*spec.EPSILON
        g = experiment.gradients(panels)
        np.testing.assert_allclose(g["A"], [2., -3.], atol=1e-13)
        q = {"gradients": g}
        e = experiment.effects(50, [q, q])
        self.assertAlmostEqual(e["50/local_credit_cosine"], 1.)
        self.assertAlmostEqual(e["50/local_credit_dot"], 6.5)

    def test_paired_query_shares_prefix_state_but_not_future_noise(self):
        models, pred, _, cal = self.sources()
        args = spec.arguments(410011, preflight=True)
        args.horizon = 200
        experiment.source.native.init_worker(models["50"].config, args)
        query = spec.seed_roles(410011, preflight=True)["queries"][0]
        with patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
                patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
            row = experiment.worker_query((experiment.source.native.joint.inference_weights(models["50"]), query, 50, pred, cal["50"]["envelope"]))
        self.assertEqual((row["pairing"], row["zero_branch"], row["source_freeze"]), ("passed",)*3)
        self.assertLessEqual(row["innovation_max_error"], 3e-5)
        a, b = row["panels"]["A"]["zero"], row["panels"]["B"]["zero"]
        self.assertEqual(a["prefix_seed"], b["prefix_seed"])
        self.assertNotEqual(a["suffix_seed"], b["suffix_seed"])
        self.assertNotEqual(a["suffix_return"], b["suffix_return"])
        self.assertEqual(row["cost"]["native_episodes"], 10)
        self.assertEqual(row["cost"]["branch_zero_checks"], 3)
        self.assertEqual(row["cost"]["advice_ols_fits"], 2)

    def test_reduced_run_measured_counts_and_roster_rejection(self):
        sources = self.sources()
        arguments = spec.arguments
        short = lambda r, **kw: SimpleNamespace(**{**vars(arguments(r, **kw)), "horizon": 200})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(spec, "options", return_value={"workers": 1, "queries": 1}), \
                    patch.object(spec, "arguments", side_effect=short), patch.object(experiment, "load_source", return_value=sources), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
                    patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), \
                    patch.object(experiment.source.learning, "update_mean", side_effect=AssertionError("unexpected training")):
                cell = experiment.run(410011, preflight=True, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=True))
                summary = experiment.aggregate([cell], preflight=True)
                self.assertEqual(summary["local_credit_gate"], "mechanical_only")
                self.assertEqual(len(summary["endpoints"]), 6)
                self.assertFalse(list(Path(directory).rglob("*.pt")))
                self.assertFalse(list(Path(directory).rglob("*.npz")))
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["queries"][0]["panels"]["B"]["zero"]["suffix_seed"] += 1
                with self.assertRaisesRegex(ValueError, "suffix noise"):
                    experiment.qualify(bad, preflight=True)

    def test_gate_requires_nonzero_aligned_credit_both_periods(self):
        endpoints = {k: {"ci": [.5, 1.]} for k in spec.ENDPOINTS}
        with patch.object(experiment.statistics, "aggregate", side_effect=lambda *a, **kw: {"endpoints": copy.deepcopy(endpoints)}):
            self.assertEqual(experiment.aggregate([], preflight=False)["local_credit_gate"], "supported_both_periods")
            endpoints["50/local_credit_dot"]["ci"] = [-.1, 1.]
            self.assertEqual(experiment.aggregate([], preflight=False)["local_credit_gate"], "partial")
            endpoints["100/local_credit_cosine"]["ci"] = [-.1, 1.]
            self.assertEqual(experiment.aggregate([], preflight=False)["local_credit_gate"], "not_supported")

    def test_fresh_noise_rosters_exact_budgets_dynamic_nodes(self):
        seen = set()
        for pref in (True, False):
            for root in spec.roots(preflight=pref):
                for q in spec.seed_roles(root, preflight=pref)["queries"]:
                    seeds = [q["scenario_seed"], q["prefix_noise_seed"], *q["suffix_noise_seeds"].values()]
                    self.assertFalse(seen.intersection(seeds))
                    seen.update(seeds)
                task = task_specification("run", root, preflight=pref)
                self.assertIsNone(task["require_node"])
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(spec.budget(preflight=False)["native_episodes"]*8, 1920)
        self.assertEqual(spec.budget(preflight=False)["native_steps"]*8, 2304000)
        self.assertEqual(len(qualification_task("run", preflight=False)["wait_for_files"]), 8)
        with self.assertRaises(ValueError):
            experiment.aggregate([], preflight=False)


if __name__ == "__main__":
    unittest.main()
