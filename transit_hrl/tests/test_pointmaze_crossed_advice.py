import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_crossed_advice as experiment
from scripts import pointmaze_crossed_advice_stage108_spec as spec
from scripts.submit_pointmaze_crossed_advice_stage108_scheduleurm import task_specification, qualification_task
from test_pointmaze_optional_plan import OptionalPlanTest
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class CrossedAdviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sources(self):
        originals, pred, _, cal = OptionalPlanTest().sources()
        models = {p: {m: copy.deepcopy(model) for m in spec.LOWERS} for p, model in originals.items()}
        for group in models.values():
            for model in group.values():
                with torch.no_grad():
                    model.lower_actor.net[0].weight[:, 392:].fill_(.02)
                    model.upper_actor.net[-1].bias.fill_(.15)
        return models, pred, spec.source_record(410011), cal

    def test_zero_mean_changes_only_final_upper_mean_layer_and_preserves_sources(self):
        model = self.sources()[0]["50"]["learned_hint"]
        before = copy.deepcopy(model.state_dict())
        w = experiment.execution_weights(model, "noise")
        probe = copy.deepcopy(model)
        probe.load_state_dict(w)
        states = torch.randn(16, model.config.upper_state_dim)
        dist = probe.upper_actor.distribution(states)
        torch.testing.assert_close(dist.mean, torch.zeros_like(dist.mean), atol=0, rtol=0)
        torch.testing.assert_close(dist.stddev, model.upper_actor.distribution(states).stddev, atol=0, rtol=0)
        torch.testing.assert_close(w["lower_actor"], before["lower_actor"], atol=0, rtol=0)
        torch.testing.assert_close(w["upper_value"], before["upper_value"], atol=0, rtol=0)
        experiment.source.native.curves.support.assert_frozen(model, before)
        for plan in ("learned", "forecast", "blind"):
            torch.testing.assert_close(experiment.execution_weights(model, plan),
                experiment.source.native.joint.inference_weights(model), atol=0, rtol=0)

    def test_all_eight_variants_execute_same_lower_and_pair_upper_noise(self):
        models, pred, _, cal = self.sources()
        args = spec.arguments(410011, preflight=True)
        args.horizon = 100
        experiment.source.native.init_worker(models["50"]["learned_hint"].config, args)
        evaluation = {}
        with patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
            for variant, (lower, plan) in spec.VARIANTS.items():
                row = experiment.worker_episode((experiment.execution_weights(models["50"][lower], plan),
                    108095001, variant, 50, pred, .02, cal["50"]["envelope"]))
                experiment.check_row(row, 50, 100)
                self.assertEqual(row["upper_calls"], 2 if plan in ("learned", "noise") else 0)
                self.assertEqual(row["reference_evaluations"], 0 if plan == "blind" else 100)
                evaluation[variant] = [row]
        effects = experiment.paired_effects(50, evaluation, [108095001], 410011)
        self.assertEqual(len(effects), 10)
        bad = copy.deepcopy(evaluation)
        bad["learned_hint_noise"][0]["upper_standard_noise"][0][0] += .01
        with self.assertRaises(AssertionError):
            experiment.paired_effects(50, bad, [108095001], 410011)

    def test_final_donor_loader_rejects_metadata_and_std_changes(self):
        models, pred, _, cal = self.sources()
        originals = {p: copy.deepcopy(group["learned_hint"]) for p, group in models.items()}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"result.json"
            manifest = {"groups": {}}
            for p in spec.PERIODS:
                manifest["groups"][str(p)] = {"trained": {}}
                for method in spec.LOWERS:
                    path = experiment.source.learning.final_checkpoint(models[str(p)][method], output,
                        root=410011, period=p, method=method, updates=8, protocol=spec.source)
                    manifest["groups"][str(p)]["trained"][method] = {"checkpoint": path}
            experiment.write_json(output, manifest)
            donor = lambda r, p, m: Path(directory)/"final_weights"/f"period_{p}_{m}.pt"
            with patch.object(spec, "source_result", return_value=output), patch.object(spec, "donor_checkpoint", side_effect=donor), \
                    patch.object(experiment.source, "qualify", side_effect=lambda c, **kw: c), \
                    patch.object(experiment.source, "load_source", return_value=(originals, pred, {}, cal)):
                loaded, _, record, _ = experiment.load_source(410011)
                self.assertEqual(record["donor_protocol"], spec.source.EXPERIMENT_PROTOCOL)
                torch.testing.assert_close(experiment.source.native.joint.inference_weights(loaded["100"]["forecast_hint"]),
                    experiment.source.native.joint.inference_weights(models["100"]["forecast_hint"]), atol=0, rtol=0)
                path = donor(410011, 50, "learned_hint")
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

    def test_reduced_run_has_exact_counters_no_training_or_new_artifacts(self):
        sources = self.sources()
        before = {p: {m: copy.deepcopy(model.state_dict()) for m, model in group.items()} for p, group in sources[0].items()}
        args = spec.arguments
        short = lambda r, **kw: SimpleNamespace(**{**vars(args(r, **kw)), "horizon": 100})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(spec, "options", return_value={"workers": 1, "evaluation_episodes": 1}), \
                    patch.object(spec, "arguments", side_effect=short), patch.object(experiment, "load_source", return_value=sources), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), \
                    patch.object(experiment.source.learning, "update_mean", side_effect=AssertionError("unexpected training")), \
                    patch.object(experiment.source.learning, "final_checkpoint", side_effect=AssertionError("unexpected checkpoint")):
                cell = experiment.run(410011, preflight=False, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=False))
                self.assertEqual(cell["native_planning_cost"], spec.planning_budget(preflight=False))
                self.assertEqual(len(list(Path(directory).rglob("*.pt"))), 0)
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["evaluation"]["forecast_hint_blind"][0]["upper_calls"] = 1
                with self.assertRaisesRegex(ValueError, "inference"):
                    experiment.qualify(bad, preflight=False)
        for p, group in sources[0].items():
            for m, model in group.items():
                experiment.source.native.curves.support.assert_frozen(model, before[p][m])

    def test_fresh_rosters_all20_family_and_dynamic_scheduler(self):
        seen = set()
        for pref in (True, False):
            for root in spec.roots(preflight=pref):
                seeds = set(spec.seed_roles(root, preflight=pref)["native_evaluation"])
                self.assertFalse(seen & seeds)
                self.assertFalse(seeds & set(spec.source.seed_roles(root, preflight=pref)["native_evaluation"]))
                seen |= seeds
        rows = [{"root":r,"cost":spec.budget(preflight=False),"groups":{"50":{"effects":
            {k:1. for k in spec.ENDPOINTS}}}} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c):
            summary = experiment.aggregate(rows, preflight=False)
            self.assertEqual(summary["learned_residual_confirmation"], "supported")
            self.assertEqual(len(summary["endpoints"]), 20)
            for row in rows:
                row["groups"]["50"]["effects"][spec.PRIMARY_ENDPOINTS[0]] = -1.
            self.assertEqual(experiment.aggregate(rows, preflight=False)["learned_residual_confirmation"], "not_supported")
            with self.assertRaises(ValueError):
                experiment.aggregate(rows[:-1], preflight=False)
        full = task_specification("run", 410011, preflight=False)
        self.assertEqual((full["cpu"], full["ram_mb"], full["require_node"]), (5, 4096, None))
        self.assertEqual(full["allowed_nodes"], [f"node{i:03}" for i in range(1,7)])
        self.assertIn(spec.RUNNER_SCRIPT, full["cmd"])
        self.assertEqual(len(qualification_task("run", preflight=False)["wait_for_files"]), 8)
        self.assertEqual(spec.budget(preflight=False)["native_episodes"]*8, 4096)
        self.assertEqual(spec.budget(preflight=False)["native_steps"]*8, 4915200)


if __name__ == "__main__":
    unittest.main()
