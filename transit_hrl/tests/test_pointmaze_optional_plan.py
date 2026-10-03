import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_optional_plan as experiment
from scripts import pointmaze_optional_plan_stage107_spec as spec
from scripts.submit_pointmaze_optional_plan_stage107_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as fixture
from test_pointmaze_joint_conditioned import all_seeds
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool
from test_pointmaze_upper_paths import predictor


class OptionalPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def originals(self):
        return fixture.FeasibleCreditTest().source()

    def sources(self):
        source = self.originals()
        weights = experiment.native.joint.inference_weights(source)
        model = experiment.expand_model(source, weights, weights)
        envelope = {"velocity_speed_q99": 1., "axis_min": [-1., -1.], "axis_max": [1., 1.]}
        return ({str(p): copy.deepcopy(model) for p in spec.PERIODS}, predictor(), spec.source_record(410011),
            {str(p): {"alpha": .02, "envelope": envelope} for p in spec.PERIODS})

    def test_padding_retains_strong_flat_function_and_freezes_all_donors(self):
        source = self.originals()
        before = copy.deepcopy(source.state_dict())
        weights = experiment.native.joint.inference_weights(source)
        expanded = experiment.expand_model(source, weights, weights)
        generator = np.random.default_rng(7)
        states = torch.tensor(generator.normal(size=(32, 392)), dtype=torch.float32)
        hints = torch.tensor(generator.normal(size=(32, 4)), dtype=torch.float32)
        clocks = torch.tensor(generator.normal(size=(32, 2)), dtype=torch.float32)
        with torch.no_grad():
            a = source.lower_actor.distribution(states)
            b = expanded.lower_actor.distribution(torch.cat((states, hints), 1))
            torch.testing.assert_close(a.mean, b.mean, atol=1e-6, rtol=0)
            torch.testing.assert_close(a.stddev, b.stddev, atol=0, rtol=0)
            torch.testing.assert_close(source.lower_value(torch.cat((states, clocks), 1)),
                expanded.lower_value(torch.cat((states, hints, clocks), 1)), atol=1e-6, rtol=0)
        self.assertEqual((expanded.config.lower_state_dim, expanded.config.lower_value_state_dim), (396, 398))
        self.assertTrue(torch.count_nonzero(expanded.lower_actor.net[0].weight[:, 392:]) == 0)
        experiment.native.curves.support.assert_frozen(source, before)

    def test_loads_named_final_donors_and_rejects_std_changes(self):
        source = self.originals()
        originals = {str(p): copy.deepcopy(source) for p in spec.PERIODS}
        flat, upper = copy.deepcopy(source), copy.deepcopy(source)
        with torch.no_grad():
            next(flat.lower_actor.net.parameters()).add_(.01)
            next(upper.upper_actor.net.parameters()).add_(.02)
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"groups": {}}
            output = Path(directory)/"result.json"
            for p in spec.PERIODS:
                manifest["groups"][str(p)] = {"trained": {}}
                for method, model in (("flat_lower", flat), ("joint_conditioned", upper)):
                    path = experiment.learning.final_checkpoint(model, output, root=410011, period=p,
                        method=method, updates=8, protocol=spec.source)
                    manifest["groups"][str(p)]["trained"][method] = {"checkpoint": path}
            experiment.write_json(output, manifest)
            donor = lambda r, p, m: Path(directory)/"final_weights"/f"period_{p}_{m}.pt"
            with patch.object(spec, "source_result", return_value=output), patch.object(spec, "donor_checkpoint", side_effect=donor), \
                    patch.object(experiment.baseline, "qualify", side_effect=lambda c, **kw: c), \
                    patch.object(experiment.baseline.joint.fresh, "load_source", return_value=(originals, predictor(), {}, {})):
                loaded, _, record, _ = experiment.load_source(410011)
                self.assertEqual(record["lower_method"], "flat_lower")
                self.assertEqual(record["upper_method"], "joint_conditioned")
                for model in loaded.values():
                    torch.testing.assert_close(model.lower_actor.net[0].weight[:, :392], flat.lower_actor.net[0].weight, atol=0, rtol=0)
                    torch.testing.assert_close(model.upper_actor.state_dict(), upper.upper_actor.state_dict(), atol=0, rtol=0)
                path = donor(410011, 50, "flat_lower")
                saved = torch.load(path, weights_only=False)
                saved["weights"]["lower_actor"]["log_std"] += .01
                torch.save(saved, path)
                with self.assertRaises(AssertionError):
                    experiment.load_source(410011)

    def test_hint_preserves_all392_feedback_and_is_causal_with_no_blinded_planning(self):
        models, pred, _, cal = self.sources()
        model = models["50"]
        args = spec.arguments(410011, preflight=True)
        args.horizon = 100
        weights = experiment.native.joint.inference_weights(model)
        with patch.object(experiment.native, "_WORKER", (model, args)), \
                patch.object(experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
            outputs = {}
            for v in ("base", "blind", "forecast_hint", "learned_hint", "forecast_blinded", "learned_blinded"):
                with patch.object(experiment.native.joint, "_make_task", return_value=DenseTask()):
                    outputs[v] = experiment.worker_episode((weights, 107001, 107002, v, 50, pred, .02, cal["50"]["envelope"], True))
                experiment.check_row(outputs[v][1], 50, 100)
                np.testing.assert_allclose(outputs[v][0].state[:, :392], outputs["base"][0].state[:, :392], atol=1e-6, rtol=0)
                np.testing.assert_allclose(outputs[v][0].action, outputs["base"][0].action, atol=1e-6, rtol=0)
            self.assertFalse(np.all(outputs["forecast_hint"][0].state[:, 392:] == 0))
            for v in ("base", "blind", "forecast_blinded", "learned_blinded"):
                np.testing.assert_array_equal(outputs[v][0].state[:, 392:], np.zeros((100, 4)))
                self.assertEqual(outputs[v][1]["upper_calls"], 0)
            with patch.object(experiment.native.joint, "_make_task", return_value=DenseTask(1.)):
                changed, _ = experiment.worker_episode((weights, 107001, 107002, "forecast_hint", 50, pred, .02, cal["50"]["envelope"], True))
            np.testing.assert_array_equal(changed.state[:50], outputs["forecast_hint"][0].state[:50])
            with patch.object(experiment.native.joint, "_make_task", return_value=DenseTask()), \
                    patch.object(model, "act_upper", side_effect=AssertionError("blind upper call")), \
                    patch.object(experiment.baseline.forecast, "plan_points", side_effect=AssertionError("blind prediction")):
                experiment.worker_episode((weights, 107001, 107002, "learned_blinded", 50, None, .02, {}, False))

    def test_independent_learned_hint_pairs_keep_feedback_not_upper_innovations_equal(self):
        models, pred, _, cal = self.sources()
        model, args = models["50"], spec.arguments(410011, preflight=True)
        args.horizon = 100
        roster = spec.seed_roles(410011, preflight=True)["training_rounds"][0]["credit_A"][0]
        with patch.object(experiment.native, "_WORKER", (model, args)), \
                patch.object(experiment.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                patch.object(experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
            pairs = [experiment.worker_episode((experiment.native.joint.inference_weights(model), roster["scenario_seed"], n,
                "learned_hint", 50, pred, .02, cal["50"]["envelope"], True)) for n in roster["noise_seeds"]]
            experiment.pair_check(pairs, roster, root=410011)
            self.assertFalse(np.array_equal(pairs[0][1]["upper_standard_noise"], pairs[1][1]["upper_standard_noise"]))

    def test_reduced_full_updates_lower_real_scores_and_writes_only_six_final_models(self):
        options, args = spec.options, spec.arguments
        small = lambda **kw: {**options(**kw), "updates": 1, "credit_scenarios_per_batch": 2, "evaluation_episodes": 1, "workers": 1}
        short = lambda root, **kw: SimpleNamespace(**{**vars(args(root, **kw)), "horizon": 200})
        sources, visited = self.sources(), []
        before = {p: copy.deepcopy(m.state_dict()) for p, m in sources[0].items()}
        real_update = experiment.learning.update_mean

        def inspect(model, batches, **kw):
            visited.append(kw["method"])
            self.assertEqual(kw["allocation"], {"lower": 1.})
            self.assertTrue(all(len(groups) == 4 for groups in batches.values()))
            return real_update(model, batches, **kw)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(spec, "options", side_effect=small), patch.object(spec, "arguments", side_effect=short), \
                    patch.object(experiment, "load_source", return_value=sources), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment.learning, "update_mean", side_effect=inspect), \
                    patch.object(experiment.native.joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(experiment.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
                cell = experiment.run(410011, preflight=False, output=Path(directory)/"result.json")
                self.assertEqual(cell["cost"], spec.budget(preflight=False))
                self.assertEqual(cell["native_planning_cost"], spec.planning_budget(preflight=False))
                self.assertEqual(len(visited), 6)
                self.assertEqual(len(list((Path(directory)/"final_weights").glob("*.pt"))), 6)
                for g in cell["groups"].values():
                    for m, t in g["trained"].items():
                        saved = torch.load(t["checkpoint"], map_location="cpu", weights_only=False)
                        w = saved["weights"]["lower_actor"]["net.0.weight"][:, 392:]
                        self.assertEqual(torch.count_nonzero(w).item() == 0, m == "blind")
                bad = copy.deepcopy(cell)
                bad["groups"]["50"]["evaluation"]["learned_blinded"][0]["upper_calls"] = 1
                with self.assertRaisesRegex(ValueError, "inference calls"):
                    experiment.qualify(bad, preflight=False)
                bad = copy.deepcopy(cell)
                bad["groups"]["100"]["trained"]["learned_hint"]["history"][0]["actors"]["lower"]["gradient_episodes"] //= 2
                with self.assertRaisesRegex(ValueError, "credit"):
                    experiment.qualify(bad, preflight=False)
        for p, model in sources[0].items():
            experiment.native.curves.support.assert_frozen(model, before[p])

    def test_fresh_rosters_budgets_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = all_seeds(spec.seed_roles(root, preflight=preflight))
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                self.assertFalse(set(seeds) & set(all_seeds(spec.source.seed_roles(root, preflight=preflight))))
                seen.update(seeds)
                t = task_specification("unit_stage107", root, preflight=preflight)
                self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertIsNone(t.get("require_node"))
                self.assertEqual((t["cpu"], t["ram_mb"]), (3, 3072) if preflight else (9, 8192))
                self.assertIn(spec.RUNNER_SCRIPT, t["cmd"])
            q = qualification_task("unit_stage107", preflight=preflight)
            self.assertIsNone(q["result_dir"])
            self.assertEqual(len(q["wait_for_files"]), len(spec.roots(preflight=preflight)))
        b = spec.budget(preflight=False)
        self.assertEqual((8*b["native_episodes"], 8*b["native_steps"]), (52224, 62668800))
        self.assertEqual((8*b["actor_mean_parameter_updates"], 8*b["checkpoint_writes"]), (384, 48))
        self.assertEqual((spec.budget(preflight=True)["native_episodes"], spec.budget(preflight=True)["actor_mean_parameter_updates"]), (240, 12))

    def test_all20_contrasts_and_six_primary_bounds_required(self):
        cells = [{"root": r, "groups": {"both": {"effects": dict.fromkeys(spec.ENDPOINTS, 2.)}},
            "cost": spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: c), \
                patch.object(spec, "BOOTSTRAP_DRAWS", 128), \
                patch.object(experiment.native.np, "quantile", wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells, preflight=False)
            self.assertEqual(len(result["endpoints"]), 20)
            self.assertEqual(quantile.call_args.args[1], [.05/40, 1-.05/40])
            self.assertEqual(result["optional_plan_confirmation"], "supported")
            for key in spec.PRIMARY_ENDPOINTS:
                for c in cells:
                    c["groups"]["both"]["effects"][key] = 0.
                self.assertEqual(experiment.aggregate(cells, preflight=False)["optional_plan_confirmation"], "not_supported")
                for c in cells:
                    c["groups"]["both"]["effects"][key] = 2.
            with self.assertRaises(ValueError):
                experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
