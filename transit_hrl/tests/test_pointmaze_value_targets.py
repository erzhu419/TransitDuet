import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_value_targets as experiment
from freq_hrl.experiments import pointmaze_continuing_credit as continuing
from freq_hrl.experiments import pointmaze_update_diagnostics as diagnostics
from freq_hrl.experiments import pointmaze_matched_upper as previous
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, concat_hierarchical_batches
from scripts import pointmaze_value_targets_stage64_spec as spec
from scripts.submit_pointmaze_value_targets_stage64_scheduleurm import task_specification
import test_pointmaze_update_diagnostics as archive_data
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class ValueTargetsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive_data.UpdateDiagnosticsTest.setUpClass()
        cls.helper = archive_data.UpdateDiagnosticsTest()

    def data(self):
        model = self.helper.model()
        with tempfile.TemporaryDirectory() as directory:
            native, _, _ = self.helper.archived_episode(model, directory)
        return model, continuing.episode_batch(native, native.lower.old_value, 300)

    def test_slim_raw_loop_preserves_original_value_and_adam_without_actor_calls(self):
        model, batch = self.data()
        # Include nonempty moments when testing unchanged raw-unit updates.
        continuing.lower_calibration(model, batch, root=310001, period=50, iteration=1)
        plain, fit = copy.deepcopy(model), experiment.ValueFit(copy.deepcopy(model), "gae_raw")
        _, targets = model._gae(batch.lower.reward, batch.lower.done, batch.lower.duration, batch.lower.old_value)
        expected = continuing.lower_calibration(plain, batch, root=310001, period=50, iteration=2)
        steps = []
        hook = fit.model.lower_value_optimizer.register_step_post_hook(lambda *args: steps.append(1))
        with patch.object(fit.model.lower_actor, "log_prob_entropy", side_effect=AssertionError("critic loop must not call actor")):
            row = fit.update(batch.lower, targets, root=310001, period=50, iteration=2)
        hook.remove()
        self.assertEqual(row["value_optimizer_steps"], expected["lower_value_optimizer_steps"])
        self.assertEqual(row["value_optimizer_steps"], len(steps))
        for name in ("lower_value", "lower_value_optimizer", "lower_actor", "lower_actor_optimizer", "upper_value", "upper_value_optimizer"):
            torch.testing.assert_close(getattr(plain, name).state_dict(), getattr(fit.model, name).state_dict(), atol=0, rtol=0)

    def test_output_rebase_preserves_public_values_and_fixed_training_frame(self):
        model, batch = self.data()
        fit = experiment.ValueFit(copy.deepcopy(model), "mc_normalized")
        mc = experiment.monte_carlo_returns(batch.lower, model.config.gamma)
        error = fit.initialize_frame(float(mc.mean()), float(mc.std()), batch.lower)
        self.assertLessEqual(error, 2e-4)
        initial_frame = fit.location, fit.scale
        for key, value in model.lower_value.state_dict().items():
            if not key.startswith("net.4"):
                torch.testing.assert_close(value, fit.model.lower_value.state_dict()[key], atol=0, rtol=0)
        fit.update(batch.lower, mc, root=310001, period=50, iteration=1)
        state = torch.as_tensor(batch.lower.value_state, dtype=torch.float32)
        public = copy.deepcopy(model.lower_value)
        public.load_state_dict(fit.public_state())
        with torch.no_grad():
            torch.testing.assert_close(public(state), fit.location + fit.scale * fit.model.lower_value(state), atol=2e-4, rtol=1e-6)
        self.assertEqual((fit.location, fit.scale), initial_frame)
        with self.assertRaises(ValueError):
            fit.initialize_frame(999., 1., batch.lower)
        old = experiment.ValueFit(copy.deepcopy(fit.model), "mc_normalized")
        with self.assertRaisesRegex(ValueError, "Adam must be empty"):
            old.initialize_frame(0., 1., batch.lower)

    def reference(self, model, source, source_file):
        args = spec.arguments(310001, preflight=True)
        raw = source_file.parent.with_name(source_file.parent.name + "_raw")
        cells = {}
        diagnostics.init_worker(model.config, args)
        for period in spec.PERIODS:
            p = str(period)
            cells[p] = {}
            for arm in spec.TRAIN_POLICIES:
                control = copy.deepcopy(model)

                def batch(phase, item):
                    path = raw / p / arm / phase / str(item["iteration"]) / "training"
                    pairs = [diagnostics.worker_reconstruct((joint.inference_weights(control),
                        str(path / f"episode_{r['seed']}.npz"), r["seed"], period)) for r in item["rows"]]
                    b = concat_hierarchical_batches([b for b, _ in pairs])
                    return continuing.episode_batch(b, b.lower.old_value, args.horizon)

                for item in source["calibration"][p][arm]["history"]:
                    continuing.lower_calibration(control, batch("warmup", item), root=310001, period=period, iteration=item["iteration"])
                probe = batch("train", source["training"][p][arm]["history"][0]).lower
                mc = experiment.monte_carlo_returns(probe, control.config.gamma)
                cells[p][arm] = {"critic_probe": {"episode_value_episode_mc": experiment.credit.value_metrics(probe.old_value, mc)}}
        return {"status": "complete", "root": 310001, "preflight": True, "contract": spec.source.contract(), "comparisons": cells}

    def test_archive_probe_control_freeze_units_and_all_forward_optimizer_costs(self):
        model, predictor = self.helper.model(), self.helper.predictor
        initialization = {"config": json.loads(json.dumps(model.config.__dict__)), "checkpoints": {"50": "clone50.pt", "100": "clone100.pt"}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_file, ref_file = directory / "stage57/result.json", directory / "stage63/result.json"
            with patch.object(previous, "load_source", side_effect=lambda *a, **kw:
                    ({p: copy.deepcopy(model) for p in ("50", "100")}, predictor, initialization)), \
                    patch.object(previous, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(spec, "training_result", return_value=source_file), \
                    patch.object(spec, "source_result", return_value=ref_file), \
                    patch.object(joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                source = previous.train(310001, preflight=True, output=source_file)
                experiment.write_json(ref_file, self.reference(model, source, source_file))
                gae, calls = FrequencySeparatedActorCriticPPO._gae, []

                def counted(model, *args, **kwargs):
                    calls.append(1)
                    return gae(model, *args, **kwargs)

                with patch.object(joint, "_make_task", side_effect=AssertionError("value ablation must not sample native paths")), \
                        patch.object(FrequencySeparatedActorCriticPPO, "_gae", counted), \
                        patch.object(experiment, "monte_carlo_returns", wraps=experiment.monte_carlo_returns) as mc_calls:
                    result = experiment.replay(310001, preflight=True, output=directory / "stage64/result.json")
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(len(calls), result["cost"]["calibration_gae_calls"] + result["cost"]["probe_gae_calls"])
                self.assertEqual(mc_calls.call_count, result["cost"]["mc_target_calls"])
            for p, arms in result["groups"].items():
                for arm, cell in arms.items():
                    self.assertEqual(cell["frame"]["iteration"], 1)
                    for treatment, row in cell["treatments"].items():
                        saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
                        self.assertEqual((saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["treatment"]),
                            (spec.EXPERIMENT_PROTOCOL, 310001, int(p), arm, treatment))
                        public = copy.deepcopy(saved["value_training_state"])
                        public["net.4.weight"] *= saved["scale"]
                        public["net.4.bias"] = public["net.4.bias"] * saved["scale"] + saved["location"]
                        torch.testing.assert_close(public, saved["public_value_state"], atol=0, rtol=0)
                        self.assertTrue(saved["value_optimizer_training_units"]["state"])
                        self.assertTrue(all(0 <= v <= 1 for v in row["representation"]["tanh_saturation_fraction"]))
        self.assertEqual(summary["cost"], spec.budget(preflight=True))
        self.assertEqual(summary["value_optimizer_steps"], dict.fromkeys(spec.TREATMENTS, 40))
        self.assertEqual(summary["supervised_MC_optimizer_steps"], 80)
        self.assertEqual(summary["cost"]["actor_optimizer_steps"], 0)
        for mutation in ("frame", "identity", "budget", "steps"):
            bad = copy.deepcopy(result)
            cell = bad["groups"]["50"]["joint_ppo"]
            if mutation == "frame":
                cell["frame"]["source"] = "probe"
            elif mutation == "identity":
                cell["frozen_actor_upper_and_Adam"] = "failed"
            elif mutation == "budget":
                bad["cost"]["mc_supervised_updates"] -= 1
            else:
                cell["history"][0]["updates"]["mc_normalized"]["value_optimizer_steps"] -= 1
            with self.assertRaises(ValueError):
                experiment.qualify(bad, preflight=True)

    def test_frozen_factorial_budget_substantive_gate_and_dynamic_pool(self):
        b = spec.budget(preflight=False)
        self.assertEqual(8 * b["archive_episodes"], 4352)
        self.assertEqual(8 * b["extra_critic_scalar_calls"], 15667200)
        self.assertEqual(8 * b["calibration_updates"], 2048)
        self.assertEqual(8 * b["mc_supervised_updates"], 1024)
        self.assertEqual(8 * b["initialization_value_rows"], 614400)
        self.assertEqual(8 * b["representation_forward_batches"], 2432)
        self.assertEqual((spec.CANDIDATE, spec.MIN_PROBE_EV), ("mc_normalized", .1))
        self.assertIn(310037, spec.roots(preflight=False))
        for preflight in (True, False):
            task = task_specification("unit_stage64", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (2, 4096) if preflight else (9, 12288))
        cells = [{"root": root, "cost": b} for root in spec.roots(preflight=False)]
        with patch.object(experiment, "qualify", side_effect=lambda c, **kw: ({"root": c["root"]},
                dict.fromkeys(spec.TREATMENTS, 1), [{"root": c["root"], "period": 50, "arm": "joint_ppo"}] if c["root"] == 310037 else [])):
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(summary["native_trial_prerequisite"], "hold")
        self.assertEqual(summary["candidate_fit_failures"][0]["root"], 310037)
        self.assertEqual(len(summary["root_rows"]), 8)
        with self.assertRaises(ValueError):
            experiment.aggregate(cells[:-1], preflight=False)


if __name__ == "__main__":
    unittest.main()
