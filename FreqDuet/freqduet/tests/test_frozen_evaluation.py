import copy
import tempfile
import unittest
from pathlib import Path

from runner_v3 import TransitDuetV2Runner, load_config


class FrozenEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def _config(self, logs_dir):
        cfg = copy.deepcopy(load_config(
            str(self.root / "configs_ablation/H_hiro.yaml")))
        cfg["seed"] = 17
        cfg["env"].update({
            "effective_trip_num": 4,
            "service_start_hour": 6,
            "service_end_hour": 6,
            "demand_end_time_s": 600,
            "evaluation_end_time_s": 1200,
            "demand_noise": 0.0,
        })
        cfg["coupling"]["upper_warmup_eps"] = 0
        cfg["upper"]["fleet_mode"] = "fixed"
        cfg["upper"]["N_fleet"] = 4
        cfg.setdefault("logging", {})["logs_dir"] = logs_dir
        return cfg

    def test_learned_upper_runs_without_mutating_frozen_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = TransitDuetV2Runner(self._config(tmp))
            before = runner._policy_digest()
            row = runner.run_episode(
                1,
                training=False,
                scenario_seed=1001,
                record_diagnostics=False,
            )
            self.assertGreater(row["n_dispatches"], 0)
            self.assertEqual(row["lower_policy_frozen"], 1.0)
            self.assertEqual(row["lower_critic_frozen"], 1.0)
            self.assertEqual(row["upper_policy_frozen"], 1.0)
            self.assertEqual(before, runner._policy_digest())

    def test_deployment_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(tmp)
            runner = TransitDuetV2Runner(cfg)
            expected = runner._policy_digest()
            runner._save_checkpoint(3)

            restored = TransitDuetV2Runner(cfg)
            loaded_ep = restored.load_checkpoint(
                Path(runner.log_dir) / "checkpoints")
            self.assertEqual(loaded_ep, 3)
            self.assertEqual(expected, restored._policy_digest())
            rows, _ = restored.evaluate([2001], policy_ep=3)
            self.assertGreater(rows[0]["n_dispatches"], 0)

    def test_strict_evaluation_rejects_two_file_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(tmp)
            runner = TransitDuetV2Runner(cfg)
            runner._save_checkpoint(3)
            checkpoints = Path(runner.log_dir) / "checkpoints"
            (checkpoints / "runner_ep3.pt").unlink()

            restored = TransitDuetV2Runner(cfg)
            with self.assertRaisesRegex(
                    FileNotFoundError, "deployment checkpoint"):
                restored.load_checkpoint(
                    checkpoints,
                    require_deployment_state=True,
                )


if __name__ == "__main__":
    unittest.main()
