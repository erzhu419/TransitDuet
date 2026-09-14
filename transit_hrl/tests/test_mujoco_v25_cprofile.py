import cProfile
import json
from pathlib import Path
import shlex
import tempfile
import unittest

from freq_hrl.experiments.mujoco.control_validation import build_parser
from scripts.run_mujoco_cprofile_small_export import _write_summary
from scripts.submit_mujoco_v25_cprofile_scheduleurm import command, task


class MujocoV25CProfileTest(unittest.TestCase):
    def test_profile_summary_is_compact_and_sorted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = cProfile.Profile()
            profile.runcall(lambda: sum(range(100)))
            profile.dump_stats(root / "cell.prof")
            _write_summary(root / "cell.prof", root / "small", 1.25)
            payload = json.loads((root / "small/profile_summary.json").read_text())
            self.assertEqual(payload["wall_seconds"], 1.25)
            self.assertGreater(payload["total_calls"], 0)
            cumulative = [row["cumulative_seconds"] for row in payload["top_cumulative"]]
            self.assertEqual(cumulative, sorted(cumulative, reverse=True))
            self.assertFalse((root / "small/cell.prof").exists())

    def test_profile_command_matches_representative_training_contract(self):
        tokens = shlex.split(command("profile_test"))
        control_args = tokens[tokens.index("--") + 1:]
        args = build_parser().parse_args(control_args + ["--output-dir", "/tmp/profile_test"])
        self.assertEqual(args.env_id, "HalfCheetah-v5")
        self.assertEqual(args.iterations, 8)
        self.assertEqual(args.steps, 512)
        self.assertEqual(args.upper_projection_consistency_objective, "action_sample")
        self.assertEqual(len(args.train_seeds), 4)
        self.assertEqual(len(args.selection_seeds), 4)
        job = task("profile_test")
        self.assertEqual(job["cpu"], 1)
        self.assertEqual(job["ram_mb"], 2048)
        self.assertIsNone(job["require_node"])
        self.assertEqual(len(job["allowed_nodes"]), 6)


if __name__ == "__main__":
    unittest.main()
