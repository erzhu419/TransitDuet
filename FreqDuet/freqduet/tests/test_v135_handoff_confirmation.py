from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v135_handoff_confirmation as run


class HandoffConfirmationTests(unittest.TestCase):
    def test_development_actors_do_not_enter_primary_cohort(self):
        self.assertEqual(len(run.VALIDATION_SEEDS), 10)
        self.assertFalse(set(run.VALIDATION_SEEDS) & set(run.DEVELOPMENT_SEEDS))
        self.assertEqual(len(run.ITEMS), 192)
        self.assertEqual(len(set(run.DAYS) & {2231001, 2231002, 2231003, 2231004}), 0)

    def test_six_shards_cover_every_pair_once(self):
        covered = [i for shard in range(run.SHARDS) for i in range(*run.bounds(shard))]
        self.assertEqual(covered, list(range(len(run.ITEMS))))
        self.assertTrue(all(b-a == 32 for a, b in map(run.bounds, range(run.SHARDS))))

    def test_both_arms_retain_no_terminal_coordination(self):
        old = run.ConfirmationAudit(12037, {}, 'old')
        new = run.ConfirmationAudit(12037, {}, 'handoff')
        self.assertFalse(old.active)
        self.assertFalse(new.active)
        self.assertIsInstance(new.stream, run.frozen.ActiveLaunchETA)
        self.assertNotIsInstance(old.stream, run.frozen.ActiveLaunchETA)

    def test_improvement_and_nonharm_are_distinct(self):
        stats = {k: dict(ci95=[-2., -1.]) for k in run.PRIMARY}
        stats.update({k: dict(ci95=[0., 0.]) for k in run.COMPLETION})
        self.assertTrue(run.joint_gate(stats))
        stats['cv_squared'] = dict(ci95=[-1., 0.])
        self.assertFalse(run.joint_gate(stats))
        self.assertTrue(run.joint_gate(stats, strict=False))
        stats['inclusive_terminal_delay_s'] = dict(ci95=[-1., .01])
        self.assertFalse(run.joint_gate(stats, strict=False))

    def test_missing_and_duplicate_pairs_do_not_become_zero_effects(self):
        rows = [dict(index=i, seed=seed, day=day) for i, (seed, day) in enumerate(run.ITEMS)]
        run.validate_grid(rows)
        with self.assertRaises(AssertionError):
            run.validate_grid(rows[:-1])
        with self.assertRaises(AssertionError):
            run.validate_grid([*rows[:-1], rows[0]])

    def test_smoke_artifact_directory_does_not_move_runtime_proof_root(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)/'campaign'
            with patch.object(run.frozen.prior.model, 'load_geometry', return_value={}), \
                    patch.object(run.pilot, 'rollout', side_effect=RuntimeError('stop before simulation')) as rollout:
                with self.assertRaisesRegex(RuntimeError, 'stop before simulation'):
                    run.run_path(root, Path(directory), 12011, 2231004, 'old', artifact_root=root/'smoke_runs')
                self.assertEqual(rollout.call_args.args[0], root)


if __name__ == '__main__':
    unittest.main()
