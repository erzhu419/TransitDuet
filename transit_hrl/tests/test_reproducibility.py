import subprocess
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.reproducibility import (
    derive_seed,
    registered_git_source_identity,
    source_manifest_sha256,
    training_rollout_seed,
    validate_evaluation_seed_roles,
    verify_current_freq_hrl_source_identity,
)


class ReproducibilityProtocolTest(unittest.TestCase):
    def test_structured_seed_derivation_is_stable_and_namespaced(self):
        first = derive_seed("alpha", 7, 42, 0)
        self.assertEqual(first, derive_seed("alpha", 7, 42, 0))
        self.assertNotEqual(first, derive_seed("beta", 7, 42, 0))

    def test_training_paths_change_by_replicate_root_and_iteration(self):
        seeds = {
            training_rollout_seed(rep, root, iteration, domain="trading")
            for rep in (7, 11)
            for root in (42, 123)
            for iteration in (0, 1)
        }
        self.assertEqual(len(seeds), 8)

    def test_validation_and_heldout_test_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_evaluation_seed_roles([1, 2], [2, 3])

    def test_source_manifest_hashes_code_and_config_but_not_cache_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "config.yaml").write_text("value: 1\n", encoding="utf-8")
            initial = source_manifest_sha256(root)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "ignored.py").write_text("VALUE = 99\n", encoding="utf-8")
            (root / "notes.txt").write_text("not executable input\n", encoding="utf-8")
            self.assertEqual(initial, source_manifest_sha256(root))
            (root / "config.yaml").write_text("value: 2\n", encoding="utf-8")
            self.assertNotEqual(initial, source_manifest_sha256(root))

    def test_registered_git_identity_rejects_working_source_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "nested_workspace"
            package = workspace / "pkg"
            package.mkdir(parents=True)
            (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "add", "nested_workspace/pkg"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "source"], check=True
            )
            revision, manifest = registered_git_source_identity(
                workspace, Path("pkg")
            )
            self.assertEqual(len(revision), 40)
            self.assertEqual(manifest, source_manifest_sha256(package))
            (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                registered_git_source_identity(workspace, Path("pkg"))

    def test_confirmatory_source_identity_cannot_be_unregistered(self):
        with self.assertRaisesRegex(ValueError, "verified source identity"):
            verify_current_freq_hrl_source_identity(require_verified=True)


if __name__ == "__main__":
    unittest.main()
