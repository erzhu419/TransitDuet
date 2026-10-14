import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import complete_freqduet_v34_optimized_evaluation as completion


class OptimizedCompletionTest(unittest.TestCase):
    def test_optimized_snapshot_is_in_shared_work_directory(self):
        self.assertEqual(str(completion.OPTIMIZED_ROOT),
                         "/home/zhengliang01/scheduleurm_work/"
                         "FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet")

    def test_incomplete_training_cannot_silently_restart(self):
        matrix, runner = Mock(), Mock()
        matrix.training_complete.return_value = False
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(completion, "optimized_runtime",
                             return_value=(matrix, runner, None, None)):
            with self.assertRaisesRegex(ValueError, "refusing to retrain"):
                completion.audit(Path(directory))
        runner.assert_not_called()
        matrix.training_complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
