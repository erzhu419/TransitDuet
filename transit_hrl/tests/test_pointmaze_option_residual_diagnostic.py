import unittest

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_option_residual_diagnostic as diagnostic
from test_pointmaze_optional_plan import OptionalPlanTest
from scripts import pointmaze_option_residual_diagnostic_spec as spec


class OptionResidualDiagnosticTest(unittest.TestCase):
    def test_frozen_roster_and_budget(self):
        self.assertEqual(spec.roots(), (410011,))
        self.assertEqual(spec.EPISODES_PER_PERIOD, 32)
        self.assertEqual(spec.budget()["native_episodes"], 128)
        self.assertEqual(spec.contract()["selection"], "single_root_410011_fixed_before_reading_diagnostic_results")

    def test_checkpoint_roster_is_final_branch_only(self):
        for period in spec.PERIODS:
            for arm in ("learned", "forecast"):
                self.assertIn(f"period_{period}_{arm}.pt", str(spec.checkpoint(410011, period, arm)))

    def test_loaded_readout_is_used_by_correction_metric(self):
        model = OptionalPlanTest().sources()[0]["50"]
        actor = diagnostic.trainer.branch(model)
        with torch.no_grad():
            actor.readout.bias.fill_(0.1)
        metrics = diagnostic.branch_metrics(model, actor.state_dict(), np.zeros((8, 396), dtype=np.float32))
        self.assertGreater(metrics["residual_correction_norm"], 0.)


if __name__ == "__main__":
    unittest.main()
