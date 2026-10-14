import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_freqduet_v34_duration import (
    CONFIGS, CURRENT, METRICS, REFERENCE, duration_analysis,
)


class DurationAnalysisTest(unittest.TestCase):
    def test_relative_advantage_can_shrink_while_every_policy_improves(self):
        rows = []
        for config in CONFIGS:
            early, late = (0.4, 0.3) if config == CURRENT else (0.5, 0.3)
            for train_seed in (12011, 933013):
                for eval_seed in (45007, 934011):
                    for checkpoint, value in ((39, early), (199, late)):
                        rows.append(dict(config=config, train_seed=train_seed, eval_seed=eval_seed,
                                         checkpoint_ep=checkpoint,
                                         **{metric: value for metric in METRICS}))
        frame = pd.DataFrame(rows).sample(frac=1, random_state=3)
        means, absolute, seeds, relative = duration_analysis(
            frame, lambda table, metric: (table[metric].mean(), table[metric].mean()))
        main = absolute.loc[(absolute.config == CURRENT) & (absolute.metric == "headway_cv")
                            & (absolute.cohort == "all_train_all_eval")].iloc[0]
        self.assertAlmostEqual(main.delta_ep199_minus_ep39, -0.1)
        self.assertAlmostEqual(relative.loc[relative.reference == REFERENCE, "headway_cv"].mean(), 0.1)
        self.assertEqual(len(means), 6)
        self.assertEqual(len(seeds), 6)


if __name__ == "__main__":
    unittest.main()
