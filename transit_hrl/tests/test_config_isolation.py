import unittest
from pathlib import Path

from freq_hrl.experiments.transit.config_isolation import audit_config_isolation


class ConfigIsolationTest(unittest.TestCase):
    def test_transit_ablation_configs_only_change_allowed_paths(self):
        result = audit_config_isolation(
            Path("transit_hrl/freq_transitduet/configs_freqduet")
        )
        self.assertTrue(result["passed"], result["violations"])


if __name__ == "__main__":
    unittest.main()
