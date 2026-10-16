from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_freqduet_v70_concentration as audit


class ConcentrationTest(unittest.TestCase):
    def test_secondary_journey_uses_fixed_choices_and_only_evaluation_halves(self):
        groups = []
        for future in range(32):
            rows = [dict(journey_min=0., waiting_min=0., onboard_min=0.) for _ in range(7)]
            rows[1] = dict(journey_min=-20. if future < 16 else 2.,
                           waiting_min=-5. if future < 16 else .5,
                           onboard_min=-15. if future < 16 else 1.5)
            rows[2] = dict(journey_min=-100., waiting_min=-50., onboard_min=-50.)
            groups.append(dict(rows=rows))
        result = audit.secondary_effects(groups, [dict(action_s=5), dict(action_s=0)], 0)
        self.assertEqual(result, dict(journey_min=1., waiting_min=.25, onboard_min=.75))

    def test_concentration_retains_bad_days_in_net_effect_and_omission(self):
        result = audit.concentration([-90., -10., 20.])
        self.assertEqual(result["strongest_day_index"], 0)
        self.assertEqual(result["gross_improvement_share"], .9)
        self.assertEqual(result["net_improvement_share"], 1.125)
        self.assertEqual(result["mean_without_strongest"], 5.)


if __name__ == "__main__":
    unittest.main()
