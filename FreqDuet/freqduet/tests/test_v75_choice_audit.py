from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v75_choice_audit as audit


class ChoiceAuditTest(unittest.TestCase):
    def test_selected_longer_action_can_have_false_predicted_passenger_gain(self):
        data = dict(outcomes=np.zeros((1, 32, 7, 11)), native=np.array([3]), days=np.array([10]),
                    cases=np.array([0]), windows=np.array([7200]))
        passenger = audit.old.OUTCOMES.index("passenger_person_s")
        data["outcomes"][0, :, 6, passenger] = 100.
        predictions = np.zeros((1, 7, 5))
        predictions[0, 6, 0] = -10.
        row = audit.audit_rows(data, predictions)[0]
        self.assertEqual((row["delta_action_s"], row["predicted"]["passenger_person_s"],
                          row["actual"]["passenger_person_s"]), (30, -10., 100.))
        self.assertEqual(row["selected_half_means"], [100., 100.])
        summary = audit.summarize([row])
        self.assertEqual((summary["longer"], summary["both_halves_passenger_worsen"]), (1, 1))

    def test_optimistic_choice_still_enforces_every_true_constraint(self):
        data = dict(outcomes=np.zeros((1, 32, 7, 11)), native=np.array([3]), days=np.array([10]),
                    cases=np.array([0]), windows=np.array([7200]))
        passenger = audit.old.OUTCOMES.index("passenger_person_s")
        data["outcomes"][0, :, 0, passenger] = -100.
        data["outcomes"][0, :, 0, audit.old.OUTCOMES.index("trips_incomplete")] = 1.
        data["outcomes"][0, :, 1, passenger] = -20.
        row = audit.audit_rows(data, np.zeros((1, 7, 5)))[0]
        self.assertFalse(row["changed"])
        self.assertEqual(row["optimistic"]["passenger_person_s"], -20.)
        self.assertEqual(row["optimistic"]["trips_incomplete"], 0.)
        directions = audit.direction_summary([row])
        self.assertIsNone(directions["shorter"]["actual_passenger"])


if __name__ == "__main__":
    unittest.main()
