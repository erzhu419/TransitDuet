import unittest

from scripts import analyze_native_transit_closed_loop_preflight_stage303 as audit


class ClosedLoopMenuAuditTests(unittest.TestCase):
    def records(self):
        return [{"draw": d, "future_seed": audit.source.future_seed(397, "low_noise", 8, d),
            "values": {str(a): [0., 0., 0., 0.] for a in range(6)}} for d in range(8)]

    def test_complete_identity_menu_and_expected_future_are_accepted(self):
        panel = audit.validate_panel(self.records(), {"original_choice": 0}, 397, "low_noise", 8, list(range(8)))
        self.assertEqual(panel.shape, (1, 8, 6, 4))

    def test_future_reuse_in_screen_is_rejected(self):
        records = self.records()
        for r in records: r["draw"] += 8
        with self.assertRaisesRegex(ValueError, "future"):
            audit.validate_panel(records, {"original_choice": 0}, 397, "low_noise", 8, list(range(8, 16)))

    def test_cost_accounting_and_original_action_identity_are_checked(self):
        for action in (0, 1):
            records = self.records(); records[0]["values"][str(action)][0] = 1.
            with self.assertRaisesRegex(ValueError, "Identity|decomposition"):
                audit.validate_panel(records, {"original_choice": 0}, 397, "low_noise", 8, list(range(8)))


if __name__ == "__main__": unittest.main()
