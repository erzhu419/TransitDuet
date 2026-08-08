import unittest

from scripts.paper_submission_gate import require_submission_ready


class PaperSubmissionGateTest(unittest.TestCase):
    def test_hold_blocks_default_generation(self):
        with self.assertRaisesRegex(RuntimeError, "submission_status"):
            require_submission_ready({
                "submission_status": "hold_pending_protocol_v5",
                "active_protocol": "freqduet-eval-v5",
            })

    def test_historical_override_is_explicit(self):
        require_submission_ready(
            {"submission_status": "hold_pending_protocol_v5"},
            allow_historical=True,
        )

    def test_ready_manifest_is_accepted(self):
        require_submission_ready({"submission_status": "ready"})


if __name__ == "__main__":
    unittest.main()
