import unittest

from freq_hrl.experiments.evidence_policy import (
    OBSERVED_EVIDENCE,
    PROJECTION_EVIDENCE,
    annotate_check,
    headline_status,
    is_headline_eligible,
)


class EvidencePolicyTest(unittest.TestCase):
    def test_projection_can_keep_exploratory_status_but_not_headline_status(self):
        row = annotate_check({
            "metric": "projected_avg_wait_min",
            "status": "supported",
        })
        self.assertEqual(row["evidence_class"], PROJECTION_EVIDENCE)
        self.assertFalse(is_headline_eligible(row))
        self.assertEqual(row["headline_status"], "ineligible")
        self.assertEqual(headline_status(row), "ineligible")

    def test_observed_raw_metric_is_headline_eligible(self):
        row = annotate_check({
            "metric": "promotion_raw_ep_reward",
            "status": "supported",
        })
        self.assertEqual(row["evidence_class"], OBSERVED_EVIDENCE)
        self.assertTrue(is_headline_eligible(row))
        self.assertEqual(headline_status(row), "supported")


if __name__ == "__main__":
    unittest.main()
