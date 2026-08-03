import unittest
from pathlib import Path

from freq_hrl.core.shared_core_audit import audit_shared_training_core


class SharedCoreAuditTest(unittest.TestCase):
    def test_shared_training_core_source_boundaries(self):
        audit = audit_shared_training_core(Path("."))
        self.assertEqual(audit["status"], "partial")
        self.assertEqual(audit["core_boundary"]["status"], "supported")
        self.assertEqual(audit["core_boundary"]["violations"], [])
        adapters = {
            row["adapter"]: row
            for row in audit["adapter_evidence"]
        }
        self.assertEqual(adapters["trading_ppo"]["status"], "supported")
        self.assertEqual(adapters["transit_surrogate_ppo"]["status"], "failed")
        self.assertEqual(adapters["transit_native_replay_update"]["status"], "failed")
        self.assertEqual(adapters["transit_native_actor_core"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
