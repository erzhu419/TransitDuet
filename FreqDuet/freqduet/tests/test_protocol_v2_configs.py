import unittest

from scripts.validate_freqduet_protocol_v2_configs import validate_all


class ProtocolV2ConfigTest(unittest.TestCase):
    def test_main_and_ablations_are_unconfounded(self):
        result = validate_all()
        self.assertEqual(result["status"], "valid")
        self.assertEqual(len(result["ablations"]), 7)


if __name__ == "__main__":
    unittest.main()
