import unittest

from scripts.validate_freqduet_protocol_v3_configs import validate_all


class ProtocolV3ConfigTest(unittest.TestCase):
    def test_selection_matrix_has_one_strict_physical_contract(self):
        result = validate_all()
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["protocol_version"], "freqduet-eval-v3")
        self.assertEqual(len(result["configs"]), 13)


if __name__ == "__main__":
    unittest.main()
