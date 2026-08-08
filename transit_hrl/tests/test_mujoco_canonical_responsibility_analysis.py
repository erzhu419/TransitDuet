import unittest

from freq_hrl.experiments.mujoco.canonical_responsibility_analysis import (
    canonical_gate_decision,
)


def rows(*, one_branch: bool, complete: bool):
    output = []
    for environment in ("A", "B", "C"):
        for method, passed in (
            ("freq_hrl_no_leakage", one_branch),
            ("freq_hrl_safe_selector", complete),
        ):
            output.append({
                "environment": environment,
                "method": method,
                "environment_gate_pass": passed,
            })
    return output


class MujocoCanonicalResponsibilityAnalysisTest(unittest.TestCase):
    def test_both_registered_gates_must_pass(self):
        decision = canonical_gate_decision(
            rows(one_branch=True, complete=True),
            environments=("A", "B", "C"),
        )
        self.assertEqual(decision["status"], "canonical_state_gate_passed")
        self.assertTrue(decision["one_branch_exact_invariance_gate_pass"])
        self.assertTrue(decision["complete_safe_method_gate_pass"])

    def test_exact_invariance_failure_blocks_the_gate(self):
        decision = canonical_gate_decision(
            rows(one_branch=False, complete=True),
            environments=("A", "B", "C"),
        )
        self.assertEqual(decision["status"], "canonical_state_gate_failed")
        self.assertFalse(decision["one_branch_exact_invariance_gate_pass"])
        self.assertTrue(decision["complete_safe_method_gate_pass"])


if __name__ == "__main__":
    unittest.main()
