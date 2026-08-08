import unittest

from freq_hrl.experiments.mujoco.responsibility_transfer_analysis import (
    responsibility_gate_decision,
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


class MujocoResponsibilityTransferAnalysisTest(unittest.TestCase):
    def test_both_registered_gates_must_pass(self):
        decision = responsibility_gate_decision(
            rows(one_branch=True, complete=True),
            environments=("A", "B", "C"),
        )
        self.assertEqual(decision["status"], "causal_transfer_gate_passed")
        self.assertTrue(decision["one_branch_structural_gate_pass"])
        self.assertTrue(decision["complete_safe_method_gate_pass"])

    def test_complete_method_failure_cannot_hide_behind_structural_gate(self):
        decision = responsibility_gate_decision(
            rows(one_branch=True, complete=False),
            environments=("A", "B", "C"),
        )
        self.assertEqual(decision["status"], "causal_transfer_gate_failed")
        self.assertTrue(decision["one_branch_structural_gate_pass"])
        self.assertFalse(decision["complete_safe_method_gate_pass"])


if __name__ == "__main__":
    unittest.main()
