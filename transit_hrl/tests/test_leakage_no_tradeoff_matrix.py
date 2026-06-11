import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.leakage_no_tradeoff_matrix import build_leakage_matrix


def _check(metric: str, status: str, *, direction: str = "increase") -> dict:
    return {
        "check": f"native_real_freqhrl_vs_interval_{metric}",
        "metric": metric,
        "treatment": "native_real_freqhrl",
        "control": "native_real_interval",
        "direction": direction,
        "n_common": 12,
        "delta_mean": -0.1 if direction == "decrease" else 0.1,
        "delta_ci95_low": -0.2 if direction == "decrease" else 0.02,
        "delta_ci95_high": -0.02 if direction == "decrease" else 0.2,
        "improvement_mean": 0.1,
        "improvement_ci95_low": 0.02,
        "improvement_ci95_high": 0.2,
        "win_rate": 0.75,
        "status": status,
    }


def _write_summary(path: Path, checks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"paired_checks": checks}, indent=2), encoding="utf-8")


class LeakageNoTradeoffMatrixTest(unittest.TestCase):
    def test_adaptive_native_selector_picks_joint_noharm_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            v7 = root / "v7" / "summary.json"
            v6 = root / "v6" / "summary.json"
            _write_summary(v7, [
                _check("LowerLFDrift", "supported", direction="decrease"),
                _check("control_score", "supported"),
                _check("ep_reward", "supported"),
                _check("avg_wait_min", "noninferiority_supported", direction="decrease"),
                _check("native_avg_board_wait_min", "noninferiority_supported", direction="decrease"),
                _check("native_alighted_pax", "noninferiority_supported"),
                _check("native_completed_throughput_pax", "noninferiority_supported"),
            ])
            _write_summary(v6, [
                _check("LowerLFDrift", "supported", direction="decrease"),
                _check("control_score", "supported"),
                _check("ep_reward", "supported"),
                _check("avg_wait_min", "not_supported", direction="decrease"),
            ])

            payload = build_leakage_matrix({
                "native_real_demand_service_response_v7": v7,
                "native_real_demand_throughput_safe_wait_v6": v6,
            }, min_pairs=5)

            selector = payload["adaptive_native_real_demand_selector"]
            self.assertEqual(selector["status"], "supported")
            self.assertEqual(selector["selected_domain"], "native_real_demand_service_response_v7")
            self.assertFalse(selector["strict_supported"])
            self.assertIn("avg_wait_min", selector["selected_strict_blocking_performance_metrics"])

    def test_adaptive_native_selector_blocks_when_no_profile_joint_passes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "blocked" / "summary.json"
            _write_summary(path, [
                _check("LowerLFDrift", "supported", direction="decrease"),
                _check("control_score", "supported"),
                _check("ep_reward", "supported"),
                _check("avg_wait_min", "not_supported", direction="decrease"),
                _check("native_avg_board_wait_min", "noninferiority_supported", direction="decrease"),
                _check("native_alighted_pax", "noninferiority_supported"),
                _check("native_completed_throughput_pax", "noninferiority_supported"),
            ])

            payload = build_leakage_matrix({
                "native_real_demand_service_response_v7": path,
            }, min_pairs=5)

            selector = payload["adaptive_native_real_demand_selector"]
            self.assertEqual(selector["status"], "blocked_no_native_no_tradeoff")
            self.assertFalse(selector["supported"])
            self.assertEqual(selector["selected_blocking_performance_metrics"], ["avg_wait_min"])


if __name__ == "__main__":
    unittest.main()
