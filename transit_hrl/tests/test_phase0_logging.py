import tempfile
import unittest
from pathlib import Path

from freq_hrl.core import (
    PHASE0_SCHEMA_VERSION,
    Phase0TraceLogger,
    load_phase0_records,
    validate_phase0_record_schema,
)


class Phase0LoggingTest(unittest.TestCase):
    def test_phase0_logger_writes_required_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.jsonl"
            with Phase0TraceLogger(path) as logger:
                logger.write({
                    "t": 0,
                    "domain": "test",
                    "entity_id": "entity",
                    "x_raw": [1.0],
                    "x_bin": {"timestamp": 0.0, "entity_id": "entity", "x_raw": [1.0]},
                    "z_t": {"x_low": [1.0]},
                    "a_U": [0.0],
                    "a_L": {"execution_speed": [1.0]},
                    "plan_curve": {"type": "constant", "target": [0.0]},
                    "action_effects": {"upper": [0.0], "lower": [0.0]},
                    "reward": {"task_reward": 0.0},
                })
            records = load_phase0_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["schema_version"], PHASE0_SCHEMA_VERSION)
            validate_phase0_record_schema(records[0])


if __name__ == "__main__":
    unittest.main()
