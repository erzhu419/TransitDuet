import ast
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v166_fresh_capture as capture
from test_v165_fresh_calibration import inputs


def protocols():
    values = inputs()
    plan = capture.fresh.registration(*values)
    plan['execution_source'] = '4a6d783667'
    return plan, values[0]


class FreshCaptureTests(unittest.TestCase):
    def test_frozen_method_and_complete_64_day_assignment(self):
        plan, parent = protocols()
        before = deepcopy(plan)
        capture.validate_protocol(plan, parent)
        cases = capture.capture.calibration_cases(plan)
        self.assertEqual(len(cases), 768)
        self.assertEqual({d for _, d in cases}, set(capture.fresh.CALIBRATION))
        self.assertTrue({d for _, d in cases}.isdisjoint(plan['reservation']['confirmation_days']))
        self.assertTrue({d for _, d in cases}.isdisjoint(plan['reservation']['excluded_development_days']))
        self.assertEqual(plan, before)

    def test_no_confirmation_or_development_replacement(self):
        for field in ('confirmation_days', 'excluded_development_days'):
            plan, parent = protocols()
            plan['reservation']['calibration_days'][0] = plan['reservation'][field][0]
            with self.assertRaisesRegex(AssertionError, 'cohort changed'):
                capture.validate_protocol(plan, parent)

    def test_actor_sampler_and_method_changes_block_collection(self):
        for field, value in (('actors', [1]*12), ('sampler_seeds', {}), ('checkpoint_ep', 39)):
            plan, parent = protocols()
            plan['reservation'][field] = value
            with self.assertRaisesRegex(AssertionError, 'assignment changed'):
                capture.validate_protocol(plan, parent)
        plan, parent = protocols()
        plan['support']['minimum_calibration_days'] = 18
        with self.assertRaisesRegex(AssertionError, 'method changed'):
            capture.validate_protocol(plan, parent)

    def test_entry_requires_complete_same_source_preflight(self):
        proof = dict(status='v166_passive_equivalence_passed', execution_source='test',
            protocol_source='4a6d783667', **{k: True for k in capture.EQUIVALENCE})
        args = SimpleNamespace(out_root=Path('/unused'))
        with patch.dict(os.environ, FREQDUET_DIAGNOSTIC_COMMIT='test'), patch.object(capture, 'read', return_value=proof):
            capture.require_preflight(args)
            proof['execution_source'] = 'older'
            with self.assertRaisesRegex(AssertionError, 'same-source'):
                capture.require_preflight(args)
            proof['execution_source'] = 'test'
            proof[capture.EQUIVALENCE[1]] = False
            with self.assertRaisesRegex(AssertionError, 'same-source'):
                capture.require_preflight(args)

    def test_packet_comparison_detects_changed_state_clock_and_missing_row(self):
        rows = [dict(state=dict(speed=5.), query=10., label=20.)]
        self.assertEqual(capture.exact_stream(iter(rows), iter(deepcopy(rows)), 'query'), 1)
        for broken in ([], [dict(state=dict(speed=6.), query=10., label=20.)],
                [dict(state=dict(speed=5.), query=11., label=20.)]):
            with self.assertRaisesRegex(AssertionError, 'differs'):
                capture.exact_stream(iter(rows), iter(broken), 'query')

    def test_new_entry_retains_query_before_label_pass(self):
        tree = ast.parse(Path(capture.__file__).read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        calls = {n.func.attr: n.lineno for n in ast.walk(functions['capture_case'])
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertLess(calls['save_queries'], calls['attach_labels'])
        self.assertIn('require_preflight(args)', ast.unparse(functions['pair_case']))
        self.assertIn('require_preflight(args)', ast.unparse(functions['run_batch']))


if __name__ == '__main__':
    unittest.main()
