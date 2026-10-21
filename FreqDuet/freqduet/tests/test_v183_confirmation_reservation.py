import ast
from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import reserve_freqduet_v183_confirmation as plan


class ConfirmationReservationTests(unittest.TestCase):
    def test_exact_independent_cohort(self):
        self.assertEqual(len(set(plan.DAYS)), 32)
        self.assertEqual(len(set(plan.SAMPLERS.values())), 32)
        self.assertFalse(set(plan.DAYS) & set(plan.ledger.CALIBRATION))
        self.assertFalse(set(plan.DAYS) & set(range(3352001, 3352065)))

    def test_sampler_collision_without_day_collision(self):
        result = plan.hits(dict(day=1, sampler_seed=plan.SAMPLERS[plan.DAYS[0]]))
        self.assertEqual(result['days'], [])
        self.assertEqual(len(result['samplers']), 1)

    def test_freshness_reads_headers_receipts_and_names_not_raw_outcomes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan.write(root/'old/registration.json', dict(confirmation_days=[plan.DAYS[0]]))
            plan.write(root/'old/paths/1_1_old.json', dict(day=1, sampler_seed=next(iter(plan.SAMPLERS.values()))))
            (root/'old/residuals').mkdir()
            (root/'old/residuals'/f'1_{plan.DAYS[1]}_old.jsonl.gz').write_bytes(b'not a readable trace')
            (root/'old/summary').mkdir()
            (root/'old/summary/status.json').write_text('not a readable outcome')
            (root/'generators').mkdir()
            (root/'generators/old.py').write_text('DAYS = range(1, 3)\n')
            result = plan.audit([root], [root/'generators'], dict(day_references={}, groups=[]), root/'own')
            self.assertEqual(result['decision'], 'blocked')
            self.assertEqual(len(result['collisions']), 3)
            self.assertEqual(result['errors'], [])

    def test_metadata_errors_and_missing_roots_block_freshness(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'protocol.json').write_text('invalid metadata')
            result = plan.audit([root, root/'missing'], [], dict(day_references={}, groups=[]), root/'own')
            self.assertEqual(result['decision'], 'blocked')
            self.assertEqual(len(result['errors']), 2)

    def test_literal_range_and_inventory_conflicts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan.write(root/'protocol.json', {})
            (root/'old.py').write_text(f'DAYS = range({plan.DAYS[0]}, {plan.DAYS[0]+2})\n')
            result = plan.audit([root], [root], dict(day_references={str(plan.DAYS[-1]): []}, groups=[]), root/'own')
            self.assertEqual(len(result['collisions']), 2)
            self.assertEqual(result['collisions'][1]['days'], list(plan.DAYS[:2]))

    def test_registration_preserves_full_frozen_protocol(self):
        protocol = dict(actors=list(range(12)), modes=['old', 'handoff'], confirmation=dict(paths=768,
            fixed_budget='one cohort'), execution=dict(interval_fit=False, confirmation_started=False),
            inference=dict(gate=dict(minimum_days=19)), future_estimator=dict(authorized_now=False))
        original = deepcopy(protocol)
        result = plan.reservation(protocol, dict(decision='fresh', collisions=[], errors=[]))
        self.assertEqual(result['v182_protocol'], original)
        self.assertEqual(protocol, original)
        self.assertFalse(any(result['execution'].values()))
        result['v182_protocol']['inference']['gate']['minimum_days'] = 0
        self.assertEqual(protocol, original)
        with self.assertRaises(AssertionError):
            plan.reservation(protocol, dict(decision='blocked', collisions=[1], errors=[]))

    def test_observed_archived_description_repair_does_not_change_declarations(self):
        broken = "CONFIRMATION = range(3152001, 3152033)\ndef registration():\n    result['estimator']['implementation'] = 'same V161 empirical ranks and joint atoms; future mixtures must recompute '\n        'equal-day/episode/query weights within each new key, never reuse old-key atom weights'\n"
        with self.assertRaises(SyntaxError):
            ast.parse(broken)
        repaired = plan.repair_description_only(broken)
        parsed = ast.parse(repaired)
        self.assertEqual([n.value for n in parsed.body[0].value.args], [3152001, 3152033])
        with self.assertRaises(AssertionError):
            plan.repair_description_only('an unrelated syntax failure')


if __name__ == '__main__':
    unittest.main()
