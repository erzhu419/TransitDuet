#!/usr/bin/env python3
"""Revise V237 missing-label totals from its small summary; no raw reads."""

import argparse
import json
from pathlib import Path
import unittest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-summary', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v237_persistent_supply.py')
    assert suite.countTestCases() == 7
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    summary = json.loads(a.analysis_summary.read_text())
    assert summary['source'] == '1f71d9678a' and summary['status'] == 'v237_read_only_supply_evidence_complete'
    for path in summary['paths'].values():
        for name, group in path['unique_supply_cycles'].items():
            group['next_launch_labels'] = 0 if name == 'no_next_launch' else group['cycles']
            if name == 'no_next_launch':
                assert group['cycles'] == 12 and group['next_public_delay_s'] == group['postreturn_wait_after_native_tick_s'] == 0
                group['next_public_delay_s'] = None
                group['postreturn_wait_after_native_tick_s'] = None
    summary.update(source=a.source, raw_analysis_source='1f71d9678a', tests=7,
                   raw_analysis_tests=6, revision_raw_archive_reads=0)
    out = a.out_root/'summary'
    out.mkdir(parents=True, exist_ok=False)
    (out/'status.json').write_text(json.dumps(summary, indent=2)+'\n')
    small = {k: v for k, v in summary.items() if k not in ('paths', 'scope')}
    small['paths'] = {}
    for arm, path in summary['paths'].items():
        own = path['first_own_due']
        small['paths'][arm] = dict(totals=path['totals'], own_due=path['pressure_groups']['own_due'],
            first_own_due=own,
            first_own_due_prefix_share=own['H_already_executed_s']/(own['H_already_executed_s']+own['later_holding_s']),
            unique_supply_cycles=path['unique_supply_cycles'],
            actual_write_intervals=path['cross_version']['groups']['actual_write'],
            multi_positive_message_versions=path['multi_positive_message_versions'],
            predicted_local_nonharm_cuts=path['predicted_local_nonharm_cuts'],
            return_bound_fixed_neighbour_nonbenefit_holding_s=sum(g['holding_s'] for key, g in
                path['return_bound_after_plan_evidence'].items() if key.endswith('/nonbenefit')),
            endpoints=path['endpoints'])
    (out/'compact.json').write_text(json.dumps(small, indent=2)+'\n')
    print('V237_SUMMARY_ONLY_REVISION_COMPLETE '+json.dumps(dict(source=a.source, tests=7,
        raw_archive_reads=0, summary_path=str(out/'status.json'),
        compact_bytes=(out/'compact.json').stat().st_size)), flush=True)


if __name__ == '__main__':
    main()
