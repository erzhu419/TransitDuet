#!/usr/bin/env python3
"""Materialize the user-approved, source/S-lock-bound E authorization only."""

import argparse
import json
from pathlib import Path

import freqduet_v263_evaluation_bank as bank


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    collector_source = 'e414070e27'
    assert 'source_e414070e27' in bank.__file__
    proof = read(a.results_root/'protocol_v6_v263_evaluation_bank/synthetic_e414070e27/summary.json')
    assert proof['source'] == collector_source and proof['status'] == 'v263_evaluation_collector_synthetic_passed'
    assert proof['failures'] == proof['errors'] == proof['real_evaluation_starts'] == 0
    seal = read(a.results_root/'protocol_v6_v262_selection_contrasts/sealed_S.json')
    locks = bank.choices_from_seal(seal)
    assert proof['locked_contrasts'] == locks and proof['S_seal_metadata_unchanged']
    root = a.results_root/bank.COLLECTION
    assert not list((root/'started').glob('*/*.json')), 'preserve any already-started E budget'
    root.mkdir(parents=True, exist_ok=True)
    authorization = bank.authorization(collector_source, locks)
    with (root/'evaluation_authorization.json').open('x') as stream:
        json.dump(authorization, stream, indent=2)
        stream.write('\n')
    receipt = dict(source=a.source, collector_source=collector_source,
        status='v264_E_authorized_first_E0_wave_only', S_seal_source=bank.S_SEAL_SOURCE,
        authorized_units=32, physical_start_cap=224, max_workers=8,
        first_wave=[bank.unit_key(bank.evaluation_unit(c,0)) for c in range(4)],
        first_wave_start_cap=28, remaining_units_held=28,
        release_gate='All four E0 full units pass prefix/future/execution and complete-grid checks; never gate on outcome quality.',
        real_E_starts=0, selection_recomputed=False, training_updates=0,
        main_policy_changed=False, v238_changed=False)
    out = a.results_root/'protocol_v6_v264_evaluation_collection/summary/authorization_receipt.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print('V264_E_AUTHORIZED '+json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
