"""Resolve two documented failed read-only artifacts, then reuse saved paths."""

import argparse
import json
from pathlib import Path
import traceback

from evaluation import HERE, read, write
from checks import compare_engineering
from preflight import day_hits


def review_metadata(args):
    original = read(args.root.parent/'preflight01/summary/results.json')
    plan = read(HERE/'manifest.json')
    metadata = original['metadata']
    gpu = original['gpu_inventory']
    queue = original['queue_inventory']
    assert original['completed'] and original['plan_checks_passed']
    assert metadata['roots_present'] == plan['metadata_roots']
    assert metadata['json_files_read'] > 0 and not metadata['collisions']
    assert gpu['days'] == queue['days'] == plan['days']
    assert gpu['metadata']['json_files_read'] > 0
    assert not gpu['metadata']['collisions'] and not gpu['metadata']['read_errors']
    assert not queue['collisions']

    history = Path(plan['metadata_roots'][0])
    probes = history/'protocol_v6_v279_response_alignment'
    expected = {str(probes/name/'results.json') for name in ('probe01', 'probe02')}
    assert {e['path'] for e in metadata['read_errors']} == expected
    assert all(e['error'] == 'JSONDecodeError' for e in metadata['read_errors'])
    provenance = read(HERE/'v279_completion.json')
    assert provenance['physical_starts'] == provenance['training_updates'] == 0
    assert provenance['engineering_failures'] == {
        't212187': 'numpy bool JSON serialization',
        't212188': 'numpy int64 JSON serialization'}
    assert all(t['status'] == 'failed' for t in provenance['tasks']
               if t['id'] in ('t212187', 't212188'))

    # These failed producers read the same eight archived roots as the completed
    # successor. The input manifest establishes their days without fixing the
    # truncated output or rerunning its producer.
    manifest_path = history/'protocol_v6_v278_action_diagnostic/manifest.json'
    archive = read(manifest_path)
    roots = archive['roots']
    expected_ids = {f'd{day}_w{window}' for day in range(2077001, 2077005)
                    for window in (0, 1)}
    assert len(roots) == 8 and {r['id'] for r in roots} == expected_ids
    assert all(r['status'] == 'selected' and r['day'] in range(2077001, 2077005)
               for r in roots)
    assert not day_hits(archive, set(plan['days']))
    successor_path = probes/'feedback01/summary/motion_probe.json'
    successor = read(successor_path)
    assert successor['status'] == 'read_only_initial_leg_probe_complete'
    assert successor['new_physical_paths'] == successor['training_updates'] == 0
    assert successor['selected_roots'] == 8
    assert successor['initial_travel_vehicles'] == len(successor['rows']) == 61
    assert {r['root_id'] for r in successor['rows']} == expected_ids
    resolved = []
    for path in sorted(expected):
        text = Path(path).read_text()
        try:
            json.loads(text)
        except json.JSONDecodeError as error:
            resolved.append(dict(path=path, bytes=Path(path).stat().st_size,
                error_line=error.lineno, error_column=error.colno,
                classification='documented aborted read-only V279 serialization'))
        else:
            raise AssertionError('Historical artifact changed: '+path)
    result = dict(completed=True, passed=True,
        freshness_verified_across_both_servers=True,
        original_preflight_preserved=True, original_json_files_read=metadata['json_files_read'],
        resolved_read_errors=resolved, unresolved_read_errors=[],
        input_manifest=str(manifest_path), completed_successor=str(successor_path),
        source_days=sorted({r['day'] for r in roots}), physical_starts=0,
        training_updates=0, main_policy_changed=False)
    write(args.root/'engineering_gate/summary/metadata_review.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = review_metadata(args)
        compare_engineering(args, metadata)
    except Exception:
        write(args.root/'engineering_gate/summary/review_failure.json', dict(
            completed=False, passed=False, physical_starts=0, training_updates=0,
            error=traceback.format_exc()[-5000:]))
        raise
