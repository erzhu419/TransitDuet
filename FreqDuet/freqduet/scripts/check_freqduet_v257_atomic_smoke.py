#!/usr/bin/env python3
"""Frozen-runtime class binding and synthetic tests ONLY; no real smoke entrypoint."""

import argparse
import inspect
import json
import os
from pathlib import Path
import unittest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('v254-root','v256-root','out-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',required=True)
    a=p.parse_args()
    spec=json.loads((a.v254_root/'manifest.json').read_text())
    assert spec['source']=='eaba2513da' and spec['physical_start_caps']['smoke']==8
    roots=[r for r in spec['futures'] if r['phase']=='smoke']
    assert len(roots)==1 and roots[0]['context']==0 and roots[0]['root']==3540999999
    accepted=json.loads((a.v256_root/'summary/status.json').read_text())
    assert accepted['source']=='aa8616bcfb' and accepted['status']=='v256_repair_passive_pair_passed'
    assert accepted['started_paths']==3 and accepted['failed_starts']==1
    assert {r['slot'] for r in accepted['paths'] if r['passed']}=={'passive_c0_v107_repair1','noop_c0_v107'}
    import run_freqduet_v256_passive_pulse as passive
    import freqduet_v257_atomic_smoke as adapter
    assert 'source_aa8616bcfb' in passive.__file__
    frozen,pulse,old,cycles,requests=passive.runtime_interfaces()
    joint=old.legacy.passive.prior
    repeated,event=joint.repeated,old.legacy.pilot.event
    sealed=Path(joint.__file__).parent
    assert 'protocol_v6_v211_joint_resources_c45b2d0a93/source/scripts' in str(sealed), str(sealed)
    assert Path(repeated.__file__)==sealed/'run_freqduet_v68_repeated_continuations.py', repeated.__file__
    assert Path(event.__file__)==sealed/'run_freqduet_v69_event_coupling.py', event.__file__
    assert tuple(spec['runtime_stream_names'])==adapter.STREAMS
    base=pulse.recording_audit(frozen.observer.recording_audit(
        requests.recording_audit(cycles.recording_audit(old.RecordingAudit))),'observer_only')
    implementation=adapter.atomic_audit(base,repeated,event,joint)
    assert issubclass(implementation,base)
    primitives=dict(SAME=joint.pilot.joint.assert_same_state,RNG=joint.pilot.joint.rng_state,
                    REPEATED=repeated.fork_future,EVENT=event.alignment,RESOURCE=joint.resource_snapshot)
    for name,function in primitives.items():
        os.environ['FREQDUET_V257_'+name]=inspect.getsourcefile(function)
    from env.bus import Bus
    kernels={name:inspect.getsourcefile(getattr(Bus,name)) for name in ('_start_dwelling','_process_dwelling')}
    assert all('FreqDuet-v6-engineering-v9-runtime-optimized/FreqDuet/freqduet/env/bus.py' in v for v in kernels.values())
    a.out_root.mkdir(parents=True,exist_ok=False)
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                            pattern='test_v257_atomic_smoke.py')
    assert suite.countTestCases()>0
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary=dict(status='v257_atomic_smoke_synthetic_'+('passed' if result.wasSuccessful() else 'failed'),
        source=a.source,manifest_source='eaba2513da',passive_source='aa8616bcfb',adapter_source='9f4afff2f3',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        smoke_root_reserved=3540999999,max_smoke_starts=8,
        simulator_starts=0,future_starts=0,real_smoke_starts=0,formal_starts=0,training_updates=0,
        raw_payloads_opened=0,launch_authorized=False,runtime_connected=False,selector_defined=False,
        main_policy_changed=False,v238_changed=False,
        frozen_primitives={k:inspect.getsourcefile(v) for k,v in primitives.items()},holding_kernels=kernels,
        next_gate='Separate real-smoke approval and start ledger;native then forced-native alias '
            'then only legal nonalias fixed bins. No formal bank or policy promotion.')
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('V257_SYNTHETIC '+json.dumps(summary),flush=True)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__=='__main__': main()
