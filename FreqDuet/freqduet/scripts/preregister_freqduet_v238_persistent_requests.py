#!/usr/bin/env python3
"""Seal an unconnected persistent shadow rule; synthetic inputs only."""

import argparse
import json
from pathlib import Path
import unittest


ACTIONS = (0, 5, 10, 15, 20, 30, 45)


def shadow_request(view, row):
    """Use the frozen causal cycle view and native seven-bin frontier only."""
    assert view['cycle_key'] == row['cycle_ref']
    assert view['query_s'] == row['snapshot_read_s']
    assert view['native_event_seq'] == row['snapshot_event_seq']
    holding = view['executed_holding_s']
    assert holding == row['executed_holding_before_s'] and holding >= 0
    frontier = row['shadow_frontier']
    native = frontier['native_bin']
    candidates = frontier['candidates']
    by_bin = {c['bin']: c for c in candidates}
    assert native in by_bin and len(by_bin) == len(candidates)
    assert all(c['holding_s'] == ACTIONS[c['bin']] for c in candidates)
    base = ACTIONS[native]
    association = view['own_association']
    assert association['bus_id'] == view['cycle_key'][0]
    assert association['inbound_trip_id'] == view['cycle_key'][1]
    obligation = association['obligation']
    out = dict(native_bin=native, shadow_bin=native, native_s=base, shadow_s=base,
        cycle_key=list(view['cycle_key']), executed_holding_s=holding,
        obligation_trip=None if obligation is None else obligation['trip_id'],
        exposure='none' if obligation is None else obligation['kind'],
        target_s=None, compensation_floor_s=None, admissible_bins=[],
        local_gap_delta_vs_native_s2=None, local_gain_vs_zero_s2=None,
        executed=False)

    def unchanged(reason):
        return dict(out, reason=reason)

    if frontier['status'] != 'supported':
        return unchanged(frontier['status'])
    if base == 0:
        return unchanged('native_zero')
    if obligation is None:
        return unchanged('no_own_unbacked_obligation')
    # Future unbacked is exposure, not a known shortage or an ETA/slack estimate.
    assert obligation['kind'] in ('due', 'future')
    if 0 not in by_bin:
        return unchanged('no_native_feasible_zero_reference')
    zero_delta = by_bin[0]['local_gap_cost_delta_s2']
    assert zero_delta is not None and by_bin[native]['local_gap_cost_delta_s2'] == 0
    if zero_delta <= 0:
        return unchanged('native_lacks_positive_bilateral_gain_vs_zero')
    eligible = [c for c in candidates if 0 < c['holding_s'] <= base
                and c['local_gap_cost_delta_s2'] < zero_delta]
    target = base/(1+holding/ACTIONS[-1])
    # At a quantization tie retain more compensation, not a forced lower bin.
    selected = min(eligible, key=lambda c: (abs(c['holding_s']-target), -c['holding_s']))
    return dict(out, shadow_bin=selected['bin'], shadow_s=selected['holding_s'],
        reason='persistent_compensation_request', target_s=target,
        compensation_floor_s=min(c['holding_s'] for c in eligible),
        admissible_bins=sorted(c['bin'] for c in eligible),
        local_gap_delta_vs_native_s2=selected['local_gap_cost_delta_s2'],
        local_gain_vs_zero_s2=zero_delta-selected['local_gap_cost_delta_s2'])


def registration(source):
    return dict(stage='v238_persistent_compensation_shadow_preregistration', source=source,
        parent_source='34af3ecc7b', cycle_source='7fdf968a5f', capture_source='6c3397ff7f',
        authorization='Protocol seal and synthetic reference tests only;no archive shadow requests.',
        budget=dict(archive_scans=0, archived_shadow_requests=0, passive_paths=0,
            intervention_paths=0, training_updates=0, model_fits=0, new_futures=0),
        frozen=dict(base='V107 bilateral025', fleet=12, feasible_action_bins_s=list(ACTIONS),
            upper='Unchanged V220 current-ready-backed native projection:one shared scalar,'
                'original bounds,EMA.35,900s cadence,anchors,rolling-zero-sum and direction-local caches.',
            other='Actors,critics,duals,all predictors/calibrations including V151/V146,'
                'windows,quantiles,main and native action masks unchanged.'),
        hypothesis='Conditional,cycle-persistent moderation may preserve positive local '
            'headway compensation while reducing holding exposure on a currently associated '
            'unbacked service obligation. It is not a removable-holding classifier.',
        inputs='Frozen V232 query view from current published plans/versions,current ready '
            'stock and incoming positions plus executed H;frozen V230 feasible frontier using '
            'unchanged native PREPARE F/B evidence and its original clock. Join the exact '
            'query occurrence and physical cycle. No new native context/ETA call or resampling.',
        exposure='Use V232 one-use ready FIFO allocation and position-ordinal incoming '
            'association unchanged. Own associated unbacked due OR future obligation activates '
            'the shadow branch before own-due appears. Future without ready is exposure,NOT '
            'a proven shortage. Association is neither assignment nor reservation. No own '
            'obligation keeps native. Other buses holding/pressure is not charged to this cycle.',
        ledger='Physical key=(bus,trip,observed actual launch event). Read actual executed H '
            'before query;never sum repeated prefix samples. Only real execution increments '
            'H. Shadow requests never update it,spend saved seconds or forecast a new H. '
            'Replans,cache callbacks,station/rank/association changes and pressure relief '
            'cannot reset or refund H. Observed return closes the old cycle;only an actual '
            'new launch opens H=0. Keep open/no-next-launch states explicit.',
        rule=dict(target='x=h_native/(1+H/45) seconds;h_native is the actual quantized '
                'native request at this query. H=0 is exactly native.',
            scale='45s is solely the existing maximum single holding command,not a measured '
                'slack,budget,ETA,available turnaround or fitted coefficient.',
            compensation='R(a)=(F+a-target_headway)^2+(B-a-target_headway)^2. '
                'Admit native-feasible 0<a<=h_native with R(a)<R(0);select nearest to x, '
                'ties toward larger a. The nonzero native request must itself have R(native)<R(0). '
                'Use the archived frontier differences;H is NEVER subtracted from F/B or '
                'added to R a second time. No R(a)<=R(native) gate.',
            fallback='Unsupported/terminal/missing-forward/missing-follower,native zero,no own '
                'obligation,no native-feasible zero reference or nonbeneficial native:retain native. '
                'Positive beneficial native remains in the candidate set;positive requests '
                'cannot be driven to zero. No forced one-bin cut or unconditional attenuation.',
            permissions='All seven native feasible commands remain environment permissions. '
                'This single shadow candidate never increases native holding. Enumerate the '
                'unaltered full frontier separately;no new gain,upper command or dispatch.'),
        compensation_scope='Positive local predicted improvement relative to zero is retained, '
            'NOT the entire native improvement and NOT actual CV2 nonharm. R(a)-R(native) '
            'may be positive and must be reported. Neither native nor zero is an observed '
            'counterfactual outcome. Retaining native on unsupported rows is explicit abstention.',
        forbidden='Actual future neighbour,return time,first-next-launch,real supply identity,'
            'outcome labels,late load,hidden demand/road draws,unfinished prefixes and '
            'ex-post beneficial/harmful labels cannot gate or choose a request. Do not use '
            'published lead or observed terminal waiting as fungible slack.',
        current_work='No trajectory is opened,no real query gets a new shadow request,and '
            'no live observer/controller is imported or attached. Synthetic calls test the '
            'formula,mask,compensation,ties,clocks and physical-cycle persistence only.',
        next_gate='Separately authorize a single read-only shadow audit on the TWO accepted '
            'V236 paths,all queries retained. Freeze this formula before that audit. Count '
            'support,changed bins/seconds,compensation loss vs native,resource kinds and H '
            'bands;join future supply only AFTER requests are sealed for outcome diagnostics. '
            'Compare identical current views with the memoryless H=0 ablation. No paired '
            'efficacy claim or automatic replay budget;zero support/differences stops '
            'progression rather than tuning scale,gate,floor or paths on those trajectories.',
        efficacy_gate='A later separately budgeted passive equivalence and matched-future '
            'trial must retain V107,upper-only,lower-only,joint and a memoryless control, '
            'original fleet/permissions and strong fixed+bilateral. Freeze choices before '
            'independent selection/evaluation futures. Joint passenger95%CI upper<0 AND '
            'actual CV2/readiness/original-public total launch delay/incomplete/unlaunched '
            'CI uppers<=0 versus V107 and EACH single-layer comparator,actor/day crossed '
            'paired statistics;do not count queries as independent samples.',
        limitations='This low-capacity law is a frozen hypothesis,not a validated controller. '
            'H is total exposure,not proven excess holding;distance association may change. '
            'Native fallback after relief may create later rebound. A positive local gain '
            'does not identify necessary compensation or guarantee full-day service gains.',
        runtime_connected=False, archived_shadow_evaluated=False, shadow_actions_executed=0,
        replay_authorized=False, training_authorized=False, main_policy_changed=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--parent-summary', type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    spec = registration(a.source)
    path = a.out_root/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == spec
        prior = json.loads((a.out_root/'summary/status.json').read_text())
        assert prior['status'] == 'v238_shadow_protocol_sealed'
        print('V238_EXISTING_SEAL_REUSED_NO_NEW_WORK', flush=True)
        return
    parent = json.loads(a.parent_summary.read_text())
    assert parent['source'] == spec['parent_source']
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parents[1]/'tests'),
                                              pattern='test_v238_persistent_requests.py')
    assert suite.countTestCases() > 0
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    a.out_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2)+'\n')
    summary = dict(status='v238_shadow_protocol_sealed', source=a.source,
        parent_source=parent['source'], tests=result.testsRun, budget=spec['budget'],
        protocol_path=str(path), runtime_connected=False, archived_shadow_evaluated=False,
        shadow_actions_executed=0, main_policy_changed=False, next_step=spec['next_gate'])
    (a.out_root/'summary').mkdir(exist_ok=True)
    (a.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V238_SHADOW_DESIGN_ONLY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
