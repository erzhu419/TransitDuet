#!/usr/bin/env python3
"""Seal the V238 versus supply-blind matched-future pilot; no replay entrypoint."""

import argparse
import json
from pathlib import Path
import re
import unittest

from preregister_freqduet_v238_persistent_requests import ACTIONS, registration as frozen_rule


CONTEXTS = ((12037, 2151001), (12037, 2151002), (933083, 2151001), (933083, 2151002))
ARMS = (('v107', 'native', 'native'), ('upper_only', 'v220', 'native'),
        ('lower_supply', 'native', 'v238'), ('joint_supply', 'v220', 'v238'),
        ('lower_blind', 'native', 'blind'), ('joint_blind', 'v220', 'blind'))
CONSTRAINTS = ('cv_squared', 'readiness_delay_s', 'public_launch_delay_total_s',
               'trips_unlaunched', 'trips_incomplete')
SMOKE_SEED = 2490000000


def blind_request(holding, frontier):
    """Same H response/mask/compensation,with NO plan or association input."""
    assert holding >= 0
    native = frontier['native_bin']
    candidates = {c['bin']: c for c in frontier['candidates']}
    assert native in candidates and len(candidates) == len(frontier['candidates'])
    assert all(c['holding_s'] == ACTIONS[b] for b, c in candidates.items())
    base = ACTIONS[native]
    out = dict(native_bin=native, shadow_bin=native, shadow_s=base, target_s=None,
        compensation_floor_s=None, local_gap_delta_vs_native_s2=None, local_gain_vs_zero_s2=None)
    if frontier['status'] != 'supported':
        return dict(out, reason=frontier['status'])
    if base == 0:
        return dict(out, reason='native_zero')
    if 0 not in candidates:
        return dict(out, reason='no_native_feasible_zero_reference')
    zero = candidates[0]['local_gap_cost_delta_s2']
    assert zero is not None and candidates[native]['local_gap_cost_delta_s2'] == 0
    if zero <= 0:
        return dict(out, reason='native_lacks_positive_bilateral_gain_vs_zero')
    eligible = [c for c in candidates.values() if 0 < c['holding_s'] <= base
                and c['local_gap_cost_delta_s2'] < zero]
    target = base/(1+holding/ACTIONS[-1])
    chosen = min(eligible, key=lambda c: (abs(c['holding_s']-target), -c['holding_s']))
    return dict(out, shadow_bin=chosen['bin'], shadow_s=chosen['holding_s'], target_s=target,
        compensation_floor_s=min(c['holding_s'] for c in eligible),
        local_gap_delta_vs_native_s2=chosen['local_gap_cost_delta_s2'],
        local_gain_vs_zero_s2=zero-chosen['local_gap_cost_delta_s2'],
        reason='blind_persistent_compensation_request')


def future_seed(context, future):
    assert context in range(4) and future in range(4)
    return 2400000000+100*context+future


def reserved_slots():
    slots = [dict(slot='passive_c0_'+arm, phase='passive', context=0, arm=arm,
                  future_seed=None) for arm in ('v107', 'upper_only')]
    slots += [dict(slot='smoke_c0_'+arm, phase='smoke', context=0, arm=arm,
                   future_seed=SMOKE_SEED) for arm, _, _ in ARMS]
    slots += [dict(slot=f'c{c}_f{f}_{arm}', phase='formal', context=c, arm=arm,
                   future_seed=future_seed(c, f), evaluation_half='A' if f < 2 else 'B')
              for c in range(4) for f in range(4) for arm, _, _ in ARMS]
    return slots


def joint_gate(intervals):
    return intervals['passenger_person_s']['ci95'][1] < 0 and all(
        intervals[k]['ci95'][1] <= 0 for k in CONSTRAINTS)


def registration(source):
    return dict(stage='v240_frozen_supply_vs_blind_paired_future_preregistration', source=source,
        rule_source='b4b1d00203', audit_source='cf90cfe637', cycle_source='7fdf968a5f',
        controller_source='2b0115af7b', passive_capture_source='6c3397ff7f',
        contexts=[dict(context=i, actor=a, demand_day=d, prefix_sampler=187000000+a+100*d)
                  for i, (a, d) in enumerate(CONTEXTS)],
        arms=[dict(name=n, upper=u, lower=l) for n, u, l in ARMS],
        H0_aliases=dict(native_upper='v107', v220_upper='upper_only'),
        frozen='Original12 fixed-pool vehicles and seven0/5/10/15/20/30/45s bins,all masks,'
            'actors/critics/duals,predictors/calibrations including V151/V146,windows,'
            'quantiles,clearing horizon and main unchanged. V238 law/45s scale/floor/gate '
            'remain EXACTLY source b4b1d00203;no gain search or threshold fitting.',
        upper='Native or frozen V220 current-ready-backed native+30 preview,clipped[-60,60],'
            'EMA.35,one shared scalar,900s native fresh cadence,immutable anchors and '
            'rolling-zero-sum constraints,direction-local caches. Native commands/caches '
            'stay native;only the admitted projected curve is applied. Incoming buses '
            'never fund advances;no added delay,terminal control or invented slack.',
        lower='Native bilateral025 or exact V238,or blind_request(H,frontier). Blind uses '
            'the SAME h_native/(1+H/45),positive R(a)<R(0) candidates,feasible mask,ties '
            'toward larger a and unsupported fallbacks,but takes NO plan/position/association '
            'input. Implement it directly,not by fabricating an obligation for V238. '
            'Start from V107/upper-only,NEVER stack the old V219 lower or V223 phase rule.',
        persistence='Every branch recomputes its own native bilateral request/frontier and '
            'current V232 resource view from its REAL closed-loop state at each query. '
            'H is actual executed holding since observed launch,including before the fork. '
            'No reset at anchor,replan,rank/association change or relief,no hypothetical '
            'savings feedback and no replayed action tape. Only observed new launch resets H.',
        anchor='First native FRESH upper callback at t>=7200s,inside its action hook AFTER '
            'the common native coefficient was sampled and BEFORE any modified plan is '
            'applied. Match pre-command physical state,plans,versions,H ledgers,prepared '
            'context,native action and event/RNG prefix EXACTLY across arms. Fork only '
            'after this match;reuse this sampled coefficient,do not call the actor again. '
            'Missing anchor stops the pilot without replacing context or finding a later winner.',
        futures=dict(formal=[dict(context=c, future=f, seed=future_seed(c, f),
            half='A' if f < 2 else 'B') for c in range(4) for f in range(4)], smoke_seed=SMOKE_SEED,
            selection_bank=None, mode='All six rules fixed;no winner selection or per-context fallback.'),
        coupling='Reuse frozen V68 fork_future and V211/V69 event-keyed uniform coupling. '
            'Same future root and derived streams across arms WITHIN a context/future, '
            'different roots across units/smoke. Preserve already sampled current-window '
            'arrivals/road and day latent profiles. Log common stream keys and unmatched '
            'branch-only events without forcing neighbour,return or assignment matches. '
            'Independent future innovations are NOT independent new demand days.',
        intervention='From the first lower query AFTER anchor through original service clearing, '
            'use the registered branch rule;V220 acts only at native fresh callbacks. '
            'Future actual neighbours/returns/first-next-launches and V239 outcome labels '
            'never gate. Do not reuse V239 request tape;do not choose a subset of successful stops.',
        preflight='Later authorization only:two c0 passive no-op paths versus accepted V236 '
            'V107/upper references must preserve original actions,probabilities,inputs,uniforms,'
            'event order,physical clocks,plans,endpoints and old sidecars EXACTLY;recording '
            'adds no native calls/RNG draws. Test H=0 aliases there without extra paths. '
            'Then six c0 smoke paths on the separate smoke future. Require exact common '
            'prefix/fork and authority/conservation,legal ACTUALLY executed changes in each '
            'active lower arm,an actual ready-backed plan/launch advance in EACH of '
            'upper_only-v107,joint_supply-lower_supply,joint_blind-lower_blind,and at least '
            'one supply-versus-blind difference on the SAME observed H/frontier. No support '
            'means stop,not more gain/contexts. Smoke outcomes never tune or select rules.',
        primary_contrasts=['joint_supply-v107', 'joint_supply-upper_only',
                           'joint_supply-lower_supply', 'joint_supply-joint_blind'],
        constraints=list(CONSTRAINTS),
        mechanism_contrasts=['lower_supply-lower_blind', 'lower_blind-v107',
            'joint_blind-upper_only', '(joint_supply-upper_only)-(lower_supply-v107)',
            '(joint_supply-joint_blind)-(lower_supply-lower_blind)'],
        outcomes='Full-horizon waiting+onboard passenger_person_s including censoring; '
            'journey_min over ALL generated demand;frozen BOTH-direction actual arrival '
            'CV2 population;mean and total actual-minus-first-eligible readiness delay; '
            'SUM max(actual_launch-ORIGINAL public launch,0),including control waiting; '
            'unlaunched/incomplete counts. Missing launches have no zero-imputed delay; '
            'uncleared service retains its outcomes and blocks admission.',
        diagnostics='Actual issued/executed holding,passenger waiting/onboard split,early '
            'departures,query masks/native requests/selected bins,H,compensation loss/floor '
            'saturation,resource context and unchanged-state dual shadow requests. SAME '
            'formula is NOT matched realized holding dose:report dose differences,never '
            'posthoc filter/reweight/match on realized holding. Future labels are evaluation '
            'only. Count each supply cycle once;retain changed assignments/no-next labels.',
        statistics='Pair all six arms within context/future. Report each unit,actor,day and '
            'both fixed halves A=f0,f1 / B=f2,f3. Percentile95% CI with5000 draws,seed240: '
            '(1)crossed actor/day resampling,then paired future resampling within context; '
            '(2)conditional-future resampling within each fixed context. Use identical '
            'draws for all arms/metrics/contrasts. No query-level inference or winner search.',
        gate='For EACH primary contrast require passenger95%CI upper<0 and ALL constraint '
            'CI uppers<=0 under BOTH intervals,plus strict passenger improvement/nonharm '
            'means in EACH evaluation half. No relaxed margin,composite tradeoff,optional '
            'stopping or adaptive future expansion. Supply attribution additionally needs '
            'joint_supply versus joint_blind;a generic holding improvement is not coordination. '
            'An interaction claim requires a negative passenger95%CI upper for the final '
            'difference-in-differences under both intervals,not merely a better joint mean.',
        physical_mechanism='In BOTH formal halves require a delivered upper plan AND actual '
            'launch advance versus lower_supply,plus actual lower holding changes. To claim '
            'a physical supply mechanism,show the SAME observed inbound vehicle return '
            'followed by its FIRST next actual launch serving a query-recorded obligation; '
            'compare clocks/identity to controls. Retain unmatched/nonbeneficial chains,do '
            'not select successful paths or use these labels at runtime. Absent execution '
            'or mechanism blocks that claim even if passenger means improve.',
        reserved_budget=dict(passive_max=2, smoke_max=6, formal_max=96, total_starts_max=104,
            additional_references=0, max_workers=8, launch_authorized=False,
            failure_rule='Each started attempt consumes its fixed slot,including failures. '
                'No automatic retry,replacement,extra reference or budget extension. Formal '
                'opens only after BOTH passive paths and ALL smoke engineering gates pass; '
                'then run every fixed formal unit/arm without selecting on first-half outcomes.'),
        current_budget=dict(raw_archive_reads=0, archived_shadow_requests=0, new_replays=0,
            future_starts=0, training_updates=0, model_fits=0, shadow_actions_executed=0),
        limitations='Two actors by two consumed demand-day prefixes with four futures each '
            'is a small conditional mechanism pilot,not new-day efficacy confirmation. '
            'Crossed intervals have only two clusters on each axis. Even passage admits '
            'only a separate fresh-day/full-day confirmation with strong fixed+bilateral '
            'and unchanged final gates;no fixed superiority or frequency-separation claim.',
        execution='Scheduler high,explicit manual dispatch,node001--006. Tests/seal only '
            'now,small summary/log locally;no runtime imports,controller attachment or replay entrypoint.',
        runtime_connected=False, replay_authorized=False, training_authorized=False,
        main_policy_changed=False)


def namespace_audit(root, snapshot):
    seeds = {future_seed(c, f) for c in range(4) for f in range(4)} | {SMOKE_SEED}
    assert snapshot['namespace'] == 'FreqDuet/v240' and snapshot['conflicts'] == []
    checked, conflicts = 0, []

    def contains(value):
        if isinstance(value, dict):
            return any(contains(v) for v in value.values())
        if isinstance(value, list):
            return any(contains(v) for v in value)
        if isinstance(value, int):
            return value in seeds
        if isinstance(value, str):
            return bool(seeds & {int(t) for t in re.findall(r'\d+', value)})
        return False

    for stage in sorted(root.glob('protocol_v6_*')):
        if not stage.is_dir():
            continue
        for name in ('protocol.json', 'registration.json', 'manifest.json'):
            for path in [stage/name, *stage.glob('*/'+name)]:
                if path.is_file():
                    checked += 1
                    if contains(json.loads(path.read_text())):
                        conflicts.append(str(path))
    assert checked > 0 and not conflicts, ('registered future namespace conflict', conflicts)
    return dict(registered_metadata_files=checked, conflicts=conflicts, reserved_future_roots=len(seeds),
        scheduler_snapshot=snapshot,
        scope='Known FreqDuet canonical protocol/registration/manifest metadata at root/one child '
            'level plus scheduler task metadata. No trajectory,checkpoint or raw-cache scan.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out-root', 'results-root', 'parent-summary', 'sealed-rule', 'scheduler-snapshot'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source', required=True)
    a = p.parse_args()
    spec = registration(a.source)
    status = a.out_root/'summary/status.json'
    if status.exists():
        assert json.loads((a.out_root/'protocol.json').read_text()) == spec
        assert json.loads(status.read_text())['status'] == 'v240_paired_protocol_sealed_no_replay'
        print('V240_EXISTING_SEAL_REUSED', flush=True)
        return
    assert json.loads(a.sealed_rule.read_text()) == frozen_rule('b4b1d00203')
    parent = json.loads(a.parent_summary.read_text())
    assert parent['source'] == 'cf90cfe637' and parent['status'] == 'v239_shadow_audit_complete'
    assert parent['nonzero_shadow_support_and_differences'] and not parent['main_policy_changed']
    tested = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1]/'tests'), pattern='test_v240_paired_coordination.py'))
    assert tested.testsRun > 0 and tested.wasSuccessful()
    audit = namespace_audit(a.results_root, json.loads(a.scheduler_snapshot.read_text()))
    a.out_root.mkdir(parents=True, exist_ok=False)
    (a.out_root/'protocol.json').write_text(json.dumps(spec, indent=2)+'\n')
    (a.out_root/'reserved_slots.json').write_text(json.dumps(reserved_slots(), indent=2)+'\n')
    summary = dict(status='v240_paired_protocol_sealed_no_replay', source=a.source,
        rule_source='b4b1d00203', tests=tested.testsRun, namespace_audit=audit,
        contexts=spec['contexts'], arms=spec['arms'], reserved_budget=spec['reserved_budget'],
        current_budget=spec['current_budget'], runtime_connected=False, replay_authorized=False,
        training_authorized=False, main_policy_changed=False,
        protocol_path=str(a.out_root/'protocol.json'),
        next_step='Seek separate authorization to implement the passive/no-op integration preflight; '
            'no simulator start or future opening is admitted by this seal.')
    status.parent.mkdir()
    status.write_text(json.dumps(summary, indent=2)+'\n')
    print('V240_SEALED_NO_REPLAY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
