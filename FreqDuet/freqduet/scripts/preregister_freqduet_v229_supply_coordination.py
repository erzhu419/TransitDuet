#!/usr/bin/env python3
"""Seal a physical-cycle coordination study; never import or run a controller."""

import argparse
import ast
import json
from pathlib import Path
import unittest


ACTIONS = (0, 5, 10, 15, 20, 30, 45)
LIVE_FIELDS = ('time_s', 'event_seq', 'prepare_event_seq', 'state_captured_s',
               'ready', 'active', 'plans', 'onboard', 'executed_holding_before_s',
               'base_bin', 'feasible_bins', 'terminal_masked', 'context')


def local_gap_cost_change(forward, backward, candidate, native):
    """Change in the frozen one-step two-gap quadratic, not full-day CV2."""
    return 2*(candidate-native)*(candidate+native-(backward-forward))


def registration(source):
    return dict(stage='v229_supply_chain_persistent_coordination_design', source=source,
        authorization='Seal design,synthetic tests and source-interface audit ONLY.',
        budget=dict(archive_scans=0, passive_paths=0, intervention_paths=0, training_updates=0,
                    model_fits=0, new_future_paths=0),
        inherited=dict(base='V107 bilateral025', controller_source='f61f6c9734',
            development='V218--V228 consumed paths are development evidence,not confirmation.',
            fleet=12, holding_bins_s=list(ACTIONS), native_gain=.25,
            upper='Keep V220 current-ready-backed native projection,original coefficient '
                'bounds,EMA.35,900s cadence,anchors and rolling-zero-sum constraints. Shared '
                'coefficient does not confer simultaneous two-direction writes:callbacks, '
                'caches and EMA remain direction-local. No larger raw request or phase gain.',
            frozen='Actors,critics,duals,main,V151,V146,V167,V172,V176,V184 and all other '
                'prediction/filter/calibration versions and their support rules unchanged.'),
        hypothesis='Observed supply pressure interacts with where holding occurs in a physical '
            'vehicle cycle. Conditional coordination may preserve useful actual-headway '
            'compensation while avoiding avoidable delay on an uncovered service obligation. '
            'This is a testable hypothesis,not a label inferred from V228 cohort sums.',
        decision_information=dict(clock='Decision time is the native lower QUERY,not a later '
            'departure. Keep the exact PREPARE occurrence,cached F/B,identity/phase provenance '
            'and prepare time;record resource inventory,plans,onboard load and executed holding '
            'at query with their own clocks. Known older prepare evidence stays explicitly older; '
            'query inventory is not backfilled as prepare inventory. Retain separate native '
            'and sidecar ordinals with their join;never compare counters from different domains.',
            live_query_fields=list(LIVE_FIELDS),
            allowed='Current bus/trip/direction/position and timestamped native AVL evidence; '
                'observed launches,returns,departures,completed holding;currently published '
                'pending plans,compatible ready stock and all incoming vehicle positions. '
                'Only strictly completed prefixes may supply empirical labels or support.',
            prohibited='Future actual follower/return/assignment/departure,unfinished labels, '
                'hidden sampled road/demand windows,late onboard load,or outcome-selected '
                'anchors cannot enter features,resource allocation or action choice.'),
        slow_message='At each genuine native fresh upper callback,publish an observer-only '
            'versioned pending-obligation snapshot:original public and current posted clocks, '
            'direction,native dispatch rank,anchor/plan provenance,current ready coverage. '
            'Revisions supersede only future obligations;an executed launch is immutable. '
            'Spend each compatible ready vehicle at most once in native pending dispatch '
            'order,FIFO by observed return with stable original inventory order for ties. '
            'A return during the driving pass is available only at the next dispatch tick. '
            'Only such ready stock can fund an upper advance;incoming buses never fund '
            'promised advances. Do not turn pressure into an out-of-cadence upper action.',
        fast_cycle='Key persistence by physical bus,inbound trip and observed launch event. '
            'Destination terminal is opposite the current inbound direction. Track currently '
            'uncovered posted obligations and all compatible incoming vehicles;distance rank '
            'is potential coverage,not future assignment,return order,ETA or slack. Recompute '
            'pressure from current observations at every native query. Keep the vehicle-cycle '
            'ledger through replans until its observed return;continue an outcome link to its '
            'FIRST actual next same-bus launch,even across upper versions. Clearing or revising '
            'an obligation changes current pressure but never erases executed holding. No '
            'per-stop/per-cycle credit minting,negative holding,unused-budget spend or debt reset.',
        shadow_frontier='At each supported nonterminal native query,enumerate ALL and ONLY '
            'native feasible seven-bin actions,including unchanged native. Record a-native, '
            'current_onboard*(a-native),and R(a)-R(native),where '
            'R(a)=(F+a-H)^2+(B-a-H)^2 uses actual prepared F and frozen native B. H cancels '
            'in the difference;do not add executed holding or release/phase offsets again. '
            'Keep the resource-pressure vector and original plan clocks beside these quantities, '
            'not in an arbitrarily weighted score. Onboard exposure is immediate only;R is '
            'a local predicted proxy,not actual CV2. No zero-support or terminal row is dropped. '
            'Missing identities/clocks/resource data yield unsupported,not zero pressure/slack.',
        policy_status='This freezes the resource-cycle interface,shadow frontier and causal '
            'estimands,NOT a deployable selector. Until separately validated action-difference '
            'evidence exists,native V107 requests remain unchanged. A later selector must be '
            'a sealed function of this query-time state,can retain/increase/decrease legal '
            'holding,and must update from real later observations;no fixed one-bin cut, '
            'uniform attenuation,gain escalation or permanent suppression of later holding. '
            'The selector family and any training/calibration data must be preregistered '
            'separately before fitting or opening independent evaluation futures.',
        action_credit='For each legal candidate versus native at the SAME query and upper '
            'deployment,define delta Y as full-day matched-future passenger seconds,actual '
            'arrival CV2,readiness delay and original-public total departure delay. Keep '
            'return LEVEL prediction separate from action DIFFERENCES. Both branches retain '
            'their native later feedback;report direct holding,subsequent holding,nonholding '
            'stop/interstop/boundary changes and actual first same-bus supply,not a fixed '
            'action-tape substitute. Short-window signs cannot replace the full-horizon sign.',
        compensation_test='For a lower action versus native,positive CV2 harm with a paired '
            'CI lower>0 supports regularity benefit of retained holding RELATIVE TO THAT '
            'action pair;it does not establish global necessity. A removable supply cost '
            'requires passenger CI upper<0,CV2/readiness/public-total-delay/completion CI '
            'uppers<=0,and actual executed holding reduction with earlier same-bus return '
            'and earlier FIRST next launch serving an already-recorded obligation. Otherwise '
            'retain unresolved/tradeoff/no-execution/unsupported separately. These are future '
            'evaluation labels,never query features or filters selecting successful paths.',
        acceptance='Any later efficacy test needs V107,V220 upper-only,new-lower-only and '
            'joint with identical permissions and deployments;strong fixed+bilateral stays '
            'a required external control. Joint passenger95%CI upper<0 and actual CV2, '
            'readiness,total original-public delay,incomplete/unlaunched CI uppers<=0 '
            'against V107 AND EACH single-layer comparator. Use paired futures,actor and '
            'demand-day crossed statistics and disjoint selection/evaluation futures with '
            'choices frozen before evaluation. Report both evaluation halves and every '
            'context,including absent/erased actions,assignment changes and failures. '
            'No context replacement,relaxed margins,adaptive extra seeds or frequency claim.',
        next_gate='Before ANY passive path,seal an exact recording schema and fixed started-path '
            'budget separately. Passive legacy actions,uniforms,policy inputs,plan/dispatch '
            'events,all physical clocks,ledgers and endpoints must reproduce exactly. Require '
            'complete query/prepare joins,query inventories/load/executed prefix,upper message '
            'versions,ready eligibility and physical-cycle persistence without extra native '
            'calls/RNG consumption. Then audit shadow support,distinct feasible actions and '
            'resource overlap on consumed development paths only. Engineering support admits '
            'no efficacy,training,selector fit or automatic replay budget. Missing support '
            'stops progression;no gain/threshold search on these paths.',
        limitations='V228 has two consumed actors by two days. Cohort decomposition is not '
            'causal mediation. Local R can worsen while full-day CV2 improves or vice versa; '
            'a strict local nonharm guard around gain.25 generally blocks all reductions. '
            'Ready coverage is conditional inventory accounting,not a reserved future supply '
            'assignment. No full-return uncertainty interval is promoted or assumed valid.',
        runtime_connected=False, selector_defined=False, collection_authorized=False,
        replay_authorized=False, training_authorized=False, main_policy_changed=False)


def assigned_fields(path, function, variable):
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    expressions = [n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets)]
    assert len(expressions) == 1, (path.name, function, variable)
    expr = expressions[0]
    assert isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == 'dict'
    return sorted(k.arg for k in expr.keywords if k.arg is not None)


def source_audit(scripts):
    query = assigned_fields(scripts/'run_freqduet_v216_passive_query.py', '__call__', 'q')
    prepare = assigned_fields(scripts/'freqduet_v224_prepare_capture.py', 'prepare', 'packet')
    sidecar = assigned_fields(scripts/'freqduet_v224_prepare_capture.py', 'query', 'row')
    tree = ast.parse((scripts/'freqduet_v216_observation.py').read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'snapshot')
    result = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Return))
    snapshot = [k.arg for k in result.keywords if k.arg is not None]
    assert any(isinstance(arg, ast.Name) and arg.id == 'stamp' for arg in result.args)
    stamp = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'stamp')
    stamp_return = next(n.value for n in ast.walk(stamp) if isinstance(n, ast.Return))
    stamp_fields = [k.arg for k in stamp_return.keywords if k.arg is not None]
    available = set(query) | set(snapshot) | set(stamp_fields)
    missing_live = sorted(set(LIVE_FIELDS)-available)
    assert not missing_live, ('expected live query interface changed', missing_live)
    missing_sidecar = sorted(set(LIVE_FIELDS)-set(sidecar))
    return dict(scope='AST of frozen collector interfaces only;no archives or runtime imports.',
        live_query_fields_available=sorted(available), prepare_packet_sections=prepare,
        saved_query_sidecar_fields=sidecar, query_sidecar_fields_not_retained=missing_sidecar,
        next_capture='Copy the existing live q fields at their own query clock,join the original '
            'prepare packet,and append strict-past upper-version/physical-cycle ledgers. '
            'Do not pretend query load/inventory existed at prepare. Actual per-path coverage '
            'and exact passive equality remain untested.',
        archive_field_completeness_verified=False, recording_equivalence_verified=False,
        selector_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(code/'tests'), pattern='test_v229_supply_coordination.py')
    assert suite.countTestCases() > 0
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    audit = source_audit(code/'scripts')
    protocol = registration(args.source)
    args.out_root.mkdir(parents=True, exist_ok=True)
    path = args.out_root/'protocol.json'
    assert not path.exists(), 'V229 is already sealed'
    path.write_text(json.dumps(protocol, indent=2)+'\n')
    summary = dict(status='v229_supply_coordination_design_sealed', source=args.source,
        tests=result.testsRun, budget=protocol['budget'], source_interfaces=audit,
        runtime_connected=False, selector_defined=False, collection_authorized=False,
        replay_authorized=False, training_authorized=False, main_policy_changed=False,
        protocol_path=str(path), next_step=protocol['next_gate'])
    (args.out_root/'summary').mkdir(exist_ok=True)
    (args.out_root/'summary/status.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('V229_DESIGN_SEALED_NO_REPLAY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
