#!/usr/bin/env python3
"""Seal the V160 full-ETA interval protocol without collecting or fitting data."""

import argparse
from copy import deepcopy
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys


MODES = ('old', 'handoff')
SCOPES = ('all', 'active', 'clearance', 'hour2', 'hour13')
MIN_DAYS = 19


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_parent(parent):
    plan = parent['registration']
    require(parent['status'] == 'v160_metadata_and_preregistration_complete' and
        parent['execution_source'] == '14d930ee25', 'requires the completed V160 metadata source')
    require(plan['status'] == 'preregistered_not_collected' and
        not plan['collector_submitted'] and not parent['new_days_collected'] and
        not parent['interval_fit'], 'V160 cohorts must still be reserved, not consumed')
    cal, confirm = plan['calibration_days'], plan['confirmation_days']
    require(len(cal) == len(confirm) == 32 and len(set(cal+confirm)) == 64,
        'requires two disjoint groups of 32 whole demand days')
    require(len(plan['actors']) == len(set(plan['actors'])) == 12 and
        tuple(plan['modes']) == MODES and plan['checkpoint_ep'] == 199,
        'V160 actor/mode/checkpoint contract changed')
    require(plan['sampler_seeds'] == {str(d): 1600000000+i for i, d in enumerate(cal+confirm)},
        'V160 matched sampler allocation changed')
    require((plan['service_point_source'], plan['running_shadow_source'],
        plan['v146_extensions_s'], plan['window_s'], plan['tails']) ==
        ('1b783ea8f3', '5be4cea534', [3., 0.], 3600., [.1, .1]), 'frozen forecast contract changed')
    require(not plan['freshness']['overlap_days'] and not plan['freshness']['overlap_samplers'],
        'V160 reported a cohort conflict')
    return deepcopy(plan)


def rank_levels(day_count):
    """A deterministic rule only: this does not read residuals or fit bounds."""
    if day_count < MIN_DAYS:
        return None
    lower = (day_count+1)//10
    upper = day_count+1-lower
    return dict(day_count=day_count, lower_rank=lower, upper_rank=upper,
        lower_cdf=lower/day_count, upper_cdf=upper/day_count)


def registration(parent):
    reservation = validate_parent(parent)
    return dict(status='preregistered_not_collected', stage='v161_joint_residual_interval_protocol',
        reservation=reservation,
        frozen=dict(point='V151', point_source='1b783ea8f3',
            running_shadow_source='5be4cea534', running_shadow_role='diagnostic only; no model selection',
            controller='B200 ep199; actor, critic, dual and main policy unchanged',
            v146_extensions_s=[3., 0.], native_window_s=3600., tails=[.1, .1]),
        target=dict(quantity='same physical predicted follower remaining time to target-stop ready',
            error='prediction minus truth', channels=['running', 'service', 'residence'],
            residence='holding + execution_wait + inseparable current-ready residence',
            joint_unit='one query vector, never independently resample or add component quantiles',
            episode=['predicted_follower_trip', 'predicted_follower_bus', 'target_stop', 'direction'],
            opaque='abstain; retain full residual for separate descriptive reporting, never invent components'),
        conditioning=dict(candidate_key=['actor', 'mode', 'query_phase', 'predicted_target_phase',
                'direction', 'target_stop', 'position_status', 'native_stage_kind',
                'native_support_class', 'predicted_horizon_band'],
            query_phase='clearance iff query_time_s >= published demand_end_time_s; otherwise active',
            predicted_target_phase='clearance iff query_time_s + fixed V151 point >= demand_end_time_s',
            position_status='exact frozen V151 observable.position_status, recorded before label lookup',
            native_stage_kind='current follower service / ready-residence / known-command / moving-launch',
            native_support_class='not_applicable / finite_window / finite_historical_fallback',
            horizon_bands_s=['[0,900]', '(900,1800]', '(1800,infinity)'],
            horizon='fixed V151 point, not realized time or the running shadow',
            fallback='no pooling across a missing or low-support candidate key',
            excluded=['late policy load', 'label status', 'future holding', 'realized horizon',
                'future follower identity switches', 'future arrival or departure clocks']),
        support=dict(denominator='all query-valid follower requests with finite nonnegative V151 point; '
                'opaque and native-low-support requests remain in the abstention denominator',
            native='retain V142/V146 current service/ready rank abstention: fewer than nine completed '
                'elapsed-duration survivors cannot emit an interval; known commands and moving states '
                'have no current-stage survivor requirement',
            clocks='prefix labels require available_time_s < query_time_s; same-clock ready state '
                'requires the already validated earlier callback of that physical follower',
            fallback='retain frozen 3600s selection and all-completed-history fallback only when window empty',
            minimum_calibration_days=MIN_DAYS,
            minimum_days_reason='at least two day-mass order ranks in each fixed 0.1 tail',
            day_count='distinct calibration demand days contributing eligible same-key joint vectors',
            low_support='permanent abstention for that sealed key, including unseen confirmation keys',
            missing='missing query-time context or required support abstains; no late-load backfill',
            labels='unresolved eligible labels block calibration seal; unresolved finite confirmation '
                'labels make validation inconclusive, never zero-fill or drop them'),
        estimator=dict(role='offline empirical conditional joint residual distribution; no Gaussian assumption',
            mixture='within key: equal contributing day mass, equal target-episode mass within day, '
                'equal query mass within episode; each atom retains its complete R/S/D vector',
            total_error='sum R/S/D within the same atom',
            day_mass_ranks='D distinct days; k=floor((D+1)/10), j=D+1-k; '
                'Qlow=inverse weighted CDF(k/D), Qhigh=inverse weighted CDF(j/D)',
            inverse_cdf='smallest observed total error with cumulative weight >= requested level; no interpolation',
            interval='[max(0,m-Qhigh), max(0,m-Qlow)] in remaining seconds, m=fixed V151 point',
            finite_sample_scope='day-mass rank correction of an empirical mixture, not an exact IID or '
                'conditional conformal coverage theorem; independent confirmation decides adequacy',
            update='calibration only; no confirmation-day residual updates or point recentering',
            calibration_days_are_not_window='native prefix window remains 3600s; offline calibration '
                'mixture uses the reserved calibration days only'),
        comparator=dict(name='coarse full-ETA joint residual reference',
            key=['actor', 'mode', 'query_phase'],
            rule='same calibration days, joint atoms, weighting, day-mass ranks, native support and '
                'physical truncation as candidate, removing all other conditioning dimensions',
            matching='score and width contrasts on exactly the candidate-finite query mask; '
                'missing reference bounds make validation inconclusive',
            not_comparator='V146 stage intervals are not complete ETA intervals'),
        inference=dict(scopes=list(SCOPES), modes=list(MODES),
            scope_clocks='hour2/hour13 use floor(query_time_s/3600)==2/13; phase uses published cutoff',
            reduction='query means within same-follower/target episode, then equal episode means within '
                'actor/day; equal actor/day cells. Finite fraction uses all eligible episodes; '
                'coverage/tails/score/width use finite matched episodes',
            grid='12 actors x 32 confirmation demand days in each mode/scope; empty or missing '
                'eligible/finite cells are inconclusive, not omitted',
            bootstrap='existing hierarchical_bootstrap: independently resample actor and whole demand-day '
                'axes, sharing sampled days across actors and all paired metrics/arms/modes/scopes',
            draws=5000, seed=20260803, ci95_percentiles=[2.5, 97.5],
            one_sided_percentiles=[5., 95.],
            score='(U-L)+10*max(L-T,0)+10*max(T-U,0); nominal central coverage 0.8',
            outputs=['crossed CI', 'per-actor marginal', 'per-demand-day marginal',
                'counts of all requests/eligible/abstained/unresolved', 'native support reasons',
                'candidate-key support and coverage', 'paired score and width'],
            marginal_scope='actor/day marginals and candidate-key diagnostics do not select new gates'),
        gates=dict(coverage_lower95_min=.8, below_upper95_max=.1, above_upper95_max=.1,
            finite_fraction_lower95_min=.8, score_delta_upper95_max=0.,
            strict_score_scopes=['all', 'clearance'],
            applies_to='both modes and every one of all/active/clearance/hour2/hour13',
            claim='one conjunction: every coverage/tail/support/no-harm test passes and paired score '
                'upper95 is strictly negative in all and clearance for both modes',
            multiplicity='intersection-union admission, not a claim of simultaneous 95% per-key '
                'intervals; no choosing a subset of successful modes/scopes',
            failure='not admitted if any gate fails; missing evidence is inconclusive; no controller promotion'),
        isolation=dict(first='explicit future authorization, then calibration cohort only; none collected now',
            calibration_seal='all 768 calibration paths/receipts complete; freeze deterministic cell map, '
                'support counts, candidate and reference bounds and source before confirmation collection',
            release='calibration support must permit a finite estimate in every actor/day primary scope '
                'and actor/day-average finite fraction >=0.8 in each mode/scope, else stop for insufficient '
                'support without opening confirmation or changing grouping',
            confirmation='only after calibration seal, collect the separate 768-path cohort; '
                'apply sealed maps once, keeping all 32 days and all actors in their assigned group',
            no_reuse='no confirmation fitting, online interval adaptation, endpoint scaling, threshold '
                'selection, window changes or regrouping; subsequent revision needs new demand days',
            provenance='collector must save contexts and native support at query before any label lookup, '
                'including unresolved/opaque queries; target labels attach separately when available',
            controller='interval success alone does not admit a holding policy'),
        execution=dict(collector_submitted=False, new_days_collected=0, new_rollouts=0, new_training=0,
            interval_fit=False, confirmation_outcomes_read=False, main_policy_changed=False,
            permitted='server scheduler protocol/unit checks only; local summary only'))


def gate(readouts):
    """Consume future crossed readouts; absent evidence is not a passing zero."""
    expected = {(m, s) for m in MODES for s in SCOPES}
    identities = [(r['mode'], r['scope']) for r in readouts]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        return dict(decision='inconclusive', reason='incomplete_or_duplicate_mode_scope_grid')
    fields = ('coverage_lower95', 'below_upper95', 'above_upper95',
        'finite_fraction_lower95', 'score_delta_upper95')
    failures = []
    for row in readouts:
        if not row['complete_actor_day_grid'] or row['finite_unresolved_n'] or row['missing_reference_n']:
            return dict(decision='inconclusive', reason='incomplete_labels_support_or_reference')
        if any(row.get(k) is None or not isfinite(row[k]) for k in fields):
            return dict(decision='inconclusive', reason='missing_crossed_bound')
        tests = dict(coverage=row['coverage_lower95'] >= .8,
            below=row['below_upper95'] <= .1, above=row['above_upper95'] <= .1,
            support=row['finite_fraction_lower95'] >= .8,
            score=row['score_delta_upper95'] < 0. if row['scope'] in ('all', 'clearance')
                else row['score_delta_upper95'] <= 0.)
        failures.extend(f"{row['mode']}/{row['scope']}/{k}" for k, passed in tests.items() if not passed)
    return dict(decision='pass' if not failures else 'not_admitted', failures=failures,
        controller_admitted=False)


def run(args):
    parent = json.loads((args.v160_root/'summary/status.json').read_text())
    protocol = registration(parent)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).resolve().parents[1]/'tests'), '-p', 'test_v161_*.py', '-v'], check=True)
    protocol['execution_source'] = os.environ['FREQDUET_DIAGNOSTIC_COMMIT']
    target = args.out_root/'protocol.json'
    if target.exists():
        require(json.loads(target.read_text()) == protocol, 'sealed protocol differs; do not overwrite')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(protocol, indent=2, allow_nan=False)+'\n')
    summary = dict(status='v161_protocol_sealed_not_collected', protocol=protocol,
        parent_summary=str(args.v160_root/'summary/status.json'),
        residual_files_opened=0, interval_fit=False, new_rollouts=0, new_training=0,
        new_days_collected=0, main_policy_changed=False,
        evidence='protocol/unit checks only; no empirical interval or coverage result')
    output = args.out_root/'summary/status.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('V161_PROTOCOL_SEALED_NO_COLLECTION_NO_FIT', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v160-root', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    run(parser.parse_args())
