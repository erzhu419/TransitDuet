"""V223 algebraic prototype only; not connected to the controller runtime."""

import numpy as np

ACTIONS = (0., 5., 10., 15., 20., 30., 45.)
GAIN = .25


def launch_phase(row, identity, prepare_event_seq):
    """Resolve a delivered plan offset from the query's existing launch ledger."""
    if row is None:
        return None
    if tuple(row[k] for k in ('trip_id', 'bus_id', 'direction')) != identity:
        return None
    if row['launch_event_seq'] >= prepare_event_seq:
        return None
    native, coordinated = row['native_tick_s'], row['coordinated_tick_s']
    if coordinated == native:
        return 0.
    if (coordinated > native or row['reserved_bus_id'] != row['bus_id']
            or row['actual_launch_s'] != coordinated):
        return None
    return coordinated-native


def command(context, feasible_bins, phases=None):
    """Use actual query gaps once, with one relative phase correction if supported.

    phases is (predecessor, current, predicted physical follower), resolved at
    the SAME prepare event as the frozen native gaps. None means unsupported,
    not a zero phase. Forward-only fallback remains exactly the native rule.
    """
    correction = 0.
    if not context.evidence_valid:
        native, mode = 0., 'native_missing_forward'
    elif not context.follower_evidence_valid:
        native = GAIN*(context.target_headway_s-context.forward_headway_s)
        mode = 'native_forward_fallback'
    else:
        native = GAIN*(context.follower_departure_gap_s-context.forward_headway_s)/2
        mode = 'native_missing_phase'
        if phases is not None and all(p is not None for p in phases):
            predecessor, current, follower = phases
            correction = GAIN*(2*current-predecessor-follower)/2
            mode = 'common_reference'
    raw = native+correction
    clipped = np.clip(raw, 0., ACTIONS[-1])
    choices = np.array(sorted(feasible_bins), dtype=int)
    # Match pilot.step.ACTIONS, not the unrelated float32 policy probabilities.
    bins = np.asarray(ACTIONS, dtype=float)
    selected = int(choices[np.argmin(np.abs(bins[choices]-clipped))])
    return dict(mode=mode, native_raw_s=native, phase_correction_s=correction,
                raw_s=raw, clipped_s=float(clipped), bin=selected, action_s=ACTIONS[selected])


def registration(source):
    return dict(stage='v223_common_reference_algebra_only', source=source,
        gain=GAIN, action_bins_s=ACTIONS,
        reference='Offsets are delivered executable plan ticks relative to the native plan at '
            'the SAME upper admission,not absolute public-timetable lateness or desired +30. '
            'Latch at same-reserved-bus actual launch;no future actual-neighbour labels. '
            'Record explicit zero for unmodified launched trips;missing is not zero. '
            'Keep actual prepared forward gap and frozen native follower estimate,including '
            'already-realized nonholding time. Do not reconstruct either from release-only gaps.',
        law='One request: gain/2 * ((backward-forward)+(2*u_self-u_front-u_follower)), '
            'then native clipping and feasible-bin quantization once. Equivalently use '
            'forward reference H+u_self-u_front and backward reference H+u_follower-u_self. '
            'H cancels in the bilateral branch. Preserve native forward-only/missing-evidence '
            'branches;do not add offsets again to already projected target_headway metadata.',
        support='Resolve predecessor identity from the matched completed departure and follower '
            'identity from the native prepare-time AVL estimate. All phase launch rows must '
            'precede prepare in global event order and match trip,bus,direction. No lookup by '
            'fixed trip-number neighbour or actual future departure neighbour. Unavailable '
            'evidence keeps native bilateral025;no invented zero offset or credit.',
        tests='Zero-phase equivalence to frozen V106 rule_action;pure common/relative phase '
            'invariance;real nonholding/holding response;clipping,feasibility,ties;identity and '
            'clock abstention;no forced bin decrease for a sub-bin phase offset.',
        progression='Tests admit ONLY a separately authorized passive runtime handoff preflight. '
            'V221 lacks archived predicted follower IDs and a prepare-time phase ledger,so '
            'this work cannot retrospectively certify its support or action authority. Before '
            'any intervention budget,freeze a new prospective handoff/recording protocol;then '
            'check no-change passive equivalence and actual supported command differences. '
            'Zero executable lower differences stops efficacy progression;do not enlarge gain '
            'or force a bin. Any efficacy validation needs separately reserved independent '
            'days/futures and V107,fixed+bilateral,upper-only/lower-only controls.',
        frozen='Original12 vehicles,seven bins,V220 upper authority and cadence,all predictors '
            'and calibrations including V151/V146/V167,actors/critics/duals and main unchanged.',
        execution='scheduler high,manual dispatch,node001--006;tests only,small log returns',
        budget=dict(archive_scans=0, new_replays=0, new_training=0, new_futures=0),
        runtime_connected=False, main_policy_changed=False)
