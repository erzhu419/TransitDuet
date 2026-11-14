"""Protect forecast dispatch spacing without editing release clocks."""

from unittest.mock import patch


def backlog_edges(model):
    """One ordered block per direction, with its observed and next-plan edges."""
    blocks = []
    for direction in (False, True):
        plans = [p for p in model.snapshot.plans if p.direction == direction]
        n = sum(p.release_s <= model.snapshot.now_s for p in plans)
        if n < 2:
            continue
        # This selector is for the frozen, fixed timetable (no minimum-gap edits).
        assert all(p.min_gap_s is None for p in plans)
        assert all(p.release_s <= model.snapshot.now_s for p in plans[:n])
        window = plans[:n+1]
        edges = [(a.trip, b.trip, b.public_s-a.public_s) for a,b in zip(window, window[1:])]
        last = model.initial_last_dispatch_s[direction]
        if last >= 0:
            edges.insert(0, (None, window[0].trip, window[0].target_s))
        blocks.append((last, edges))
    return blocks


def excess_ok(candidate, reference, blocks):
    """Do not enlarge any excess above its published headway."""
    new, old = candidate['launches'], reference['launches']
    for anchor, edges in blocks:
        for predecessor, trip, target in edges:
            # Forecasts can end before the next plan. Use only modeled edges.
            if trip not in old or (predecessor is not None and predecessor not in old):
                continue
            a, b = (anchor, anchor) if predecessor is None else (new[predecessor][0], old[predecessor][0])
            if new[trip][0]-a > max(target, old[trip][0]-b):
                return False
    return True


def guarded_controller(capped, native):
    class SpacingCapped(capped):
        def select_plan(self, model):
            blocks = backlog_edges(model)
            if not blocks:
                return super().select_plan(model)
            self.activity['spacing_active_queries'] += 1
            admissible = native.admissible
            rejected = False

            def protected(candidate, reference, public):
                nonlocal rejected
                if not admissible(candidate, reference, public):
                    return False
                if excess_ok(candidate, reference, blocks):
                    return True
                rejected = True
                self.activity['spacing_rejected_scenarios'] += 1
                return False

            # Reuse V406's exact enumeration, objective and tie handling.
            with patch.object(native, 'admissible', protected):
                result = super().select_plan(model)
            self.activity['spacing_queries_with_rejection'] += rejected
            return result

    return SpacingCapped
