"""Synthetic semantic checks; no native traffic starts."""

from types import SimpleNamespace as NS
from dataclasses import replace
from collections import Counter

from coordination import Manager, admissible, choose, partial_forecasts, release_options
from freqduet_apc_flow_evidence import APCArrival
from freqduet_causal_event_graph import Snapshot, Stop, Vehicle, Visit
from plan_model import PublishedPlan


KEY, NEXT = (1, 2, 3), (1, 2, 4)


def outcome(cost=100, cv=.1, launch=120, ready=20, returned=110):
    return dict(coverage=True, failure=None, missing_commands=set(), cv2=cv, cost=cost,
                launches={8:(launch, 1)}, public_delays={8:max(0, launch-100)},
                readiness_delays={8:ready}, returns={1:returned},
                actions={KEY:(10, 10), NEXT:(20, 5)}, legal={KEY:(0,5,10,15), NEXT:(0,5,10,15)})


class Model:
    def __init__(self, high=False, forbid_fallback=False):
        self.high, self.forbid_fallback = high, forbid_fallback
        self.snapshot = NS(now_s=10, plans=(NS(trip=8, public_s=100),))

    def evaluate(self, commands=None):
        commands = commands or {}
        if commands == {KEY:5, NEXT:10}:
            return outcome(90, cv=.2 if self.high else .09)
        if commands == {KEY:5}:
            return outcome(101 if self.forbid_fallback else 99)
        return outcome()


class ReleaseModel(Model):
    def __init__(self, binding):
        super().__init__()
        plans = (PublishedPlan(8, False, 110, 360, 100),)
        self.snapshot = Snapshot(10, 10, {}, (), plans, 1, 10)
        self.plan_by_trip = {p.trip:p for p in plans}
        self.binding = binding

    def evaluate(self, commands=None):
        changed = self.snapshot.plans[0].release_s == 115
        if (commands or {}).get(KEY) == 5:
            return outcome(cost=90 if changed else 95, launch=115 if changed or not self.binding else 116)
        return outcome()


def checks():
    ref = outcome()
    assert admissible(outcome(99, returned=115), ref, {8:100})
    assert not admissible(outcome(99, launch=121), ref, {8:100})
    assert not admissible(outcome(99, launch=99), ref, {8:100})
    assert not admissible(outcome(99, ready=21), ref, {8:100})
    assert not admissible(outcome(99, cv=.11), ref, {8:100})
    assert not admissible(dict(outcome(99), launches={}), ref, {8:100})
    row = choose([Model()], KEY, 10, (0,5,10,15), None, [])
    assert row[:2] == (5, (NEXT, 10)) and row[4]['cost'] == 99
    row = choose([Model(), Model(high=True)], KEY, 10, (0,5,10,15), None, [])
    assert row[:2] == (5, None)
    row = choose([Model(forbid_fallback=True)], KEY, 10, (0,5,10,15), None, [])
    assert row[:2] == (10, None)
    assert choose([Model()], KEY, 10, (0,5,10), 15, [])[0] == 10
    plans = (PublishedPlan(8, False, 360, 360, 360),
             PublishedPlan(9, False, 720, 360, 720))
    edits = list(release_options(plans, {False:0}, 100))
    assert [p.release_s for p, _ in edits] == [375,390,420]
    assert all(a.release_s == 720 and p.target_s+a.target_s == 720 for p,a in edits)
    assert not list(release_options(plans, {False:0}, 360))
    for binding in (False, True):
        model = ReleaseModel(binding)
        edit = (replace(model.snapshot.plans[0], release_s=115),)
        row = choose([model], KEY, 10, (5,10), None, [edit])
        assert (row[2] is not None) == binding
        assert model.snapshot.plans[0].release_s == 110
    s = Snapshot(1200, 1200, {False:(Stop(3), Stop(4, False, True))},
        (Vehicle(1, 2, False, 0, 'prepare', 50, arrival_s=1190, boarded=10),),
        (), 1, 10)
    visits = [Visit(False, station, t, t+10, t, 100, 50, 10, 10, 360)
              for station in (3,4) for t in (100,460,820)]
    events = [APCArrival(i, 1, 2, False, 3, t, t, t+10, 50, 10, 10, 50)
              for i,t in enumerate((100,460,820))]
    (base, high), censored = partial_forecasts(s, visits, events, {(False,3):.1})
    key = (False,3)
    assert censored == [key] and base.forecasts[key]['rate'] == .1
    assert high.forecasts[key]['rate'] == .125 and high.initial_queues[key] > base.initial_queues[key]
    assert base.horizon_s == high.horizon_s and base.forecasts[False,4]['rate'] == 0
    manager = Manager.__new__(Manager)
    manager.env = NS(timetables=[NS(launch_turn=p.trip, launched=False) for p in plans])
    manager.counts, manager.events = Counter(), []
    manager.apply_plan(edits[0], 100)
    assert manager.events[0]['kind'] == 'publish'
    assert manager.events[0]['budgets'] == [] and manager.events[0]['plans'][0]['trip'] == 8
    from worker import helper
    from run import ASSETS
    physical = helper(ASSETS, 'physical')
    empty = dict(budget_issues=0, budget_tightenings=0, budgeted_commands=0,
                 budgeted_command_s=0, budgeted_residence_s=0, closed_budgets=0,
                 remaining_budget_count=0)
    account = physical.budget_account(dict(publish_ticks=[], queries=[],
        slow_events=manager.events, returns=[]), dict(slow=empty))
    assert account['passed']
    return ['actual launch and CV2 guards; unused return slack allowed',
            'two-stop transfer, first-action fallback and censored sensitivity',
            'published plan anchors, same-hold release binding and current legal masks',
            'historical demand prior on censored APC without claiming an upper bound',
            'real dispatch record accepted by the unchanged physical budget auditor']
