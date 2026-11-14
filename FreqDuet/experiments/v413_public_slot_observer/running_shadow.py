"""V414 passive complete-link forecast using V413's unmodified live runtime."""

from collections import Counter, defaultdict
import inspect
import math

from components import model_components
from coupled_model import _BudgetContinuation
from freqduet_causal_event_graph import MIN_VISITS, Unsupported
from frozen_v406 import ReferenceCommands, ReferenceGraph, _simulate_plan

HOUR_S = 3600


def entry_hour(clock):
    return int(clock // HOUR_S)


class RunningByHour:
    def __init__(self, completed, forecasts, query_s):
        totals, counts = defaultdict(float), Counter()
        for key, visits in completed.items():
            for visit in visits:
                if visit.completed_s < query_s and visit.available_s < query_s:
                    hour = entry_hour(visit.arrival_s - visit.run_s)
                    totals[key, hour] += visit.run_s
                    counts[key, hour] += 1
        self.means = {k: totals[k] / n for k, n in counts.items() if n >= MIN_VISITS}
        self.forecasts = forecasts
        self.calls = Counter()

    def seconds(self, key, entry_s):
        mean = self.means.get((key, entry_hour(entry_s)))
        self.calls['hourly' if mean is not None else 'pooled'] += 1
        return math.ceil(self.forecasts[key]['run_s'] if mean is None else mean)


def detached_kernel():
    # Reuse the frozen event loop, changing only full-link duration lookups.
    source = inspect.getsource(_simulate_plan)
    replacements = (
        ('def _simulate_plan(graph, commands, homogeneous, continuation):',
         'def simulate_shadow(graph, commands, homogeneous, continuation, running):'),
        ("math.ceil(forecast[d, s.routes[d][0].station]['run_s'])",
         'running.seconds((d, s.routes[d][0].station), now)'),
        ("math.ceil(forecast[v['d'], target.station]['run_s'])",
         "running.seconds((v['d'], target.station), now)"),
    )
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    scope = dict(_simulate_plan.__globals__)
    exec(compile(source, '<V414 detached complete-link clock>', 'exec'), scope)
    return scope['simulate_shadow']


def evaluate_shadow(model, simulate):
    s = model.snapshot
    assert s.current_bus == -1
    running = RunningByHour(model.completed, model.forecasts, s.now_s)
    submitted = ReferenceCommands({}, {})
    result = simulate(model, submitted, False, _BudgetContinuation(model, submitted), running)
    return result, dict(running.calls)


class ShadowSlots:
    """Keep native V413 slot records; evaluate the candidate on a separate graph."""

    def __init__(self, observer_module, leads):
        self.module = observer_module
        self.base = observer_module.PublicSlots(leads)
        self.records = self.base.records
        self.shadow = []
        self.simulate = detached_kernel()

    def before_step(self, env, bridge):
        module, simulate, scenarios = self.module, self.simulate, []
        if self.base.schedule is None:
            self.base.schedule = module.slot_schedule(env.timetables, self.base.leads)
        if env.current_time not in self.base.schedule:
            return

        class ShadowGraph(ReferenceGraph):
            def __init__(self, *args):
                super().__init__(*args)
                self.shadow_model = ReferenceGraph(*args)

            def evaluate(self, commands=None, homogeneous=False):
                baseline = super().evaluate(commands, homogeneous)
                record = dict(baseline=model_components(self.snapshot, baseline))
                try:
                    candidate, counts = evaluate_shadow(self.shadow_model, simulate)
                except Unsupported as error:
                    record.update(candidate=[], link_calls={}, returns={}, launches={}, failure=str(error))
                else:
                    record.update(candidate=model_components(self.snapshot, candidate), link_calls=counts,
                        returns=candidate['returns'], launches=candidate['launches'], failure=candidate['failure'])
                scenarios.append(record)
                return baseline

        original = module.ReferenceGraph
        start = len(self.records)
        module.ReferenceGraph = ShadowGraph
        try:
            self.base.before_step(env, bridge)
        finally:
            module.ReferenceGraph = original
        slots = self.records[start:]
        if not slots[0]['scenarios']:
            scenarios = []
        assert len(scenarios) == len(slots[0]['scenarios'])
        self.shadow.append(dict(query_s=env.current_time,
            slots=[(r['trip'], r['lead_s']) for r in slots], scenarios=scenarios))

    def finish(self, start, end):
        self.base.finish(start, end)
