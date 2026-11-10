"""One V276 change: native bilateral continuation, with resource prediction receipts."""

from freqduet_flow_controller import FlowController
from freqduet_native_continuation import NativeGraph, PrepareEvidence, inputs


class ConsistentFlowController(FlowController):
    def __init__(self, env, launches, credit):
        super().__init__(env, launches, credit)
        self.evidence = PrepareEvidence()
        self.predictions = []

    def attach(self, Bus):
        super().attach(Bus)
        self.evidence.attach(Bus)

    def detach(self):
        self.evidence.detach()
        super().detach()

    def observe(self, bus, legal, native_action):
        self.evidence.check(bus, legal, native_action, float(self.env.current_time))
        return super().observe(bus, legal, native_action)

    def build_model(self, value, visits):
        base, info = super().build_model(value, visits)
        return NativeGraph(base, inputs(self.env, self.prefix, self.evidence)), info

    def record_prediction(self, result, value):
        if result['action_s'] == value.native_action_s or 'reference' not in result:
            return
        def resources(model, vehicle):
            return dict(return_s=model['returns'].get(vehicle.bus),
                next_launch_s=model['next_launch'].get(vehicle.bus),
                remaining_trip_h_s=sum(h for (b,trip,_),(_,h) in model['actions'].items()
                    if (b,trip)==(vehicle.bus,vehicle.trip)))
        own = next(v for v in value.vehicles if v.bus==value.current_bus)
        self.predictions.append(dict(query_s=value.now_s,bus=own.bus,trip=own.trip,
            station=value.routes[own.direction][own.index].station, native_s=value.native_action_s,
            request_s=result['action_s'], vehicles=[dict(bus=v.bus,trip=v.trip,direction=v.direction,
                committed_station=value.routes[v.direction][v.index].station if v.phase=='hold' else None,
                reference=resources(result['reference'],v), full=resources(result['selected'],v),
                fallback=resources(result['fallback'],v)) for v in value.vehicles if v.phase!='terminal']))

    def summary(self):
        return dict(super().summary(),same_state_rule_checks=self.evidence.checked,
            resource_prediction_queries=len(self.predictions),continuation='native_prepare_bilateral025')
