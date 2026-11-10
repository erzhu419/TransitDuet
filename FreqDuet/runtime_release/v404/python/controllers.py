"""Accelerated implementations of three existing, explicitly selected policies."""

from fast_continuation import FastArrivalMetricGraph, FastNativeGraph
from freqduet_arrival_metric_control import ArrivalMetricController, choose_current
from freqduet_consistent_flow import ConsistentFlowController
from freqduet_flow_controller import FlowController
from freqduet_native_continuation import inputs


class FastNativeController(ConsistentFlowController):
    def build_model(self, value, visits):
        base, info = super().build_model(value, visits)
        return FastNativeGraph(base, base.observed), info


class FastArrivalController(ArrivalMetricController):
    def build_model(self, value, visits):
        base, info = FlowController.build_model(self, value, visits)
        model = FastArrivalMetricGraph(base, inputs(self.env, self.prefix, self.evidence))
        recorded = self.env.headway_events.headways_s
        moments = model.arrival_prefix.moments
        assert moments.count == len(recorded)
        assert moments.total == sum(recorded)
        assert moments.square_total == sum(h*h for h in recorded)
        return model, info


class FastCurrentController(FastArrivalController):
    def select_plan(self, model):
        return choose_current(model, self.homogeneous)


CONTROLLERS = dict(native_beam=FastNativeController, arrival_beam=FastArrivalController,
                   arrival_current=FastCurrentController)
