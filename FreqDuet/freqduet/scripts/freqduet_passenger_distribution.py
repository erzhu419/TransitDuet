"""Past-APC Poisson arrivals and binomial alighting with fixed historical means."""

import numpy as np

from freqduet_closed_loop_forecast import fluid_rollout
from freqduet_holding_distribution import DRAWS, METRICS, probabilities, sample_action
from freqduet_travel_distribution import TravelPath

MODES = ("initial_queue", "arrival_queue", "passenger_joint")


def fit_passengers(apc, observed):
    now, stops = observed["time_s"], {}
    prefix = apc.loc[apc.time_s < now]
    grouped = dict(tuple(prefix.groupby(["direction", "station_id"])))
    for direction, route in observed["routes"].items():
        rows = []
        for station, flow in zip(route["station_ids"], observed["flows"][direction]):
            services = grouped.get((direction == "True", station))
            stock, age = 0, 0.0
            if services is not None:
                last = services.loc[services.time_s.idxmax()]
                stock, age = int(last.left_behind), now - float(last.time_s)
                if last.time_s != flow["last_observed_s"]:
                    raise ValueError("APC prefix and frozen flow snapshot disagree")
            expected = stock + flow["rate_per_s"] * age
            if abs(expected - flow["queue"]) > 1e-8:
                raise ValueError("initial-queue mean changed from the historical APC prior")
            rows.append(dict(stock=stock, age_s=age, rate_per_s=flow["rate_per_s"],
                             alight_fraction=flow["alight_fraction"]))
        stops[direction] = rows
    return dict(now_s=now, dt_s=observed["dt_s"], horizon_s=1800, stops=stops,
                samples=len(prefix), latest_time_s=float(prefix.time_s.max()) if len(prefix) else None)


class PassengerProcess:
    def __init__(self, model, mode, draw):
        self.model, self.mode = model, mode
        self.continuous_arrivals = mode == "initial_queue"
        self.initial_rng = np.random.default_rng(20261017 + draw)
        self.arrival_rng = np.random.default_rng(20261018 + draw)
        self.alight_rng = np.random.default_rng(20261019 + draw)

    def initialize(self, flows, vehicles):
        for direction, stops in self.model["stops"].items():
            for params, flow in zip(stops, flows[direction]):
                flow["queue"] = float(params["stock"] + self.initial_rng.poisson(params["rate_per_s"] * params["age_s"]))
        for vehicle in vehicles.values():
            if vehicle["load"] != int(vehicle["load"]):
                raise ValueError("observed onboard count is not an integer")

    def arrival_events(self, end):
        if self.continuous_arrivals:
            return
        now, dt, horizon = self.model["now_s"], self.model["dt_s"], self.model["horizon_s"]
        # Draw the same full 30-minute process for both horizons, so their
        # 15-minute prefixes agree. Tick aggregation avoids one event per person.
        for direction, stops in self.model["stops"].items():
            for station, params in enumerate(stops):
                count = self.arrival_rng.poisson(params["rate_per_s"] * horizon)
                times = np.ceil((now + self.arrival_rng.uniform(0, horizon, count)) / dt) * dt
                times, counts = np.unique(times, return_counts=True)
                for time, count in zip(times, counts):
                    if time <= end:
                        yield float(time), direction, station, int(count)

    def exchange(self, vehicle, stop, terminal):
        load = vehicle["load"]
        if terminal:
            return load, 0.0
        if self.mode == "passenger_joint":
            if load != int(load) or stop["queue"] != int(stop["queue"]):
                raise ValueError("joint passenger process became fractional")
            alight = float(self.alight_rng.binomial(int(load), stop["alight_fraction"]))
        else:
            alight = load * stop["alight_fraction"]
        board = min(stop["queue"], vehicle["capacity"] - load + alight)
        return alight, board

    def check_final(self, flows, vehicles):
        values = [stop["queue"] for stops in flows.values() for stop in stops]
        values += [vehicle["load"] for vehicle in vehicles.values()]
        if min(values) < -1e-8:
            raise ValueError("negative forecast passenger count")
        if self.mode == "passenger_joint" and any(value != int(value) for value in values):
            raise ValueError("joint passenger final state is fractional")


def passenger_ensemble(observed, horizon, holding, travel_model, passenger_model, mode,
                       draws=DRAWS, event_sink=None):
    metrics, mass_errors = {metric: [] for metric in METRICS}, []
    for draw in range(draws):
        holding_rng = np.random.default_rng(20261015 + draw)
        travel = TravelPath(travel_model, "shared_block", np.random.default_rng(20261016 + draw))

        def predict(vehicle, time, fraction):
            route = observed["routes"][str(vehicle["direction"])]
            return travel(dict(vehicle, travel_station_id=route["station_ids"][vehicle["station_index"]]), time, fraction)

        result = fluid_rollout(observed, horizon, "response",
            action_predictor=lambda values: sample_action(probabilities(holding, values), holding_rng.random()),
            event_sink=None if event_sink is None else lambda row: event_sink(draw, row), travel_predictor=predict,
            passenger_process=PassengerProcess(passenger_model, mode, draw))
        mass_errors.append(abs(result["passenger_mass_error"]))
        for metric in metrics:
            metrics[metric].append(result[metric])
    return dict(mean={metric: float(np.mean(values)) for metric, values in metrics.items()},
                draws=metrics, max_mass_error=max(mass_errors))
