"""Event-fed implementation of the frozen V120 completed-prefix predictor."""

import audit_freqduet_v120_component_eta as model


class OnlineMeans(model.CompletedMeans):
    def __init__(self):
        super().__init__([])

    def add(self, key, completion, value):
        if value < 0:
            raise AssertionError('negative observed component duration')
        times, sums = self.groups.setdefault(key, ([], [0.]))
        if times and completion < times[-1]:
            raise AssertionError('observed component events are not chronological')
        times.append(completion)
        sums.append(sums[-1]+value)


class OnlineComponentETA:
    def __init__(self, geometry):
        self.geometry = geometry
        self.history = OnlineMeans()
        self.launches, self.motion, self.departures, self.pending = {}, {}, {}, {}

    def launch(self, trip, direction, time):
        terminal = self.geometry[direction]['ids'][0]
        self.launches[trip] = time
        self.departures[(trip, terminal, direction)] = time
        self.motion[trip] = model.MovingPrefix([], [dict(station_id=terminal, time_s=time)])

    def arrival(self, key, time, holding):
        trip, station, direction = key
        route = self.geometry[direction]
        index = route['index'][station]
        if index == 0:
            raise AssertionError('terminal arrival entered ordinary stop telemetry')
        previous = (trip, route['ids'][index-1], direction)
        start = self.departures[previous]
        self.history.add(('running', direction, station), time, time-start)
        motion = self.motion[trip]
        motion.arrivals.append(dict(station_id=station, time_s=time))
        motion.at.append(time)
        self.pending[key] = dict(arrival=time, holding=holding)

    def ready(self, key, time):
        stop = self.pending[key]
        stop['ready'] = time
        self.history.add(('service', key[2], key[1]), time, time-stop['arrival'])

    def departure(self, key, time, holding):
        trip, station, direction = key
        if key in self.pending:
            stop = self.pending.pop(key)
            executed = holding-stop['holding']
            ticks = time-stop['ready']-executed
            self.history.add(('holding', direction, station), time, executed)
            self.history.add(('ticks', direction, station), time, ticks)
        elif station != self.geometry[direction]['ids'][0]:
            raise AssertionError('ordinary departure lacks observed arrival/ready')
        self.departures[key] = time
        motion = self.motion[trip]
        motion.departures.append(dict(station_id=station, time_s=time))
        motion.dt.append(time)

    def predict(self, evidence, direction, station):
        raw = evidence['follower_gap_prediction_s']
        result = dict(prediction_time_s=evidence['ready_time_s'], component_used=False, gap_s=raw)
        if raw is None:
            return dict(result, status='unavailable')
        if evidence['follower_source'] != model.clock.JOURNEY:
            return dict(result, status='other_eta_source')
        trip, now = evidence['predicted_follower_trip_id'], evidence['ready_time_s']
        progress = evidence['speed_mps']*(now-self.launches[trip])
        route = self.geometry[direction]
        target = route['index'][station]
        if abs(progress+evidence['spatial_gap_m']-route['positions'][target]) > 1e-6:
            raise AssertionError('online launch clock does not match target-stop geometry')
        origin = self.motion[trip].origin(now, progress, route)
        if origin is None:
            return dict(result, status='not_causally_moving')
        prediction = model.component_prediction(self.history, route, direction, origin, target, progress, now)
        if prediction is None:
            return dict(result, status='missing_completed_component')
        stops = prediction['service_s']+prediction['holding_s']+prediction['ticks_s']
        return dict(result, **prediction, status='predicted', component_used=True,
            gap_s=prediction['running_s']+stops+prediction['target_service_s'])
