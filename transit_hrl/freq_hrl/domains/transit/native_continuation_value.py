"""Resample only post-probe innovations after the original causal snapshot."""

from freq_hrl.domains.transit.native_timetable_context import NativeTimetableContextAudit


class NativeContinuationValueAudit(NativeTimetableContextAudit):
    def __init__(self, env, model, probe_index, *, continuation_seed, forced_candidate=None, reference=None):
        super().__init__(env, model, probe_index, forced_candidate=forced_candidate, reference=reference)
        self.continuation_seed = continuation_seed

    def select_candidate(self, candidates):
        selected = super().select_candidate(candidates)
        if len(self.decisions) == self.probe_index:
            tape = self.env.scenario_tape
            self.env.scenario_tape = type(tape)(self.continuation_seed, version=tape.version)
            self.probe.update(continuation_seed=self.continuation_seed,
                continuation_start_time_s=float(self.env.current_time))
        return selected
