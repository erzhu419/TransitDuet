# Stage150: Causal Signed Native Dispatch

Stage149 found little fixed-HIRO-goal opportunity. The manual's next step is
actual timetable authority, not further expansion of that scalar goal sweep.
Inspection identified a concrete channels/haar execution defect: upper was
queried only at nominal launch, making negative departure shifts ineffective.

In the isolated native copy, commit once at nominal minus `delta_max`, observing
the current simulator state. Release at nominal plus the committed signed
offset, subject to the existing fleet cap and service-start boundary. HIRO,
warmup and fixed-expert timing, estimator, action bounds and RE-SAC are retained.
The original FreqDuet source is not changed.

Qualification reuses physical-only Stage148 roots217/229 and their frozen final
weights: reproduce the original HIRO baseline, reproduce the zero-dispatch
neutral control, demonstrate the original negative-offset failure, then execute
causal -60/+60 commands. Ten full-clock episodes, 613,800 ticks, no optimization,
no checkpoint download. All jobs are dynamically eligible for node001-006.
The 31 focused dispatch/frontier/authority/diagnostic tests pass, including
signed action bounds, one-shot causal queries and fleet-blocked releases.

Next: after execution qualifies, train dispatch policies under a matched budget;
only then compare learned timetable control and frequency-routing ablations.

## Limitations

Fixed dispatch probes using HIRO-trained weights qualify execution, not learned
channels performance. This repair is not yet a learned rolling timetable curve,
promotion result, or independent frequency-separation performance claim.
