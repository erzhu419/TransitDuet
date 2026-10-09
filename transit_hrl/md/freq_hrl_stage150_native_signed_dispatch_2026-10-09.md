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
Run `native_transit_dispatch_stage150_qualification_20261009_r1` completed
as `t138738/t138739` on node004/node006. Both HIRO baselines and zero-dispatch
neutral controls reproduced exactly. Each episode queried upper once for each
of 262 trips. Legacy -60 changed no departure or physical outcome; repaired -60
advanced 261 trips by 60 seconds (first trip clipped at service start), while
+60 delayed all 262 trips. No network updates or release lateness occurred.

Against zero, -60 changes cost by -0.015085/+0.011205 and restricted wait by
-0.128/+0.032 minutes for roots217/229. The +60 cost deltas are +0.423966/+0.007058;
root217 needs one extra peak vehicle. Execution is correct, but a constant
advance is not a consistent service improvement.

Next: Stage151 trains HIRO versus signed dispatch, crossed with legacy versus
service-interval credit, on new matched roots. Test learned upper against its
zero-action deployment before expanding frequency-routing ablations.

## Limitations

Fixed dispatch probes using HIRO-trained weights qualify execution, not learned
channels performance. This repair is not yet a learned rolling timetable curve,
promotion result, or independent frequency-separation performance claim.
