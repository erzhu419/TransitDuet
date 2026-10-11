# Stage186: Learned Full Timetable Value Encoder

Encode every Stage185 event with shared 6-8 projection, kernel3 temporal
convolution and masked mean/max pooling. Add16 learned features to the
original98 inputs: 114-16-3 cost/wait heads, 2,147 parameters.

Keep Stage184's initial predictor by zeroing the new sequence projection.
Normalize real training events only; no truncation or padding statistics.
Preserve all120 states/720 labels, twenty scene groups, target scaling,
proper MSE, optimizer and all28 frozen100-step schedules. Reuse Stage184
as control; primary is timetable_sequence:wait_feasible, same cost/wait gate.

Scheduler only, node001-006 unbound, one CPU/3GB: 28 fits/2,800 updates.
No control fitting, native episodes or actor updates. Same update count,
extra encoder computation. Thirty-one focused tests pass, including masks,
serialization, initial predictor, loss gradients and legacy fitting behavior.

## Result and Next Step

t143351 completed exit0 on node005: 28 fits/2,800 updates qualified.
Runtime was about172 seconds; the 1,673,805-byte result auto-synced.
All encoder gradient maxima are nonzero (0.0474-0.0599).

| Split | root397 cost | root401 cost | Overall wait, min |
|---|---:|---:|---:|
| Scene | +0.0134538 | -0.0126304 | +0.0085750 |
| Regime | +0.0007745 | -0.0062992 | +0.0111250 |
| Lower root | +0.0019536 | -0.0054860 | +0.0212250 |

Scene/regime wait improves versus Stage184, lower-root wait worsens;
scene cost worsens, regime/lower-root cost improves. Both selectors fail
the joint gate. Scene false-feasible choices rise from28/51 to39/69.
Next: independently registered development-scene expansion.

## Limitations

This is grouped value training under original greedy continuation, not
repeated learned upper actions or end-to-end HRL. Scene coverage remains
twenty development groups; training cost/wait are negative but held-out wait
is positive. Complete input and active gradients have not established stable
transfer. Third/fourth action-label scenes remain untouched.
