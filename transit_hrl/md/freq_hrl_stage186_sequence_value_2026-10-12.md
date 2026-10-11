# Stage186: Learned Full Timetable Value Encoder

Use every Stage185 preaction event: shared 6-8 tanh projection, 8-8
kernel3 temporal convolution, masked mean/max pooling. Concatenate its
16 learned features with the original98 inputs; 114-16-3 cost/wait heads,
2,147 parameters. No event truncation or padding in normalization/pooling.

Keep Stage184's initial predictor by zeroing the new sequence projection.
Train event normalization on real training events only. Preserve all120
states/720 labels, twenty scene groups, target scaling, proper MSE,
optimizer and all28 frozen100-step schedules. Reuse Stage184 as control.
Primary: timetable_sequence:wait_feasible; same both-root/three-split gate.
Report false-feasible choices alongside actual cost/wait. Verify encoder
loss gradients, not merely parameter movement from weight decay.

Scheduler only, node001-006 unbound, one CPU/3GB: 28 fits/2,800 updates.
No new control fitting, native episodes or actor updates. Same update
count, but the sequence branch adds computation.

Thirty-one focused tests pass, including ragged padding invariance,
NumPy/Torch inference agreement, matching initial predictor, train-only
event scales, actual encoder gradients, frozen groups/steps, label isolation,
primary gate and existing floor/tail/scene-stopping behavior.

## Limitations

This is grouped value training under original greedy continuation, not
repeated learned upper actions or end-to-end HRL. Scene coverage remains
twenty development groups. Third/fourth action-label scenes stay untouched
unless the fixed primary gate passes and a new transfer study is registered.
