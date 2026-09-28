# Stage-33: independent optimizer-root qualification

**PASS:** all 15 pre-registered, nominal Bonferroni-adjusted root-bootstrap
lower bounds are positive. Eight fresh controllers trained 384 iterations;
qualification used 2480 fit pairs and 960 queries, without exclusions or retuning.

Settled ISE benefit versus current-repeat: +0.03738, CI [0.02273, 0.05070];
shuffled-history: +0.03670, CI [0.01871, 0.05123]; lag-one: +0.02124,
CI [0.00717, 0.03743]. Mean settled-rate MSE falls from lag-one's 0.06889
to 0.06103 (11.42%); reduction CI [0.00315, 0.01134].
Seven of eight pointwise root gates pass. Root 310049's lag-one loss remains:
ISE -0.00093; MSE 0.05704 versus 0.05682. All eight motion gates pass.

Eight server audits and an independent root-count bootstrap match; checkpoint
replay errors are zero. Method: 38796700 steps, 3440 previews, 80 solves,
1000 scalar RHS. Verification: 9600 steps, 3440 previews, 80 solves.
Only 756242 controller-JSON bytes and a 50577-byte summary were pulled;
the 4502422-byte response JSON, raw arrays and checkpoints stay remote.

## Limitations
Evidence covers average local 150-step equal-call PointMaze response utility,
not whole-episode reward, planner-call savings or cross-domain deployment.
Historical selection scores were lost; future exports are fixed without
inventing missing scores or claiming training-curve improvement.

Next: deploy the frozen forecast/response decision in full-episode control,
using fresh paths and counting candidate previews as actual planner work.
