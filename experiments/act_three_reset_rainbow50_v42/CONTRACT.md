Three-reset Rainbow 18-to-50 continuation study v42
===================================================

V42 prospectively tests whether V41's phase-conditioned Rainbow controllers
were undertrained. It resumes only each task's immutable episode-18 optimizer,
replay, RNG, and policy snapshot. V41 screen outcomes, held-out outcomes, and
selected methods are forbidden as V42 inputs. V41 artifacts remain immutable
and no V41 rollout is re-executed.

The frozen ACT policy, learned four-phase detector, action grid
`[1, 1.5, 2, 2.5, 3]`, reward, Rainbow hyperparameters, and safety monitor are
unchanged. Each task receives exactly 32 new training episodes, bringing the
inherited training history to 50 episodes. The same three outcome-blind,
geometry-stratified V41 poses are cycled round-robin; their cumulative visit
counts become `[17, 17, 16]`.

After episode 50, the terminal policy is frozen. It is then evaluated without
learning on:

* three matched fixed-pose probes for native 1x and terminal Rainbow;
* ten fresh randomized screen resets; and
* fifty untouched randomized held-out resets per final controller.

Rainbow qualifies only with 3/3 fixed-pose successes, at least 9/10 fresh-screen
successes, and zero physics or safety incidents in the V42 extension and probes.
Otherwise selection fails closed to native 1x. The screen and held-out banks are
fresh and disjoint from V41 and from each other. Held-out outcomes cannot tune,
repair, or replace a controller.

Accounting reports inherited V41 training rollouts, new extension rollouts,
fixed-pose references, randomized screens, held-out executions, cache hits, and
incidents separately. Speedup on the three repeated poses is reported only from
the matched frozen native/Rainbow probes, never from the nonstationary training
trajectory.
