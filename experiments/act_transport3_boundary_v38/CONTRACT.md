# Transport-first terminal-boundary study v38

This prospective study tests one cheap semantic hypothesis: use `3x` during
free-space transport, then downshift only after a current-observation terminal
approach event. Historical outcomes are used only to pin the accelerated v37
champion. No v20-v37 rollout is re-executed or counted in the online budget.

The frozen ACT policy, simulator, learned four-phase detector, success
criterion, and speed grid `[1, 1.5, 2, 2.5, 3]` are unchanged. The approach
gate reads only current FK end-effector positions and the current learned phase.
It cannot read rewards, contacts, future actions, terminal success, object
state, or future detector outputs. Once triggered during transport it stays
latched until transport ends.

Each task receives exactly 25 fresh online search episodes:

1. The preregistered transport-first challenger runs on five discovery seeds.
2. The v37 champion and challenger each run on the same ten fresh paired seeds.

The challenger requires at least `4/5` discovery successes, `9/10` paired
successes, and zero physics or safety incidents. It replaces the champion when
it improves paired successes without losing more than 5% failure-aware
throughput, or ties paired successes while improving throughput by at least 5%.
It also replaces a champion below `9/10` when it qualifies. Ambiguous evidence
retains the accelerated champion. Native `1x` is used only if both accelerated
controllers show a clear paired failure (`<=7/10`) or a registered incident.

Only after all searches seal does a fresh untouched 50-seed bank compare native
`1x`, the v37 champion, and the selected controller. Identical controllers share
one executed receipt. Search executes exactly 75 new rollouts total; held-out
evaluation executes at most 450. No tuning is permitted on held-out outcomes.
