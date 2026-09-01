# Frozen Rainbow-50 scripted-transfer correlation diagnostic V44

This is a mechanism diagnostic, not a new speed-policy search and not an ACT
performance replication. It loads the three terminal policies selected by the
sealed ACT Rainbow-50 benchmark at source `298c6d16784f228df0b1f455d0e41b4276ec5184`.

The old controller already consumed only the current learned four-phase one-hot
observation. No redundant training is permitted. Before any rollout, V44 must
validate each selected receipt, search completion receipt, terminal checkpoint
hash, detector hashes, observation dimension, and legacy speed grid.

For each task, decode the deterministic phase-to-speed map directly from the
four basis observations. Then run exactly ten fresh randomized scripted-policy
resets for native `1x` and the frozen Rainbow-50 policy on matched seeds. Log
every speed decision with nominal episode progress, learned phase, chosen speed,
Q values, Q margin, and phase transition. Report phase/speed counts, progress
deciles, phase occupancy, normalized mutual information, and descriptive
episode-clustered Spearman correlation between speed and progress.

The transfer deliberately changes the base controller from ACT chunked/FK to
the scripted waypoint controller/mocap fallback. The checkpoint environment
guard may be bypassed only in this diagnostic, with the exact differences
recorded in the source receipt. Results must be labelled out-of-distribution
mechanism evidence and may not be used to promote a deployable controller.

No historical rollout may be reexecuted. The V43 randomized Rainbow-100 workflow
and every V1-V43 artifact remain immutable. Resume only matching V44 receipts;
never launch a duplicate workflow.
