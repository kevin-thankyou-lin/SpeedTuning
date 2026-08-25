# Uniform 3.5x paired-final extension contract

This extension fills the only missing fixed-uniform point requested for the
paper plot. It evaluates schedule `[3.5, 3.5, 3.5, 3.5]` on each task's exact
frozen STRIDER v4 50-seed final bank. It does not rerun native controls, alter
STRIDER selection, inspect final outcomes during search, or open any new bank.

The result is combined only with the same-seed STRIDER v4 native control when
computing successful-rollout speedup and failure-aware throughput delta. Every
episode runs to first success or the normal terminal horizon. Safety and
physics incidents remain explicit. Each task has exactly one versioned attempt.
