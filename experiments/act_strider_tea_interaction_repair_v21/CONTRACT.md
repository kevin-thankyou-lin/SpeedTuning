# Tea interaction-speed causal repair v21

This is a post-v18 causal diagnostic, not an independent final benchmark.  The
three v18 Tea failures motivated exactly one change to the frozen STRIDER
schedule: lower only the interaction phase from `2.0x` to `1.5x`.

1. Freeze the ACT policy, learned phase detector, centered-cup success rule,
   source hashes, and the two schedules before opening any v21 outcome.
2. Compare incumbent `[2.0, 2.5, 1.5, 2.0]` with interaction repair
   `[2.0, 2.5, 1.5, 1.5]`; do not propose another schedule from v21 outcomes.
3. Build a fresh outcome-blind 16-pose panel from the declared independent
   uniform Tea reset range.  Its nested prefixes are `4 -> 8 -> 16`.
4. Run both controllers on every reached prefix in the same pose order.  Cache
   keys bind schedule, pose, policy, detector, environment, and source hashes;
   cache hits never increment scientific rollout accounting.
5. Stop the repair for a safety/physics incident, at `<=2/4`, or at `<=6/8`.
   At 16, require at least `15/16`, zero incidents, and strictly more successes
   than the incumbent to select the repair.  An equal success count retains the
   incumbent because the repair is slower by construction.
6. The maximum new budget is 32 scientific rollouts.  Preserve all v17, v18,
   and v20 roots.  This diagnostic never opens or reuses a final bank.
