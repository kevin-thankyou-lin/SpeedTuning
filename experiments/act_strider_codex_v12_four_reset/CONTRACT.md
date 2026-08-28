# STRIDER Codex four-reset search iteration v12

This is a tiny, search-only schedule iteration for frozen ACT policies on Pick,
Tea, and Insertion. It is not a reliability benchmark or certification run.

## Four-reset panel

- Each task receives four fresh IID resets from its frozen uniform reset
  distribution. Seeds are registered before any rollout outcome is read.
- Every candidate sees the same ordered four primary resets. A simulator-invalid
  attempt is excluded and replaced only from the registered reserve.
- A candidate must succeed on all `4/4` resets with zero safety incidents to
  remain eligible. No result from four resets is called reliable or certified.

## Search

1. Start at uniform `2x` and continue the registered uniform ladder through
   `2.5x`, `3x`, and `3.5x` only while each candidate is `4/4`.
2. If `2x` fails, screen uniform `1.5x` as the immutable fallback.
3. On the first rejected uniform candidate, compare blinded Codex video
   attribution with same-seed telemetry attribution and test each distinct
   one-phase, one-rung repair.
4. A phase-conditioned repair may replace a `4/4` uniform incumbent only if it
   is also `4/4` and improves failure-aware throughput by at least three percent
   on the matched panel.
5. The valid-rollout search ceiling is 32 per task. The final bank remains
   unopened; any selected schedule requires a separate held-out evaluation.

The frozen policy, learned causal phase detector, speed set, success checkers,
Codex observation restrictions, and simulator-error policy are unchanged from
v11. Tea uses the center-inside-oriented-mug-volume criterion.
