# STRIDER Codex diverse four-reset search v13

This is a tiny, search-only schedule iteration for frozen ACT policies on Pick,
Tea, and Insertion. It is not a reliability benchmark or certification run.

## Outcome-blind diverse reset panel

- For each task, sample a frozen pool of 64 fresh simulator resets without
  executing the policy or reading reward, success, or trajectory outcomes.
- Remove invariant initial-pose coordinates and min-max normalize each varying
  coordinate over the frozen pool.
- Select the first reset farthest from the pool centroid, then greedily select
  each subsequent reset to maximize its distance to the nearest selected reset.
  Ties use the lowest seed. Freeze four resets and their complete pose receipt.
- Pick and Tea therefore cover their varying 2-D object-position spaces;
  Insertion covers the varying 4-D joint position space of its two objects.
- Every schedule sees the same ordered four resets. A simulator-invalid attempt
  is excluded and replaced only from the registered reserve.

## Search

1. Require strict `4/4` success and zero safety incidents for eligibility.
2. Start at uniform `2x`; continue through `2.5x`, `3x`, and `3.5x` only while
   every candidate is `4/4`. If `2x` fails, test uniform `1.5x` as fallback.
3. On the first rejected uniform candidate, compare blinded Codex video
   attribution with same-seed telemetry attribution and test each distinct
   one-phase, one-rung repair.
4. A repair replaces a `4/4` uniform incumbent only if it is also `4/4` and
   improves failure-aware throughput by at least three percent on this panel.
5. Cap the search at 32 simulator-valid rollouts per task. Do not open the final
   bank; selected schedules require a separate held-out evaluation.

The frozen policy, learned causal phase detector, speed set, success checkers,
Codex observation restrictions, and simulator-error policy are unchanged from
v11. Tea uses the center-inside-oriented-mug-volume criterion.
