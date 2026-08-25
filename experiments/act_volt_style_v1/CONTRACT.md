# Learned-phase VOLT-style baseline contract

This baseline tests the binary timing abstraction suggested by VOLT while
keeping the frozen ACT policy used by the rest of this benchmark. It is
therefore labeled **VOLT-style (frozen policy)** rather than a paper-faithful
VOLT reproduction: [original VOLT](https://arxiv.org/abs/2606.06323) uses
vision-language segmentation to reformat demonstrations and retrains an
imitation policy, whereas this experiment applies the already validated
learned phase detector at execution time.

## Controller family

The learned detector exposes four causal phases in this fixed order:

1. `pre_grasp` -- fast;
2. `grasp_lift` -- slow;
3. `transport` -- fast;
4. `interaction` -- slow.

Every candidate has exactly two independent values, `fast_speed` and
`slow_speed`, and executes the schedule
`[fast_speed, slow_speed, fast_speed, slow_speed]`. Uniform schedules are the
special case `fast_speed == slow_speed`.

## Search and selection

- Use the same fresh ordered 15-seed search bank as STRIDER for a paired
  comparison, while executing independent VOLT-style rollouts.
- Use the same staged gates: at least 4/5 and 9/10 to continue and at least
  14/15 to qualify.
- Reject any candidate containing a safety event; halt on a physics error.
- Charge failed episodes through their full terminal horizon in achieved
  throughput.
- Begin with uniform 2x. Climb through uniform 2.5x and 3x while candidates
  qualify; if 2x rejects, test uniform 1.5x.
- After the first rejected uniform, reduce the shared slow speed by one
  adjacent rung while retaining its fast speed. If all uniforms qualify,
  promote only the shared fast speed by one adjacent rung.
- Retain the best qualified uniform as an explicit incumbent. The two-speed
  candidate replaces it only if it qualifies, matches or exceeds the
  incumbent's success count on the same search bank, has no incidents, and
  strictly improves failure-aware achieved throughput.
- Search executes at most 60 new rollouts per task.

## Final evaluation and shared controls

Selection is atomically sealed before any final-bank receipt is read. The
selected two-speed controller is evaluated once on the same untouched 50-seed
bank as STRIDER. Exact native and fixed-uniform controller receipts are shared
by schedule, policy, detector, environment, and seed hash; they are not rerun.
If VOLT-style selects a fixed uniform schedule, its final receipts are shared
entirely and it executes zero duplicate final rollouts.

Report success rate, failure-aware throughput change relative to native 1x,
successful-rollout speedup, all safety and physics incidents, search cost,
new final cost, and shared cache hits. The held-out empirical frontier is
descriptive and cannot alter the sealed search selection.
