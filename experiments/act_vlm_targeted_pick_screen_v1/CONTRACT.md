# Targeted qualitative-VLM Pick screen

This is a post-hoc exploratory search over the frozen multiview ACT Pick policy.
It is not part of the sealed ACT benchmark, and it is not an independent test
of whether a VLM could invent the user's disclosed schedule.

## Candidate-generation contract

- The phase vocabulary is `pre_grasp`, `grasp_lift`, `transport`,
  `interaction`; allowed speeds are `1, 1.5, 2, 2.5, 3, 3.5, 4`.
- A future blinded VLM supplies coarse phase labels. Their frozen mapping is
  `protected=1.5`, `moderate=2.5`, `fast=3`, `ceiling=4`.
- The candidate `[3,1.5,4,4]` is explicitly registered as a
  `user_disclosed_hypothesis`, not as independent VLM discovery.
- The candidate `[3,1.5,3,3]` is the coarse-label ablation
  `[fast,protected,fast,fast]`; it is not claimed as an independent VLM output.
- Standard comparators are uniform `2x` and uniform `2.5x`. The already-known
  `[2.5,1.5,2.5,2.5]` schedule is included as a prior-incumbent comparator.
- Candidate order is frozen before outcomes: uniform `2x`, uniform `2.5x`,
  prior incumbent, coarse-label ablation, user-disclosed hypothesis.

## Rollout and selection contract

- Discovery seeds are exactly `140220000..140220002`; every candidate and
  native `1x` sees the same frozen three object poses.
- Gate seeds are exactly `140220100..140220119`; the selected candidate and
  matched native `1x` use the same frozen twenty poses.
- Reserved final seeds remain exactly `140210000..140210099` and are unopened.
- Discovery costs 18 rollouts: three native plus five schedules by three poses.
- Only schedules with `3/3`, zero safety events, and zero physics errors are
  eligible. Select the lowest mean successful first-success steps; break exact
  ties by fewer distinct speeds and then preregistered candidate order.
- The selected discovery schedule alone enters the fresh matched `5 -> 10 ->
  20` gate: at 5, `0..2` rejects; at 10, `0..8` rejects; at 20, `0..17`
  rejects and `18..20` qualifies for continued search.
- Total maximum new rollouts are 58: 18 discovery, 20 matched native gate
  controls, and at most 20 selected-candidate gate rollouts.
- A workspace exit is a failed candidate rollout. Any physics error or
  unclassified runtime safety event halts the lane. No candidate is rerun.
- `3/3` is discovery only and `18/20` is a search qualification, not
  certification or deployment. The final bank is never opened by this lane.

