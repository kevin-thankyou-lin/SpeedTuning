# Phase-adaptive 5 -> 10 -> 15 Pick search

This is a prospective method-validation lane over the frozen multiview ACT Pick
policy. It is exploratory and separate from the sealed benchmark and earlier
VLM-targeted studies.

## Search prior

- The causal RGB/proprio detector chooses among the observed phases
  `pre_grasp`, `grasp_lift`, `transport`, and `interaction` online.
- The proposal layer freezes only qualitative phase-risk labels:
  `cautious`, `protected`, `open`, `open`.
- Numeric schedules are generated mechanically across global aggression levels
  `2, 2.5, 3, 3.5, 4`, capped by label at `2.5, 1.5, 4, 4` respectively.
- Uniform `2x` and `2.5x` are mandatory comparators. Candidate definitions and
  order are frozen before outcomes.
- Existing Pick evidence informed the qualitative labels, so this lane does
  not claim that an independent VLM rediscovered the schedule. The reusable
  method accepts blind qualitative labels on a new task.

## Reliability and accounting

- Search seeds are exactly `140230000..140230014`, in that order, shared by
  every schedule and matched native `1x`.
- Reserved final seeds are exactly `140240000..140240099` and remain unopened.
- Every candidate first runs five trials. `0..3/5` rejects and `4..5/5`
  continues. At ten, `0..8/10` rejects and `9..10/10` continues. At fifteen,
  `0..13/15` rejects and `14..15/15` qualifies for continued search.
- These are preregistered candidate-selection gates, not a confidence interval,
  certification test, or deployment claim.
- A candidate is never rerun. Immutable receipts recover crashes without
  consuming a seed twice. Maximum pre-final consumption is 120 rollouts:
  fifteen native plus seven candidates times fifteen; early rejection should
  consume less.
- Any physics error or runtime safety incident halts the lane. Workspace exits
  count as failed accelerated trials and prevent that candidate from qualifying.
- Selection ranks observed reliability before matched-native successful-rollout
  speedup. A phase-adaptive candidate cannot displace a qualified uniform with
  lower success count or lower speedup.
