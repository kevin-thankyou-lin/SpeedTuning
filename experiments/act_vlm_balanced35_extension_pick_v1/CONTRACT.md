# Pick balanced-3.5 post-hoc extension

This extension tests the user-disclosed schedule `[2.5,1.5,3.5,3.5]` after
the targeted Pick screen completed.

- Frozen policy, learned phase detector, controller, task, success criterion,
  and speed grid are unchanged from the parent targeted screen.
- Parent root is the exact `cda7f3e0...` targeted-screen run. Its identity,
  completion, discovery, result, and gate-state hashes are frozen into the new
  identity before any candidate outcome is opened.
- Candidate seeds and object poses are exactly the parent's gate bank
  `140220100..140220119`.
- The parent's matched native `1x` controls are reused byte-for-byte. No native
  rollout is reexecuted.
- Candidate schedule is exactly `[2.5,1.5,3.5,3.5]`.
- Candidate gate is staged `5 -> 10 -> 20`: `0..2/5` rejects, `0..8/10`
  rejects, and `0..17/20` rejects. `18..20/20` qualifies for continued search.
- Maximum new scientific exposure is 20 candidate rollouts. Cache hits do not
  increment exposure, and no seed may be rerun.
- A physics error or unclassified runtime safety event halts the lane. A
  workspace exit counts as a candidate failure and prevents promotion.
- The reserved final seeds `140210000..140210099` remain unopened. A qualified
  result is not certification or deployment evidence.

