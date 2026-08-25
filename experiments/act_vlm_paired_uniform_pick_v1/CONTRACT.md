# Paired uniform comparison for the Pick adaptive frontier

This post-hoc comparison answers whether the qualified adaptive schedule is
better than standard uniform execution on exactly the same twenty gate poses.

- Frozen ACT policy, learned detector, controller, phase definitions, task,
  and success criterion are unchanged.
- The exact targeted-screen parent (`cda7f3e0...`) supplies twenty matched
  native controls and frozen poses for seeds `140220100..140220119`.
- The exact balanced extension (`0c76a17...`) supplies the already-paid
  `[2.5,1.5,3.5,3.5]` result. It is hash-pinned and never rerun.
- Two new schedules are registered before outcomes: uniform `2x`, followed by
  uniform `2.5x`. Each sees the same twenty parent poses.
- Each uniform uses the registered `5 -> 10 -> 20` gate. `18..20/20` qualifies;
  lower counts reject. No failed or cached rollout is replaced.
- At most forty new candidate rollouts are permitted. Native reruns are zero.
- Selection is reliability first: among clean, qualified, complete schedules,
  maximize exact success count, then matched-native speedup, then schedule
  simplicity. A separate Pareto frontier reports reliability/speed tradeoffs.
- The adaptive schedule strictly beats the best uniform only if its success
  count and matched-native speedup are both no lower, with at least one strict.
- Any physics error or unclassified runtime safety event halts the lane.
- Reserved final seeds `140210000..140210099` remain unopened. This paired
  search-bank comparison is not final certification or deployment evidence.

