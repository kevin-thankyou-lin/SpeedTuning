# Exploratory ACT VLM derivative-frontier expansion

This is a post-hoc exploratory extension. It is not part of the sealed
six-method ACT speed benchmark and must never be merged into that benchmark's
preregistered result table.

- Base policy: frozen multiview ACT artifacts from the passed benchmark
  manifest at source `866c9f436caf0a73e5e08ef83be38cbe89a23a61`.
- Task: Pick only.
- Runtime observation: the sealed causal RGB/proprio four-phase detector.
- Phase order: `pre_grasp`, `grasp_lift`, `transport`, `interaction`.
- Allowed speeds: `1, 1.5, 2, 2.5, 3, 3.5, 4`.
- Discovery seeds: `140100000..140100002`.
- Ranking seeds: `140100100..140100109`.
- First accelerated candidate: uniform `[2,2,2,2]`.
- Each candidate is evaluated on all three discovery scenes.
- Every ACT rollout runs the full horizon for safety auditing. Speed is measured
  by successful first-success steps, matching the sealed ACT benchmark metric.
- Frontier acquisition: raise one phase by one adjacent rung. Rank proposals by
  conservative expected absolute steps saved, computed from measured phase
  workload and a VLM-estimated safe-success probability.
- The acquisition score is not rollout evidence. A proposal must actually run.
- Preserve six discovery episodes for a one-phase causal backoff ladder.
- Rank only the runner-designated base/backoff finalists on the same ten fresh
  poses. Qualification requires zero safety violations and at least `9/10`.
- Native `1x` remains the deployment fallback if no accelerated finalist
  qualifies.
- Budget: at most 50 native/candidate/ranking rollouts before any later final
  benchmark. No original search or final seed is reused.
