# STRIDER three-task ACT baseline contract

This lane produces preliminary, receipt-bearing STRIDER numbers for the frozen
multiview ACT policies used by the sealed speed benchmark.

## Fixed policy and controller

- Tasks: Pick, Tea, and Insertion.
- The ACT checkpoint, dataset statistics, cameras, temporal ensemble, learned
  RGB/proprio phase detector, success detector, and simulator are hash-pinned
  by the passed benchmark manifest.
- ACT parameters are frozen. STRIDER changes only the execution speed selected
  at observable phase entries.
- Phases are `pre_grasp`, `grasp_lift`, `transport`, and `interaction`.
- Allowed speeds are `1, 1.5, 2, 2.5, 3, 3.5, 4`.

## Agent proposals and search

- The proposal receipt is frozen before this lane opens any search outcome.
- It contains uniform `2x` and `2.5x` comparators plus three semantic-risk
  schedules per task. The Pick proposal is explicitly a development-task
  proposal informed by prior discussion; it is not an independent discovery.
- Each candidate sees the same three discovery poses. Uniform `2x` is always
  first.
- The fastest clean `3/3` schedule becomes the causal-backoff base. If no
  schedule is clean `3/3`, the base is the completed schedule with the most
  successes, breaking ties by successful execution time.
- For an accelerated failure, attribution uses the earliest phase that fails to
  advance relative to its matched native trace. If the base has no discovery
  failure, the task-semantic protected phase is used for the registered
  reliability backoff. Only that phase is lowered, by one and two adjacent
  speed rungs.
- The base and first clean backoff are compared on the same ten fresh ranking
  poses. Qualification requires zero safety violations and at least `9/10`.
- Search exposure is at most 50 episodes per task, including native previews,
  uniform comparators, agent proposals, causal repairs, and ranking episodes.

## Evaluation and reporting

- After selection is frozen, STRIDER runs exactly 50 evaluations on the same
  task-specific final seed bank used by the sealed baselines.
- Existing same-seed native `1x` receipts are verified and reused; native
  rollouts are not repeated.
- Success-only speedup is reported beside success count.
- Achieved throughput is `successes / sum(episode metric steps)`, where a
  successful episode is charged through first success and a failed episode is
  charged through its terminal horizon. Throughput delta is relative to the
  reused native `1x` receipts.
- All safety violations and physics errors are reported. No failed rollout may
  be replaced or rerun.
- Because prior Pick development results were visible before this contract, the
  three-task table is preliminary. A paper claim requires a new preregistered
  benchmark or explicit development/test task separation.

