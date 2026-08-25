# Frozen-ACT fixed-uniform final-bank contract

This lane fills the missing paper-table rows for fixed uniform execution at
`1.5x`, `2x`, `2.5x`, and `3x` on Pick, Tea, and Insertion.

- The ACT policy, learned phase detector, simulator, action representation,
  success criterion, and task-specific 50-seed final banks are hash-pinned by
  the existing benchmark manifest.
- All four speeds are registered before any result from this lane is read.
  Every speed runs all 50 seeds; there is no selection or early stopping.
- The exact existing same-seed native `1x` receipts are verified and reused.
  No native rollout is rerun.
- Each speed is a separate reported baseline. Results may not be collapsed into
  a post-hoc best-uniform row without also showing all four constituent rows.
- Successful-rollout speedup is reported beside success count. Achieved
  throughput is successes divided by total charged episode steps: success is
  charged through first success and failure through its terminal horizon.
- Safety violations and physics errors are preserved. No failed episode may be
  replaced or rerun.

