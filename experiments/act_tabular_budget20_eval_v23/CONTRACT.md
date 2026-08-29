STRIDER v23: Tabular controller after 20 training episodes

- Reconstruct each task's exact Tabular controller from the first 20 immutable v20 training receipts.
- Freeze and hash all three controllers before opening an evaluation cell.
- Reuse the exact v22 50-seed bank for a paired episode-20 versus episode-25 budget comparison.
- Require v22 to be sealed before launch; never inspect partial v22 outcomes to alter a controller.
- Execute 150 new final rollouts and zero new training rollouts.
- Preserve all v20 and v22 states. Resume only a contiguous identity-matched v23 prefix.
- Report this as a post-hoc paired learning-curve extension, not an independently preregistered benchmark.
