# Recovery ledger

- The first correctly pinned grouped workflow used Pick as its lead task.
  Pick completed first, which ended the group after Insertion had sealed 43 of
  its 50 final receipts. Insertion had no failure and its search plus 43 final
  receipts remain immutable.
- `act_strider_baseline_insertion_resume_l40.yaml` requires that exact partial
  state: matching identity and selection receipts, exactly 43 final receipts,
  and no result/completion marker. The benchmark runner verifies every cached
  seed and executes only the seven missing final seeds.
- Tea's first correctly pinned attempt failed before its first rollout because
  its explicit simulator reset requires a 39-value state suffix. The Tea-only
  recovery uses the fixed reset contract under a fresh source identity, so no
  Tea rollout was rerun.
