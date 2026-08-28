# STRIDER frozen-selection matched final-50 evaluation v15

This run evaluates the already frozen v13 and v14 STRIDER selections. It does
not tune, select, repair, or promote a schedule using these outcomes.

1. Verify the complete v13 and v14 identities, selection hashes, completion
   hashes, common search bank, search-only status, and parent-child link.
2. Freeze four named controllers: native `1x`, the v13 uniform incumbent, the
   v13 selected schedule, and the v14 selected schedule.
3. Deduplicate schedule-identical controllers before execution and retain their
   method names as aliases in the result. This yields four unique controllers
   for Pick and three each for Tea and Insertion.
4. Evaluate every unique controller on the same ordered 50 fresh reset poses.
   A pose with a simulator-invalid result from any controller is excluded from
   all controllers and replaced only from the registered reserve.
5. Count failures and their elapsed time in achieved throughput. Report success
   rate, successful-rollout speedup, and failure-aware throughput relative to
   matched native execution.
6. Preserve all state receipts and videos, report physical attempts separately
   from scientific rollouts, and verify that parent selections remain unchanged.

The ACT policy, learned causal RGB/proprio phase detector, schedule semantics,
task success checkers, and simulator-error policy are frozen. Tea uses the
registered teabag-center-inside-oriented-mug-volume success criterion.
