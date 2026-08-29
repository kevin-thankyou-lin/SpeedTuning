# STRIDER replay on the episode-20/25 RL bank

This is a post-hoc, same-seed comparison of the already-frozen v17 STRIDER
controllers against the sealed v22/v23 low-budget RL controllers.

1. Read each STRIDER schedule only from the hash-verified v18 identity and
   completion receipts staged for v20. Do not tune or search.
2. Freeze all three STRIDER schedule receipts before opening any evaluation
   cell.
3. Evaluate exactly one STRIDER controller on each task's 50 v22 final seeds.
   Do not rerun native, Tabular, Rainbow, v20, v22, or v23 episodes.
4. Count simulator-invalid attempts as failures and retain every safety event.
   Resume only a contiguous identity-matched prefix.
5. After all three cells seal, compare STRIDER against Tabular-20, Tabular-25,
   and Rainbow-25 using the exact paired state receipts. Report success
   discordance and time-to-success only on pairs where both methods succeed.
6. This replay evaluates frozen controllers after prior results were known. It
   is a paired benchmark extension, not a new independent method-selection
   experiment.
