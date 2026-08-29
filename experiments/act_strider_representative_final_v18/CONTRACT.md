# STRIDER representative-panel untouched final bank v18

This is the prospective final holdout for the frozen v17 representative-panel
selections. The panel redesign itself was motivated by earlier v16 outcomes, so
v18 evaluates v17 honestly but does not erase that post-v16 provenance.

1. Before opening any outcome, freeze 50 primary and 20 reserve simulator-reset
   seeds per task. These banks are mutually disjoint and disjoint from v16 search,
   v16 final, and v17 mathematical-panel IDs.
2. Verify each task's v17 `IDENTITY.json`, `SELECTION.json`, and `COMPLETE.json`
   hashes. Require `opens_final_bank=false`; never rewrite v17 artifacts.
3. Evaluate exactly three unique controllers on the same ordered reset poses:
   native `[1,1,1,1]`, uniform `[1.5,1.5,1.5,1.5]`, and the frozen v17 selected
   schedule. Do not add candidates, stop early, attribute failures, or tune.
4. Require 50 simulator-valid matched triples. If a reset/controller attempt is
   simulator-invalid, exclude the complete matched triple from the scientific
   denominator, preserve the attempt, and advance to the next frozen reserve.
   Safety/workspace violations remain counted failures and make that accelerated
   controller non-deployable even if the full descriptive bank is completed.
5. Report success count/rate beside every timing statistic. Compute successful-
   rollout speedup from the same-bank native successful mean first-success steps;
   do not hide failures inside the speed ratio.
6. The final scientific cost is 150 rollouts per task and 450 total when no
   simulator-invalid replacement is needed. Cache hits never increment physical
   attempts, and a completed bank is never reopened for selection.

