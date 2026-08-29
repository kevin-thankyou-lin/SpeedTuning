# Tabular and Rainbow reproduction on the v18 final bank

The original selected controller bytes are inaccessible under the current Osmo
identity. This study therefore retrains reproductions; it does not claim to
replay the original frozen Tabular or Rainbow policies.

1. Use the exact hash-matching Pick, Tea, and Insertion ACT artifacts and frozen
   learned-phase detector used by v18.
2. Run the unchanged registered Tabular and Rainbow training implementations on
   their original ordered 50-seed search banks. Do not inspect v18 outcomes
   during training or controller sealing.
3. Seal each terminal controller and completion receipt before its v18 final
   cell can open.
4. Evaluate the six sealed reproductions on exactly the 50 primary v18 seeds;
   never rerun native, uniform, STRIDER, or any cached v18 episode.
5. Preserve simulator/physics incidents and safety violations. A simulator-
   invalid primary attempt invalidates an exact paired claim; it is not silently
   retried or replaced.
6. Report these as retrained reproductions and compare them to the sealed v18
   controller receipts only after all six final cells have valid completion
   hashes.

The authorized new budget is 300 training episodes plus 300 final episodes.
