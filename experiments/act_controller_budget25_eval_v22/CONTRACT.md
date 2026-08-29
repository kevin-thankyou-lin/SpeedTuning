STRIDER v22: episode-25 learned-controller evaluation

- Freeze the exact Tabular and Rainbow controller state after the first 25 immutable v20 online-training receipts for Pick, Tea, and Insertion.
- Reconstruct Tabular by replaying the registered first-visit Monte Carlo updates over exactly that prefix.
- Extract Rainbow from the hash-bound `episode-25.pt` resume checkpoint without further optimization.
- Seal all six controller artifacts before opening any v22 final seed.
- Evaluate both methods on the same fresh 50-seed bank within each task: Pick `267000000..049`, Tea `267000100..149`, and Insertion `267000200..249`.
- Count simulator-invalid attempts as failures and retain every safety incident.
- Never re-execute or modify a v20 search rollout. Resume only a contiguous, identity-matched v22 final prefix.
- Interpret this as a 25-rollout training-budget retrospective reproduction because the completed v20 results were already observed; the v22 final banks themselves remain untouched at freeze time.
