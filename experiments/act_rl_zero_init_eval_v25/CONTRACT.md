STRIDER ACT RL zero-initialization control v25

This study evaluates the actual zero-training greedy Tabular policy and a newly
seed-fixed, untrained Rainbow network on the exact v22/v23/v24 50-seed bank.

The Tabular Q table is all zeros. The registered speed order makes its
deterministic argmax select native 1x in every phase. Rainbow is constructed
with an explicit PyTorch seed immediately before network initialization, uses
the architecture registered for v20, retains zero/one observation
normalization, and is put in evaluation mode. It is a reproducible new random
initialization, not a reconstruction of the uncheckpointed historical v20
episode-0 weights.

All six controllers must be hash-frozen before any v25 final episode. No
training rollout is permitted. v24 must be sealed before v25 starts, and the
single L40 lane must remain non-overlapping.
