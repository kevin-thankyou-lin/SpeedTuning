# Multiview ACT baselines

These are the frozen base policies for ACT-backed SpeedTuning experiments.  Do
not replace them with scripted policies or older top-camera checkpoints.

## Accepted policies

All policies were produced from source commit
`228f51f3ea3f56efee291dd2409e49534d024cda`.  They consume, in this exact
order, `angle`, `left_wrist`, and `right_wrist` RGB images plus 14 measured
joint positions and normalized episode progress.  They predict absolute
14-joint commands in 100-action chunks and use ACT's overlapping temporal
ensemble with `m=0.01`.

| Task | Persistent checkpoint root | Best epoch | Fresh final bank | SR |
| --- | --- | ---: | --- | ---: |
| Pick | `/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-pick-3pv-wrists-20260824-v1/pick` | 1797 | `9150000..9150049` | 49/50 (98%) |
| Tea | `/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-tea-3pv-wrists-20260823-v1/tea` | 1965 | `9250000..9250049` | 50/50 (100%) |
| Insertion | `/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-insertion-3pv-wrists-20260824-v1/insertion` | 1636 | `9350000..9350049` | 49/50 (98%) |

Each root contains `checkpoints/policy_best.ckpt`,
`checkpoints/dataset_stats.pkl`, `checkpoints/policy_config.json`, the sealed
dataset and training summaries, `evaluation.json`, `summary.json`, and
`SHA256SUMS`.

### Policy artifact hashes

| Task | `policy_best.ckpt` | `dataset_stats.pkl` | `policy_config.json` |
| --- | --- | --- | --- |
| Pick | `01f73838acd4c50b4b0db815f2ae9c845d343fb7f00983ee30736d13f34dbd89` | `1aa06430677e631c6aabb082f6c27b21cba4d287a4e49fb48379e8ad206299c8` | `994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180` |
| Tea | `f6ed29c07bd4a840fd05ca0b6308c729d81ed3d703ec4f9a29f12a3b0504f596` | `6f6a9e2e8a75a3194e3215200e575da2cda296e56413ae57f0c5be24c678cae0` | `994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180` |
| Insertion | `013ae8dfb88383fb3ed01498285d82a35dd19de1d12ad0ddeb3758151907e0ca` | `35ef807f30ba564f713a326b93a5c6b1e7200a2bb3c759b1529621d5f0c3222a` | `994e00f5d8ba6f26d7ef067d2819470d551b087b407248f737c723230936b180` |

The independent AMLFS audit is frozen in
`osmo/audit_multiview_act_artifacts_l40.yaml`; successful receipt:
`speedtuning-audit-multiview-act-artifacts-20260824-v2-1`.  The earlier `v1`
receipt is intentionally retained: it failed closed because it used the wrong
Tea date.

## Training recipe

- 270 HDF5 demonstrations per task, split into 250 train and 20 validation
  episodes.
- `angle + left_wrist + right_wrist` RGB at 640 by 480.
- 14 measured joints plus raw normalized episode progress (`qpos_dim=15`).
- Absolute 14-joint action targets, `num_queries=100`.
- Original ACT loss reduction and ACT optimizer configuration; 2,000 epochs,
  batch size 8, 64,000 optimizer updates.
- Pick and Tea collection replayed 270/270 successfully.  Insertion preserved
  its ordinary collection outcomes: 264/270 source successes and 265/270 joint
  replay successes.

## Pure evaluation reproduction

Use the frozen environment and code from source commit `228f51f...`.  Verify
the selected root before loading it:

```bash
(cd "$ROOT" && sha256sum -c SHA256SUMS)
```

Then run one of the following without writing into the sealed root:

```bash
# Pick
ROOT=/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-pick-3pv-wrists-20260824-v1/pick
.venv/bin/python -m scripts.evaluate_original_act \
  --task pick_and_place --checkpoint-dir "$ROOT/checkpoints" \
  --checkpoint-name policy_best.ckpt --output /tmp/pick-act-repro.json \
  --num-rollouts 50 --seed-base 9150000 --progress-condition \
  --camera-names angle left_wrist right_wrist

# Tea
ROOT=/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-tea-3pv-wrists-20260823-v1/tea
.venv/bin/python -m scripts.evaluate_original_act \
  --task tea_bag --checkpoint-dir "$ROOT/checkpoints" \
  --checkpoint-name policy_best.ckpt --output /tmp/tea-act-repro.json \
  --num-rollouts 50 --seed-base 9250000 --progress-condition \
  --camera-names angle left_wrist right_wrist

# Insertion
ROOT=/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-insertion-3pv-wrists-20260824-v1/insertion
.venv/bin/python -m scripts.evaluate_original_act \
  --task insertion --checkpoint-dir "$ROOT/checkpoints" \
  --checkpoint-name policy_best.ckpt --output /tmp/insertion-act-repro.json \
  --num-rollouts 50 --seed-base 9350000 --progress-condition \
  --camera-names angle left_wrist right_wrist
```

Expected successes are Pick 49, Tea 50, and Insertion 49.  The original
evaluator records the full horizon in `steps`; the acceleration benchmark must
also record the first successful controller step so speedup is not computed
from three identical fixed horizons.

## Speed-wrapper admission gate

No acceleration search may begin until the ACT speed wrapper demonstrates
uniform-1x parity on all three accepted 50-state banks.  The wrapper must retain
all three cameras, the progress scalar, per-physics-step ACT inference, and the
overlapping temporal ensemble.  The generic chunk runner does not satisfy this
contract.

