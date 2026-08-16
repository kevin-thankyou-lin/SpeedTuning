# Online reference alignment

This repository includes a segment-independent V0 probe for mapping each live
causal video clip to a continuous position in one canonical reference video.
It does not train a semantic phase classifier and does not contain a playback
policy. Segment boundaries and speeds remain editable metadata.

## Dataset structure

`capture_reference_alignment_dataset.py` creates:

```text
dataset/
  manifest.json
  landmarks.csv
  trajectory-00-seed-.../
    trajectory.json
    angle/*.jpg
  trajectory-01-seed-.../
    trajectory.json
    angle/*.jpg
  ...
```

The first trajectory is a constant-1x reference. Query trajectories use a
deterministic piecewise timing warp. Each trajectory record contains frame
paths, wall time, and normalized scripted-policy time. Policy time is used only
as exact simulator correspondence truth; the encoder and online aligner never
receive it. No semantic segment labels are stored.

For real videos, replace `landmarks.csv` with manually marked sparse
correspondences using the same columns. These landmarks are evaluation data,
not training labels.

## V0 method

The smallest tested system is:

```text
trailing 1.0-second RGB clip
  -> frozen ImageNet ResNet18 per-frame features
  -> [clip mean, latest frame, latest-minus-earliest]
  -> cosine emission likelihood
  -> causal HMM-style monotonic filter
  -> continuous reference position and confidence
```

The filter permits at most one reference-frame backtrack and five-frame forward
motion per update. Its confidence combines posterior entropy, similarity
margin, posterior width, and recent path consistency.

`R3D-18` at 0.5 seconds is also implemented as a frozen video-encoder baseline.
It was slower and less accurate than temporally pooled ResNet18 on this small
benchmark.

## Reusable API

The core class consumes clip embeddings, so encoders can be changed without
changing segment metadata:

```python
from reference_alignment import OnlineReferenceAligner

aligner = OnlineReferenceAligner(reference_clip_embeddings)
result = aligner.update(live_clip_embedding)

result.reference_position
result.reference_index
result.confidence
result.local_progress_rate
```

Raw-frame use is supported by supplying a causal `clip_encoder` callable. It is
called only with the current and preceding frames.

Segment lookup is deliberately separate:

```python
from reference_alignment import ReferencePositionSpeedPolicy

speed_policy = ReferencePositionSpeedPolicy(
    [(0.20, 0.35, 1.0), (0.70, 0.80, 1.5)],
    default=4.0,
)
speed = speed_policy(result.reference_position)
```

Editing those intervals requires no model retraining.

## V0 benchmark

The 2026-08-16 probe used one reference and five query executions per task.
Insertion had one failed query execution, which was preserved but excluded from
the successful-trajectory alignment aggregate.

Best frozen RN18 result, using a trailing 1.0-second window:

| Task | Mean error | Median error | 90th percentile | Catastrophic jumps | Encoder throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pick-and-place | 2.04% | 1.73% | 3.72% | 0 | 379.5 frames/s |
| Randomized tea | 2.86% | 2.03% | 6.88% | 0 | 388.6 frames/s |
| Insertion | 2.25% | 2.05% | 4.46% | 0 | 380.6 frames/s |

The unconstrained global nearest neighbor had lower framewise error, but its
trajectory contained unstable backward and occasional large jumps. The causal
filter traded a small amount of instantaneous accuracy for a stable coordinate.
Low-confidence frames had substantially higher error than high-confidence
frames, so confidence is useful directionally but is not yet calibrated as a
probability.

## Decision and limitations

V0 supports the hypothesis on these in-domain simulator trajectories. V1 TCC
adaptation is not justified by this benchmark alone, and V2 unmatched/reversal
states were not tested because successful queries contained no deliberate extra
actions or reversals.

The next falsification test should use real held-out people, camera/background
variation, and manually annotated correspondence landmarks. Add V1 only if
those errors indicate an appearance-correspondence problem. Add V2 only when
observed pauses, repetitions, skipped actions, or divergence break the causal
path.

## Commands

Capture one segment-free task dataset:

```bash
python scripts/capture_reference_alignment_dataset.py \
  --task pick_and_place \
  --seeds 100 101 102 103 104 105 \
  --output data/reference-alignment-pick
```

Benchmark frozen encoders:

```bash
python scripts/benchmark_reference_alignment.py \
  --dataset data/reference-alignment-pick \
  --output results/reference-alignment-pick \
  --encoders rn18_temporal_pool r3d18 \
  --windows 0.25 0.5 1.0 \
  --video-windows 0.5 \
  --device cuda
```
