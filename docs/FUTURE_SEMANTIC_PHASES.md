# Future semantic phases for chunk tempo control

This prototype predicts only semantic phase IDs.  It does not regress object
poses, contact, phase progress, or a latent future state.

## Contract

At every receding-horizon decision, the frozen task policy proposes an action
chunk.  A phase predictor consumes causal observation features plus that
proposed chunk and returns one stable phase ID at each configured nominal
chunk offset:

```text
offset       0          1          2                3
phase        transport  transport  preinsert_align  insert
```

`FuturePhaseSequencePredictor` bundles one categorical head per offset.
Training labels are read from the recorded future trajectory, but runtime
prediction uses only the current observation and proposed action chunk.

Use `--label-mode semantic-phase` when records contain a stable
`semantic_phase_id`.  `reward-phase4` remains available as the existing
four-stage simulator probe.  For example:

```bash
python scripts/train_supervised_phase_intent.py \
  --dataset /path/to/phase-bank \
  --action-features /path/to/action-features.npz \
  --label-mode semantic-phase \
  --future-offsets 0,1,2,3 \
  --output /path/to/future-phase-model
```

The output contains `visual.future-phases.pkl`,
`action.future-phases.pkl`, and `fused.future-phases.pkl`, plus untouched-seed
metrics for every future offset.

## Runtime stride rule

`SemanticPhaseStrideController` maps each phase ID to a configured chunk
stride.  Before applying the current phase's stride, it checks every action
that would be skipped.  It falls back to native speed when:

- a skipped action is predicted to enter a phase with a lower speed;
- a required future offset was not predicted;
- a phase ID is unknown; or
- either gripper command changes inside the skipped interval.

`SemanticPhaseChunkRunner` predicts a fresh chunk, predicts its future phase
IDs, chooses one safe stride, and executes that stride until the next replan.
There is no independent framewise speed signal.

```python
from chunked_policy import SemanticPhaseChunkRunner
from semantic_phase import SemanticPhaseStrideController

controller = SemanticPhaseStrideController(
    {
        "approach_object": 3.0,
        "pregrasp_align": 1.0,
        "grasp_confirm": 1.0,
        "transport_object": 3.0,
        "preinsert_align": 1.0,
        "insert_object": 1.0,
        "retract": 3.0,
    }
)
runner = SemanticPhaseChunkRunner(
    task_chunk_predictor,
    runtime_phase_predictor,
    controller,
)
```

The first scientific evaluation should freeze the semantic ontology and phase
speed table before using blind rollout seeds.  The current implementation is a
model/runtime contract and does not claim a trained semantic predictor result.
