# ACT speed benchmark task list

This file is the handoff contract for the persistent tmux Codex supervisor.
Complete stages in order.  Never launch a later stage when an earlier gate is
open, and never duplicate a healthy workflow.

## Stage 0: freeze and audit

- [x] Record all three ACT roots, hashes, camera/order/progress/action contracts.
- [x] Independently audit the sealed 49/50, 50/50, 49/50 receipts.
- [ ] Emit one immutable run manifest containing Git, ACT, phase-detector,
  environment, controller, contract, and seed-bank hashes.

## Stage 1: ACT speed-adapter parity

- [ ] Add an ACT-specific speed adapter with 15-D qpos, nominal-time progress,
  all three policy cameras, per-step inference, and overlapping 100-step
  temporal ensembling.
- [ ] Keep detector rendering independent: the learned detector may consume
  only `angle`, but ACT must still receive both wrists.
- [ ] Add first-success-step and safety accounting; physics exceptions become
  rollout failures and do not abort a bank.
- [ ] Pass unit tests and uniform-1x action parity tests.
- [ ] Reproduce Pick 49/50, Tea 50/50, Insertion 49/50 on the accepted banks.

## Stage 2: matched search

For every task and every method below, spend exactly 50 search rollouts on the
contract's search seeds.  Search receipts are atomic and resumable.  Do not use
the final seeds for debugging, ranking, or repair.

- [ ] Uniform speed sweep.
- [ ] Learned-phase subtask schedule search.
- [ ] Learned-phase tabular RL.
- [ ] Learned-phase Rainbow RL.
- [ ] Offline AWE proxy.  Label this as a proxy, not full SAIL.
- [ ] SAIL-inspired adaptive speed modulation with its distinct executable
  contract and provenance.

All phase-dependent methods must verify the frozen learned-detector hash.  A
method that does not need a phase detector must not receive detector outputs.
The 50-rollout accounting is method-specific and must be preregistered before
the first episode: a sweep may divide the bank among candidates, whereas an
online RL method uses all 50 as sequential training episodes and freezes its
terminal checkpoint (never a retrospectively selected checkpoint).  A
candidate evaluated on ten matched states must pass at least 9/10; any safety
violation rejects it.  Preserve the fastest observed best-effort result
separately from the selectable candidate.  The untouched 50-state final bank is
the only common reliability estimate across all methods.

## Stage 3: untouched final evaluation

- [ ] Freeze one selected artifact per task/method before opening final seeds.
- [ ] Evaluate exactly 50 fresh final rollouts per task/method.
- [ ] Evaluate one shared uniform-1x native reference on the same final states.
- [ ] Seal per-state results, aggregates, hashes, and completion markers.
- [ ] Report SR and success-only first-success-step speedup for every cell.

## Stage 4: review and report

- [ ] Independently validate episode counts, seed disjointness, checkpoint
  hashes, detector usage/non-usage, and absence of duplicate episodes.
- [ ] Produce a task by method table with SR, speedup, safety, physics errors,
  search budget, and final budget.
- [ ] Link every aggregate to its manifest and per-state receipts.

The Diffusion Policy implementation/training is a separate matched study and
must not block ACT speed benchmarking after Stage 1 parity passes.
