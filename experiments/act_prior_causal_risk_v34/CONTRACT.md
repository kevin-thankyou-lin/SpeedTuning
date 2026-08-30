# Prior-guided causal risk-gated STRIDER v34

This prospective study freezes the ACT task policies, learned four-phase
detector, simulator, success criterion, and common speed grid
`[1, 1.5, 2, 2.5, 3]` used by v26-v33.

The method combines two outcome-blind priors. The semantic prior supplies the
initial schedule `[2, 1.5, 3, 1.5]`; the hash-pinned v33 SAIL-inspired precision
head ranks one-phase promotions and guards. Its 60 offline training rollouts
are disclosed and reused without re-execution. This is not paper-faithful SAIL.
The v34 method and risk-gate design were created after inspecting v33 results;
v34 is therefore a prospective evaluation on fresh banks, not an outcome-blind
reanalysis of v33. No v33 outcome is available to the runtime, candidate
generator, or v34 selection rule.

Each task receives exactly 25 new online search episodes:

1. Native reference, combined-prior warm start, and three outcome-conditioned
   one-phase promotions or causal backoffs run on the same three discovery
   seeds (`5 x 3 = 15`).
2. The two accelerated finalists each run on five additional registered seeds
   (`2 x 5 = 10`).

Every candidate uses the same frozen observation-only risk gate. It drops the
effective speed to `1x` at entry to a protected learned phase or immediately
after an observed gripper transition. It releases after one consecutive stable
observation. The gate sees no future action, terminal result, privileged
contact, or final-bank outcome. Causal failure repair lowers only the earliest
same-seed divergent phase by one registered rung. Successful candidates promote
one phase at a time by precision-weighted estimated saved steps.

The selection rule first prefers an `8/8` finalist, then failure-aware
throughput; `7/8` is the provisional floor. Zero safety and physics incidents
are mandatory. Search receipts are descriptive selection evidence, not a
reliability certification.

After all three searches seal, the same fresh untouched 50-seed bank evaluates:

- native `1x`;
- the selected phase schedule without the gate;
- the identical schedule with the gate.

This isolates the gate while keeping the selected schedule fixed. Search uses
75 new rollouts total and held-out evaluation uses 450. No v20-v33 search or
final rollout is reused or re-executed.
