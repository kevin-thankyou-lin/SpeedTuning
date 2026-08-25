# Paired-uniform reporting repair v2

This zero-rollout audit repairs one aggregate wording defect in the completed
Pick paired-uniform comparison.

- Source is the exact terminal v1 `RESULT.json` produced by commit
  `97b0cb2fb8d5e8a13c1e97b719af1ad3abfd7f8e`.
- All candidate schedules, completed counts, successes, verdicts, speedups,
  safety counts, and physics counts are copied byte-for-byte.
- No rollout, model load, simulator execution, native control, or final seed is
  permitted.
- If no uniform schedule qualifies but the adaptive schedule qualifies, report
  `balanced_wins_registered_gate_over_all_uniforms=true`; report
  `balanced_strictly_beats_best_uniform=null` because no qualified uniform
  exists for that narrower comparison.
- Recompute the reliability-first selection and Pareto frontier from the frozen
  candidate table, then hash both the source and repaired report.

