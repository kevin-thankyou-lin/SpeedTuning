# Frozen multiview ACT speed benchmark results

The table reports the untouched 50-state final banks. `FSS mean` and `Native
mean` are successful-rollout-only first-success physics steps. `Speedup` is
`Native mean / FSS mean`. Every task uses one shared matched native reference
on its final states. The `Safety` and `Physics` columns are final-bank counts;
all final and native banks were clean. Across the separate search banks there
was one physics error, in Tea `sail_inspired_adaptive`, and zero safety
violations. That search incident makes the Tea SAIL-inspired candidate
non-deployable despite its clean 49/50 final result.

Path aliases used below:

- `B` = `/mnt/amlfs-04/home/linke/speedtuning-act-speed-benchmark-v1/runs/298c6d16784f228df0b1f455d0e41b4276ec5184`
- `R` = `/mnt/amlfs-04/home/linke/speedtuning-act-speed-benchmark-v1/runs/866c9f436caf0a73e5e08ef83be38cbe89a23a61`
- `M-B` = `/mnt/amlfs-04/home/linke/speedtuning-act-speed-benchmark-v1/attempts/298c6d16784f228df0b1f455d0e41b4276ec5184/run_manifest.json`
- `M-R` = `/mnt/amlfs-04/home/linke/speedtuning-act-speed-benchmark-v1/attempts/866c9f436caf0a73e5e08ef83be38cbe89a23a61/run_manifest.json`

| Task | Method | Search | Final | Successes/50 | SR | FSS mean | Native mean | Speedup | Safety | Physics | Manifest | Selected artifact | Per-state evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Pick | `uniform_sweep` | 50 | 50 | 50/50 | 1.00 | 145.34 | 269.98 | 1.858x | 0 | 0 | `M-B` | `B/pick/uniform_sweep/search/selected.json` | `B/pick/uniform_sweep/final/states` |
| Pick | `learned_phase_subtask` | 50 | 50 | 50/50 | 1.00 | 201.30 | 269.98 | 1.341x | 0 | 0 | `M-R` | `R/pick/learned_phase_subtask/search/selected.json` | `R/pick/learned_phase_subtask/final/states` |
| Pick | `learned_phase_tabular_rl` | 50 | 50 | 50/50 | 1.00 | 177.38 | 269.98 | 1.522x | 0 | 0 | `M-B` | `B/pick/learned_phase_tabular_rl/search/selected.json` | `B/pick/learned_phase_tabular_rl/final/states` |
| Pick | `learned_phase_rainbow_rl` | 50 | 50 | 50/50 | 1.00 | 150.70 | 269.98 | 1.792x | 0 | 0 | `M-B` | `B/pick/learned_phase_rainbow_rl/search/selected.json` | `B/pick/learned_phase_rainbow_rl/final/states` |
| Pick | `awe_offline_proxy` | 50 | 50 | 50/50 | 1.00 | 260.06 | 269.98 | 1.038x | 0 | 0 | `M-B` | `B/pick/awe_offline_proxy/search/selected.json` | `B/pick/awe_offline_proxy/final/states` |
| Pick | `sail_inspired_adaptive` | 50 | 50 | 50/50 | 1.00 | 258.60 | 269.98 | 1.044x | 0 | 0 | `M-B` | `B/pick/sail_inspired_adaptive/search/selected.json` | `B/pick/sail_inspired_adaptive/final/states` |
| Tea | `uniform_sweep` | 50 | 50 | 44/50 | 0.88 | 305.02 | 441.78 | 1.448x | 0 | 0 | `M-B` | `B/tea/uniform_sweep/search/selected.json` | `B/tea/uniform_sweep/final/states` |
| Tea | `learned_phase_subtask` | 50 | 50 | 49/50 | 0.98 | 288.76 | 441.78 | 1.530x | 0 | 0 | `M-R` | `R/tea/learned_phase_subtask/search/selected.json` | `R/tea/learned_phase_subtask/final/states` |
| Tea | `learned_phase_tabular_rl` | 50 | 50 | 49/50 | 0.98 | 304.98 | 441.78 | 1.449x | 0 | 0 | `M-B` | `B/tea/learned_phase_tabular_rl/search/selected.json` | `B/tea/learned_phase_tabular_rl/final/states` |
| Tea | `learned_phase_rainbow_rl` | 50 | 50 | 46/50 | 0.92 | 288.28 | 441.78 | 1.532x | 0 | 0 | `M-B` | `B/tea/learned_phase_rainbow_rl/search/selected.json` | `B/tea/learned_phase_rainbow_rl/final/states` |
| Tea | `awe_offline_proxy` | 50 | 50 | 50/50 | 1.00 | 396.68 | 441.78 | 1.114x | 0 | 0 | `M-B` | `B/tea/awe_offline_proxy/search/selected.json` | `B/tea/awe_offline_proxy/final/states` |
| Tea | `sail_inspired_adaptive` | 50 | 50 | 49/50 | 0.98 | 388.37 | 441.78 | 1.138x | 0 | 0 | `M-B` | `B/tea/sail_inspired_adaptive/search/selected.json` | `B/tea/sail_inspired_adaptive/final/states` |
| Insertion | `uniform_sweep` | 50 | 50 | 39/50 | 0.78 | 190.05 | 355.65 | 1.871x | 0 | 0 | `M-B` | `B/insertion/uniform_sweep/search/selected.json` | `B/insertion/uniform_sweep/final/states` |
| Insertion | `learned_phase_subtask` | 50 | 50 | 48/50 | 0.96 | 249.40 | 355.65 | 1.426x | 0 | 0 | `M-R` | `R/insertion/learned_phase_subtask/search/selected.json` | `R/insertion/learned_phase_subtask/final/states` |
| Insertion | `learned_phase_tabular_rl` | 50 | 50 | 46/50 | 0.92 | 311.67 | 355.65 | 1.141x | 0 | 0 | `M-B` | `B/insertion/learned_phase_tabular_rl/search/selected.json` | `B/insertion/learned_phase_tabular_rl/final/states` |
| Insertion | `learned_phase_rainbow_rl` | 50 | 50 | 44/50 | 0.88 | 222.84 | 355.65 | 1.596x | 0 | 0 | `M-B` | `B/insertion/learned_phase_rainbow_rl/search/selected.json` | `B/insertion/learned_phase_rainbow_rl/final/states` |
| Insertion | `awe_offline_proxy` | 50 | 50 | 50/50 | 1.00 | 319.12 | 355.65 | 1.114x | 0 | 0 | `M-B` | `B/insertion/awe_offline_proxy/search/selected.json` | `B/insertion/awe_offline_proxy/final/states` |
| Insertion | `sail_inspired_adaptive` | 50 | 50 | 49/50 | 0.98 | 342.06 | 355.65 | 1.040x | 0 | 0 | `M-B` | `B/insertion/sail_inspired_adaptive/search/selected.json` | `B/insertion/sail_inspired_adaptive/final/states` |

## Shared matched native references

| Task | Successes/50 | SR | Success-only FSS mean | Per-state evidence |
|---|---:|---:|---:|---|
| Pick | 50/50 | 1.00 | 269.98 | `R/pick/native_1x/final/states` |
| Tea | 50/50 | 1.00 | 441.78 | `R/tea/native_1x/final/states` |
| Insertion | 49/50 | 0.98 | 355.65 | `R/insertion/native_1x/final/states` |

## Gate and audit receipts

- Engineering uniform-1x parity passed at source
  `f7afd5db7b4a910fd8a873a8d42e48f89a5b28bb`: Pick 49/50, Tea 50/50,
  and Insertion 49/50, with zero safety violations and physics errors. The
  parity task identities under
  `/mnt/amlfs-04/home/linke/speedtuning-act-speed-benchmark-v1/attempts/f7afd5db7b4a910fd8a873a8d42e48f89a5b28bb/parity`
  contain the camera order, nominal progress clock, per-step inference, and
  temporal-ensemble receipts. Its run-manifest SHA-256 is
  `198880f243272aa4cdfe2a7bb9378701900157c5a56322f3e0c7f9b73c7077ee`.
- Independent audit workflow:
  `speedtuning-act-speed-audit-20260824-v2-3` (`COMPLETED`, exit 0).
- Audit JSON:
  `/mnt/amlfs-04/home/linke/speedtuning-act-speed-benchmark-v1/reports/17ad53b81bd99cc67fb6403ca1825f6dd430281a/audit.json`
  (SHA-256 `186f496dae00482358eaccf30918c8ccad63941d6ba1e503c7bf03170dca4225`).
  Its rendered `RESULTS.md` SHA-256 is
  `754ef346e257d628c0183d1d08047fdb7aca409ae2cd4b381721c15e7d23d279`.
- The audit validated 900 registered search receipts, 900 untouched method
  final receipts, and 150 shared native receipts. It checked exact seed sets,
  search/final disjointness, identity hashes, manifests, immutable selection
  hashes, completion hashes, detector boundaries, controller receipts, and
  global duplicate absence. Aggregate incidents were zero safety violations
  and one physics error across search, final, and native receipts; the sole
  physics error was the Tea SAIL-inspired search incident noted above.
- The three `learned_phase_subtask` searches preserve their first 20 receipts
  from source `298c6d1` by origin path and SHA-256 and record
  `rollouts_reexecuted: false`; source `866c9f4` ran only the remaining 30.

`awe_offline_proxy` is the preregistered offline proxy and is not full SAIL.
`sail_inspired_adaptive` has a distinct executable preregistration and
provenance and is not claimed as paper-faithful SAIL.
