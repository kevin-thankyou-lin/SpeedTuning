# Full fixed-uniform ACT final banks

All registered fixed-uniform schedules were evaluated on all 50 frozen final
seeds for every task. No schedule was selected or stopped early. The existing
same-seed native banks were reused without rerunning any native episode.

Throughput charges successful episodes through first success and failed
episodes through their terminal horizon. Successful-rollout speedup excludes
failed episodes and is therefore always reported beside success rate and
achieved throughput.

| Task | Fixed schedule | Success | SR | Successful-rollout speedup | Throughput delta vs. native 1x | Safety | Physics |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pick | 1.5x | 50/50 | 1.00 | 1.441x | +44.1% | 0 | 0 |
| Pick | 2x | 50/50 | 1.00 | 1.857x | +85.7% | 0 | 0 |
| Pick | 2.5x | 48/50 | 0.96 | 2.274x | +115.3% | 0 | 0 |
| Pick | 3x | 45/50 | 0.90 | 2.712x | +136.0% | 0 | 0 |
| Tea | 1.5x | 46/50 | 0.92 | 1.444x | +31.8% | 0 | 0 |
| Tea | 2x | 38/50 | 0.76 | 1.878x | +40.6% | 0 | 0 |
| Tea | 2.5x | 19/50 | 0.38 | 2.290x | -14.9% | 0 | 0 |
| Tea | 3x | 6/50 | 0.12 | 2.744x | -68.1% | 0 | 0 |
| Insertion | 1.5x | 48/50 | 0.96 | 1.445x | +41.5% | 0 | 0 |
| Insertion | 2x | 39/50 | 0.78 | 1.872x | +47.7% | 0 | 0 |
| Insertion | 2.5x | 33/50 | 0.66 | 2.287x | +52.9% | 0 | 0 |
| Insertion | 3x | 31/50 | 0.62 | 2.688x | +69.7% | 0 | 0 |

## Accounting and provenance

- New fixed-uniform episodes: `3 tasks x 4 schedules x 50 = 600`.
- Native final episodes reused: `150`; native episodes rerun: `0`.
- Execution source: `07949ba566f0f1fa68f8dc18cc527a5bbade0160`.
- Workflows: `speedtuning-act-fixed-uniform-{pick,tea,insertion}-20260825-v1-1`.
- Every workflow completed at retry `0` with exit code `0`.
- Result root:
  `/mnt/amlfs-04/home/linke/speedtuning-act-fixed-uniform-final-v1/runs/07949ba566f0f1fa68f8dc18cc527a5bbade0160`.
- Independent audit workflow:
  `speedtuning-act-strider-table-audit-20260825-v3-1`.
- Audit source: `b6a4cd489335d5e25daf3bc7f4f4f241ba61f5dc`.
- Combined report root:
  `/mnt/amlfs-04/home/linke/speedtuning-act-strider-baseline-v1/reports/b6a4cd489335d5e25daf3bc7f4f4f241ba61f5dc`.
- Combined `RESULTS.json` SHA-256:
  `fd7bc43c19967fd9f9cb3a4d514034341609197519f5e4a42a97616bff2447da`.
- Combined `RESULTS.md` SHA-256:
  `9970cf5f06b7e38340eef1b87d321346887b73c07704f6773889a8ae6ae2f0e8`.

The broader combined report also contains native, selected uniform sweep,
learned-subtask, tabular RL, Rainbow RL, internal AWE/SAIL proxies, and STRIDER
on the same per-task 50-seed final banks. The internal AWE and SAIL rows remain
explicitly labeled as proxies rather than paper-faithful implementations.
