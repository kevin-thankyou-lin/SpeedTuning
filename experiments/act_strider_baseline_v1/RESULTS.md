# Full frozen-ACT speed results with STRIDER

Successful episodes are charged through first success; failures are charged
through their terminal horizon. Thus, throughput includes the cost of failures
rather than reporting speed only on successful episodes.

The filled native/uniform/STRIDER/VOLT-style rows use one fresh paired
50-seed bank per task. The remaining hollow-marker baselines use the earlier
sealed 50-seed bank. Every throughput delta uses the matching native control
from its own bank. Comparisons within either bank are paired; cross-bank points
provide task-distribution context and are not same-seed head-to-head tests.

| Task | Method | Success | SR | Successful-rollout speedup | Throughput delta vs 1x | Safety | Physics |
|---|---|---:|---:|---:|---:|---:|---:|
| Pick | Native 1x | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Pick | Uniform 1.5x | 49/50 | 0.98 | 1.442x | +40.1% | 0 | 0 |
| Pick | Uniform 2x | 47/50 | 0.94 | 1.857x | +70.7% | 0 | 0 |
| Pick | Uniform 2.5x | 46/50 | 0.92 | 2.256x | +102.1% | 0 | 0 |
| Pick | Uniform 3x | 45/50 | 0.90 | 2.679x | +133.4% | 0 | 0 |
| Pick | Learned subtask | 50/50 | 1.00 | 1.341x | +34.1% | 0 | 0 |
| Pick | Tabular RL | 50/50 | 1.00 | 1.522x | +52.2% | 0 | 0 |
| Pick | Rainbow RL | 50/50 | 1.00 | 1.792x | +79.2% | 0 | 0 |
| Pick | AWE offline proxy | 50/50 | 1.00 | 1.038x | +3.8% | 0 | 0 |
| Pick | SAIL-inspired | 50/50 | 1.00 | 1.044x | +4.4% | 0 | 0 |
| Pick | VOLT-style (learned phase) | 48/50 | 0.96 | 2.052x | +94.2% | 0 | 0 |
| Pick | STRIDER | 47/50 | 0.94 | 1.857x | +70.7% | 0 | 0 |
| Tea | Native 1x | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Tea | Uniform 1.5x | 45/50 | 0.90 | 1.450x | +29.3% | 0 | 0 |
| Tea | Uniform 2x | 38/50 | 0.76 | 1.875x | +40.6% | 0 | 0 |
| Tea | Uniform 2.5x | 22/50 | 0.44 | 2.304x | -0.8% | 0 | 0 |
| Tea | Uniform 3x | 6/50 | 0.12 | 2.703x | -68.0% | 0 | 0 |
| Tea | Learned subtask | 49/50 | 0.98 | 1.530x | +49.6% | 0 | 0 |
| Tea | Tabular RL | 49/50 | 0.98 | 1.449x | +41.7% | 0 | 0 |
| Tea | Rainbow RL | 46/50 | 0.92 | 1.532x | +40.3% | 0 | 0 |
| Tea | AWE offline proxy | 50/50 | 1.00 | 1.114x | +11.4% | 0 | 0 |
| Tea | SAIL-inspired | 49/50 | 0.98 | 1.138x | +11.1% | 0 | 0 |
| Tea | VOLT-style (learned phase) | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Tea | STRIDER | 45/50 | 0.90 | 1.450x | +29.3% | 0 | 0 |
| Insertion | Native 1x | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Insertion | Uniform 1.5x | 49/50 | 0.98 | 1.441x | +41.0% | 0 | 0 |
| Insertion | Uniform 2x | 42/50 | 0.84 | 1.868x | +55.5% | 0 | 0 |
| Insertion | Uniform 2.5x | 39/50 | 0.78 | 2.278x | +76.6% | 0 | 0 |
| Insertion | Uniform 3x | 26/50 | 0.52 | 2.683x | +38.5% | 0 | 0 |
| Insertion | Learned subtask | 48/50 | 0.96 | 1.426x | +39.4% | 0 | 0 |
| Insertion | Tabular RL | 46/50 | 0.92 | 1.141x | +6.8% | 0 | 0 |
| Insertion | Rainbow RL | 44/50 | 0.88 | 1.596x | +41.6% | 0 | 0 |
| Insertion | AWE offline proxy | 50/50 | 1.00 | 1.114x | +14.0% | 0 | 0 |
| Insertion | SAIL-inspired | 49/50 | 0.98 | 1.040x | +4.1% | 0 | 0 |
| Insertion | VOLT-style (learned phase) | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Insertion | STRIDER | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |

STRIDER's primary row is its frozen search output. It retained uniform `2x` on
Pick, uniform `1.5x` on Tea, and failed closed to native on Insertion. The
VOLT-style proxy selected `[2.5, 2, 2.5, 2]` on Pick and failed closed to native
on Tea and Insertion. Selection-bank outcomes were not revised after opening
the final banks.

`AWE offline proxy`, `SAIL-inspired`, and `VOLT-style` are preregistered
frozen-policy proxies, not paper-faithful retraining implementations. Pick is a
disclosed development task. These numbers are preliminary rather than the final
paper benchmark.

## Reliability-throughput frontier

The paper-ready frontier figure is available as
[`figures/reliability_throughput_frontier.pdf`](figures/reliability_throughput_frontier.pdf)
and
[`figures/reliability_throughput_frontier.png`](figures/reliability_throughput_frontier.png).
Green curves mark empirical non-dominated methods within each task; dashed gray
curves connect the paired uniform schedules, including native `1x`. The older
`Uniform sweep` rows are intentionally omitted because they are outcomes of
this same controller family on a different bank, not a distinct method. Hollow points come from the
earlier sealed baseline bank and filled points from the fresh STRIDER/VOLT bank.

## Audit receipt

- Audit workflow: `speedtuning-act-strider-table-audit-20260825-v3-1`
- Audit source: `b6a4cd489335d5e25daf3bc7f4f4f241ba61f5dc`
- Machine-readable report SHA-256:
  `fd7bc43c19967fd9f9cb3a4d514034341609197519f5e4a42a97616bff2447da`
- Markdown report SHA-256:
  `9970cf5f06b7e38340eef1b87d321346887b73c07704f6773889a8ae6ae2f0e8`
- Fixed-uniform source:
  `07949ba566f0f1fa68f8dc18cc527a5bbade0160`
- Fixed-uniform accounting: `600` new episodes; `150` native controls
  reused; `0` native reruns.
- Pick/Insertion STRIDER source:
  `0e57760999bc01a1ef021904a83df96ab47c4e46`
- Tea STRIDER source: `3adaa1e986acce7b2f73953218adaa9c9b6a3789`
- Native controls were reused by exact seed; no native rollout was rerun.

## Fresh STRIDER/VOLT audit

- STRIDER source: `525109ba1bb6384ecf487b9dad49942d32e4287a`
- STRIDER final accounting: `750` final rollouts and `110` search rollouts;
  every task evaluated five unique fixed controllers on 50 seeds.
- STRIDER result SHA-256 (Pick):
  `a02dd8b0ec16fafb99a2056a6fe5c479bf291a4fe95a4d60fc8976293a54c9e1`
- STRIDER result SHA-256 (Tea):
  `e260e4aae12644a9c9488d6d0c67b9822114427bbe3af0da14ffbff95e36a8f5`
- STRIDER result SHA-256 (Insertion):
  `1a96b3b768a386cacb58c601fe6844a9753d808a41cbc6f02c02c3abb9df58bf`
- VOLT-style source: `7dcb2fb0eca47778479856d98dcefc3c1d22f390`
- VOLT-style accounting: `110` search rollouts, `50` new final rollouts,
  `750` shared final receipts, and `0` shared final reruns.
- VOLT-style result SHA-256 (Pick):
  `26702c9a57bac577f399f8c8a157042485414be669c6f77787845a2b36e997a4`
- VOLT-style result SHA-256 (Tea):
  `e5d21127c986acbf2e104278d5ffbecf5f31c0f3fe529d4af7137007b251c087`
- VOLT-style result SHA-256 (Insertion):
  `67c137e2cd000350abd89ab78cd4136b4cd9ebd7b490aeb150758428fa30cde8`
