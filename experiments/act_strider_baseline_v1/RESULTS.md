# Full frozen-ACT speed results with STRIDER

All rows use the same 50 final seeds within each task. Successful episodes are
charged through first success; failures are charged through their terminal
horizon. Thus, throughput includes the cost of failures rather than reporting
speed only on successful episodes.

| Task | Method | Success | SR | Successful-rollout speedup | Throughput delta vs 1x | Safety | Physics |
|---|---|---:|---:|---:|---:|---:|---:|
| Pick | Native 1x | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Pick | Uniform 1.5x | 50/50 | 1.00 | 1.441x | +44.1% | 0 | 0 |
| Pick | Uniform 2x | 50/50 | 1.00 | 1.857x | +85.7% | 0 | 0 |
| Pick | Uniform 2.5x | 48/50 | 0.96 | 2.274x | +115.3% | 0 | 0 |
| Pick | Uniform 3x | 45/50 | 0.90 | 2.712x | +136.0% | 0 | 0 |
| Pick | Uniform sweep | 50/50 | 1.00 | 1.858x | +85.8% | 0 | 0 |
| Pick | Learned subtask | 50/50 | 1.00 | 1.341x | +34.1% | 0 | 0 |
| Pick | Tabular RL | 50/50 | 1.00 | 1.522x | +52.2% | 0 | 0 |
| Pick | Rainbow RL | 50/50 | 1.00 | 1.792x | +79.2% | 0 | 0 |
| Pick | AWE offline proxy | 50/50 | 1.00 | 1.038x | +3.8% | 0 | 0 |
| Pick | SAIL-inspired | 50/50 | 1.00 | 1.044x | +4.4% | 0 | 0 |
| Pick | STRIDER | 48/50 | 0.96 | 2.276x | +115.5% | 0 | 0 |
| Tea | Native 1x | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Tea | Uniform 1.5x | 46/50 | 0.92 | 1.444x | +31.8% | 0 | 0 |
| Tea | Uniform 2x | 38/50 | 0.76 | 1.878x | +40.6% | 0 | 0 |
| Tea | Uniform 2.5x | 19/50 | 0.38 | 2.290x | -14.9% | 0 | 0 |
| Tea | Uniform 3x | 6/50 | 0.12 | 2.744x | -68.1% | 0 | 0 |
| Tea | Uniform sweep | 44/50 | 0.88 | 1.448x | +26.0% | 0 | 0 |
| Tea | Learned subtask | 49/50 | 0.98 | 1.530x | +49.6% | 0 | 0 |
| Tea | Tabular RL | 49/50 | 0.98 | 1.449x | +41.7% | 0 | 0 |
| Tea | Rainbow RL | 46/50 | 0.92 | 1.532x | +40.3% | 0 | 0 |
| Tea | AWE offline proxy | 50/50 | 1.00 | 1.114x | +11.4% | 0 | 0 |
| Tea | SAIL-inspired | 49/50 | 0.98 | 1.138x | +11.1% | 0 | 0 |
| Tea | STRIDER | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Insertion | Native 1x | 49/50 | 0.98 | 1.000x | +0.0% | 0 | 0 |
| Insertion | Uniform 1.5x | 48/50 | 0.96 | 1.445x | +41.5% | 0 | 0 |
| Insertion | Uniform 2x | 39/50 | 0.78 | 1.872x | +47.7% | 0 | 0 |
| Insertion | Uniform 2.5x | 33/50 | 0.66 | 2.287x | +52.9% | 0 | 0 |
| Insertion | Uniform 3x | 31/50 | 0.62 | 2.688x | +69.7% | 0 | 0 |
| Insertion | Uniform sweep | 39/50 | 0.78 | 1.871x | +47.6% | 0 | 0 |
| Insertion | Learned subtask | 48/50 | 0.96 | 1.426x | +39.4% | 0 | 0 |
| Insertion | Tabular RL | 46/50 | 0.92 | 1.141x | +6.8% | 0 | 0 |
| Insertion | Rainbow RL | 44/50 | 0.88 | 1.596x | +41.6% | 0 | 0 |
| Insertion | AWE offline proxy | 50/50 | 1.00 | 1.114x | +14.0% | 0 | 0 |
| Insertion | SAIL-inspired | 49/50 | 0.98 | 1.040x | +4.1% | 0 | 0 |
| Insertion | STRIDER | 39/50 | 0.78 | 1.779x | +38.6% | 0 | 0 |

STRIDER's primary row is its deployable output. If no accelerated candidate
passes the frozen search reliability gate, it reports the native fallback; this
occurred on Tea. The faster rejected Tea candidate remains in the machine-
readable audit as exploratory evidence and does not alter selection.

`AWE offline proxy` and `SAIL-inspired` are preregistered internal proxies, not
paper-faithful implementations of AWE or SAIL. Pick is a disclosed development
task whose proposals were informed by prior discussion. These numbers are
therefore preliminary rather than the final paper benchmark.

## Reliability-throughput frontier

The paper-ready frontier figure is available as
[`figures/reliability_throughput_frontier.pdf`](figures/reliability_throughput_frontier.pdf)
and
[`figures/reliability_throughput_frontier.png`](figures/reliability_throughput_frontier.png).
Green curves mark empirical non-dominated methods within each task; dashed gray
curves connect the independently evaluated fixed-uniform schedules.

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
