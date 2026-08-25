# Preliminary frozen-ACT speed results with STRIDER

All rows use the same 50 final seeds within each task. Successful episodes are
charged through first success; failures are charged through their terminal
horizon. Thus, throughput includes the cost of failures rather than reporting
speed only on successful episodes.

| Task | Method | Success | SR | Successful-rollout speedup | Throughput delta vs 1x | Safety | Physics |
|---|---|---:|---:|---:|---:|---:|---:|
| Pick | Native 1x | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Pick | Uniform sweep | 50/50 | 1.00 | 1.858x | +85.8% | 0 | 0 |
| Pick | Learned subtask | 50/50 | 1.00 | 1.341x | +34.1% | 0 | 0 |
| Pick | Tabular RL | 50/50 | 1.00 | 1.522x | +52.2% | 0 | 0 |
| Pick | Rainbow RL | 50/50 | 1.00 | 1.792x | +79.2% | 0 | 0 |
| Pick | AWE offline proxy | 50/50 | 1.00 | 1.038x | +3.8% | 0 | 0 |
| Pick | SAIL-inspired | 50/50 | 1.00 | 1.044x | +4.4% | 0 | 0 |
| Pick | STRIDER | 48/50 | 0.96 | 2.276x | +115.5% | 0 | 0 |
| Tea | Native 1x | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Tea | Uniform sweep | 44/50 | 0.88 | 1.448x | +26.0% | 0 | 0 |
| Tea | Learned subtask | 49/50 | 0.98 | 1.530x | +49.6% | 0 | 0 |
| Tea | Tabular RL | 49/50 | 0.98 | 1.449x | +41.7% | 0 | 0 |
| Tea | Rainbow RL | 46/50 | 0.92 | 1.532x | +40.3% | 0 | 0 |
| Tea | AWE offline proxy | 50/50 | 1.00 | 1.114x | +11.4% | 0 | 0 |
| Tea | SAIL-inspired | 49/50 | 0.98 | 1.138x | +11.1% | 0 | 0 |
| Tea | STRIDER | 50/50 | 1.00 | 1.000x | +0.0% | 0 | 0 |
| Insertion | Native 1x | 49/50 | 0.98 | 1.000x | +0.0% | 0 | 0 |
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

## Audit receipt

- Audit workflow: `speedtuning-act-strider-table-audit-20260824-v1-3`
- Audit source: `fb54928c0b780a367e09d3af0ddb985545580e99`
- Machine-readable report SHA-256:
  `fb88049f35b7273293d305f4dcf6d66624abc123ce3bebf0bfd86a3f5c6a875e`
- Markdown report SHA-256:
  `f3dc4c833619efe70fb3f7acff43c3b01f1631813e5f011a609580d4b87d57fe`
- Pick/Insertion STRIDER source:
  `0e57760999bc01a1ef021904a83df96ab47c4e46`
- Tea STRIDER source: `3adaa1e986acce7b2f73953218adaa9c9b6a3789`
- Native controls were reused by exact seed; no native rollout was rerun.
