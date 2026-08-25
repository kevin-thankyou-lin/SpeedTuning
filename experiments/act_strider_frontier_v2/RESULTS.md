# STRIDER frontier v2 final results

Each task used a fresh 15-seed search bank followed by an untouched 50-seed
final bank. Failures are charged through the terminal horizon. The final bank
evaluated native `1x` and uniform `1.5x`, `2x`, `2.5x`, and `3x`; the selected
STRIDER controller aliases one of these five schedules on every task.

| Task | Final controller | Success | Throughput delta vs. native | Successful-rollout speedup | Selected by STRIDER |
|---|---|---:|---:|---:|---:|
| Pick | Native `1x` | 50/50 | +0.0% | 1.000x | No |
| Pick | Uniform `1.5x` | 49/50 | +40.1% | 1.442x | No |
| Pick | Uniform `2x` | 47/50 | +70.7% | 1.857x | Yes |
| Pick | Uniform `2.5x` | 46/50 | +102.1% | 2.256x | No |
| Pick | Uniform `3x` | 45/50 | +133.4% | 2.679x | No |
| Tea | Native `1x` | 50/50 | +0.0% | 1.000x | No |
| Tea | Uniform `1.5x` | 45/50 | +29.3% | 1.450x | Yes |
| Tea | Uniform `2x` | 38/50 | +40.6% | 1.875x | No |
| Tea | Uniform `2.5x` | 22/50 | -0.8% | 2.304x | No |
| Tea | Uniform `3x` | 6/50 | -68.0% | 2.703x | No |
| Insertion | Native `1x` | 50/50 | +0.0% | 1.000x | Yes |
| Insertion | Uniform `1.5x` | 49/50 | +41.0% | 1.441x | No |
| Insertion | Uniform `2x` | 42/50 | +55.5% | 1.868x | No |
| Insertion | Uniform `2.5x` | 39/50 | +76.6% | 2.278x | No |
| Insertion | Uniform `3x` | 26/50 | +38.5% | 2.683x | No |

All controllers recorded zero safety violations and zero physics errors.
Selection was sealed before opening the final bank and was not changed after
observing these results. Consequently, the selected Pick controller is
descriptively dominated on its held-out bank; this is preserved rather than
post-hoc retuned.

## Receipts

- Source: `525109ba1bb6384ecf487b9dad49942d32e4287a`
- Search rollouts: `45` Pick, `35` Tea, `30` Insertion (`110` total).
- Final rollouts: `250` per task (`750` total); no schedule-seed reruns.
- Pick `RESULT.json`: `a02dd8b0ec16fafb99a2056a6fe5c479bf291a4fe95a4d60fc8976293a54c9e1`
- Tea `RESULT.json`: `e260e4aae12644a9c9488d6d0c67b9822114427bbe3af0da14ffbff95e36a8f5`
- Insertion `RESULT.json`: `1a96b3b768a386cacb58c601fe6844a9753d808a41cbc6f02c02c3abb9df58bf`

