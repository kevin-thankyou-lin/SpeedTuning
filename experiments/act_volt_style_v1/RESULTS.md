# Learned-phase VOLT-style final results

This is a frozen-policy proxy for VOLT's binary fast/slow timing abstraction,
not a paper-faithful VOLT retraining reproduction. It used the same fresh search
and final seed banks as STRIDER v2. Native and fixed-uniform final receipts were
shared by exact identity and never rerun.

| Task | Selected schedule | Success | Throughput delta vs. native | Successful-rollout speedup | New final rollouts |
|---|---|---:|---:|---:|---:|
| Pick | `[2.5, 2.0, 2.5, 2.0]` | 48/50 | +94.2% | 2.052x | 50 |
| Tea | Native `[1, 1, 1, 1]` | 50/50 | +0.0% | 1.000x | 0 |
| Insertion | Native `[1, 1, 1, 1]` | 50/50 | +0.0% | 1.000x | 0 |

All final controllers recorded zero safety violations and zero physics errors.
The Pick controller is on its paired empirical reliability-throughput frontier;
Tea and Insertion failed closed to native on their search banks.

## Receipts

- Source: `7dcb2fb0eca47778479856d98dcefc3c1d22f390`
- Search rollouts: `45` Pick, `35` Tea, `30` Insertion (`110` total).
- New final rollouts: `50`; shared final receipts: `750`; shared reruns: `0`.
- Pick `RESULT.json`: `26702c9a57bac577f399f8c8a157042485414be669c6f77787845a2b36e997a4`
- Tea `RESULT.json`: `e5d21127c986acbf2e104278d5ffbecf5f31c0f3fe529d4af7137007b251c087`
- Insertion `RESULT.json`: `67c137e2cd000350abd89ab78cd4136b4cd9ebd7b490aeb150758428fa30cde8`
