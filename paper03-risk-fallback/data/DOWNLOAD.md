# Paper 03 — Dataset Acquisition

**No new downloads required.**

Paper 03 replays the scenarios you already converted for Paper 01. That reuse is precisely
why it is scheduled last.

## What you need

| Source | Where it comes from |
|---|---|
| Converted ScenarioNet scenarios | Paper 01, `paper01-coverage-transfer/data/` |
| Trained predictors + CP calibration | Paper 01, `paper01-coverage-transfer/results/` |
| Miscalibrated cross-dataset predictors | Paper 01 transfer matrix — needed for H3 |

## Setup

```
cd F:\CLAUDE\AI1\paper03-risk-fallback\code\metadrive
pip install -e .

cd F:\CLAUDE\AI1\paper03-risk-fallback\code\scenarionet
pip install -e .
```

## Verify headless operation

Confirm rollouts run with rendering DISABLED. Rendering is the only part that wants a
better GPU than yours; headless simulation is CPU-bound and your 18 cores are an advantage.

See `code/scenarionet/documentation/example.rst` for the real API — read that rather than
trusting commands written here.

## Checklist

- [ ] MetaDrive + ScenarioNet installed
- [ ] Headless rollout confirmed working
- [ ] Paper 01 scenarios replay successfully
- [ ] Parallel throughput benchmarked across 18 cores -> notes/sim_throughput.md
- [ ] Rendering tested ONCE, for figures only
