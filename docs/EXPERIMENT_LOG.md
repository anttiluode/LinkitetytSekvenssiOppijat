# Experiment log

## 1. World and calibration

See `docs/CALIBRATION.md`. Tuned against ideal observers only.

## 2. Predictions frozen

`PREDICTIONS.md` committed before any learner code existed.

## 3. Pilot smoke run (not a result)

Pilot seed 1000, **200 steps** instead of 2000, three arms, only to check that everything runs and
to time it. The numbers are recorded because they were seen; they did not change anything.

| arm | train loss first→last 100 | SELF | PAIR23 | INTERO_DROP | SELF with mirror on | seconds |
|---|---|---:|---:|---:|---:|---:|
| ALONE | 1.696 → 1.636 | 0.697 | 0.842 | 0.292 | — | 10.8 |
| FAITHFUL | 1.691 → 1.625 | 0.657 | 0.825 | 0.259 | 0.926 | 11.5 |
| COLEARN | 1.692 → 1.626 | 0.660 | 0.837 | 0.253 | 0.899 | 14.7 |

Ideal self-knower on the same test episodes: 0.783.

Only change after the pilot: a cosmetic `loss.detach()` in the loss log (removes a warning;
no effect on training).

## 4. Frozen run

Runner: `scripts/run_frozen.py`, one process per (seed, arm), two at a time.
Summary: `scripts/summarize.py` → `results/summary.json`, written before the frozen run.

Result (`results/summary.json`): G0 pass; H1–H5 all fail. Infants raised alone reached 0.774
self-knowledge against an ideal of 0.785. Mirror arms slightly behind at every checkpoint.

## 5. Post-hoc v0.1 (after the frozen result)

`POSTHOC.md` written and committed (0c1610f) before its code or any run. Code in `lso/social.py`,
kept separate so v0 stays reproducible. 96 runs: 3 alphas × 8 frozen seeds × 4 arms.
Result (`results/posthoc_summary.json`): PH1 pass, PH2 fail, PH3 fail, PH4 pass.

Figure fix (presentation only): the v0 panel first used a y-axis starting at 0.72, which made
differences of 0.005 look large; it now starts at 0.
