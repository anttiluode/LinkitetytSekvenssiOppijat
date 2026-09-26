# Predictions — frozen before any learner was trained

Written and committed before the first line of learner code. Any later change to the design,
metrics, thresholds or seeds is recorded in the ledger at the bottom, with its reason.

## The question

Can an ordinary sequence learner come to know its own inner state **because another learner
mirrored it**, and does that knowledge stay after the other is gone? And if the other has a blind
spot, does the learner inherit it?

This is the machine version of the social-biofeedback idea (Gergely & Watson 1996): infants learn
to recognise their own states through a caregiver's contingent mirroring. The yoked arm is the
machine version of Murray & Trevarthen's double-video replay.

## Fixed design

**World** (`lso/world.py`, frozen after calibration): hidden drive d ∈ {0,1,2,3}, sticky Markov
(stay 0.96); episode length 64. The infant's sensations s are tokens (6 symbols) whose transitions
depend on d; interoception is a weak cue of d (amplitude 0.55, noise 1) only the infant feels; the
face is a clear cue of d (amplitude 2.0, noise 1) only the elder sees. World seed 0.

**Infant**: GRU, hidden 32. Input per step: one-hot s_t, interoception_t, mirror_t (4), mirror-present
flag (1). Output: logits for s_{t+1}. Trained **only** on its own next-sensation cross-entropy.
Adam, learning rate 3e-3, batch 64 fresh episodes per step, gradient clip 1.0, 2000 development steps.

**Elders never receive gradients from the infant.** Learning is decentralized.

**Arms.** In every arm with a mirror, each development episode has the mirror on with probability 0.5
(flag = 1, mirror shown) or off (flag = 0, zeros).

| arm | mirror when on |
|---|---|
| ALONE | never on |
| YOKED | the faithful elder's display for a *different* episode in the same batch (same statistics, not contingent) |
| FAITHFUL | ideal elder's posterior over this infant's drive from its face, lagged one step |
| BIASED | as FAITHFUL, but the elder cannot tell drive 2 from 3 and shows their average for both |
| COLEARN | an elder GRU (hidden 32) learning at the same time, from its own error only, to predict the infant's next face; it shows tanh of a fixed random 4-D projection of its hidden state, lagged one step |

**Test.** 1024 fresh episodes per seed, **mirror off** for every arm. Hidden states from steps 4–63.
Probes are linear (multinomial logistic regression on standardized features), fitted on the first
half of test episodes and scored on the second half.

| metric | definition |
|---|---|
| SELF | probe accuracy for d_t from the infant's hidden state |
| PAIR23 | binary probe accuracy, drive 2 vs 3, on steps where d ∈ {2,3} |
| PAIR01 | same for 0 vs 1 (control pair) |
| NLL | infant's next-sensation NLL |
| INTERO_DROP | SELF minus SELF when interoception is replaced by signal-free noise at test |
| CURVE | SELF at development steps 250, 500, 1000, 2000 |

Frozen seeds: **3, 17, 29, 41, 59, 73, 97, 113**. Pilot/smoke seed (not a result): 1000.

## Gates

**G0 — the setup works** (otherwise every result is void):
- mean SELF ≥ 0.40 in every arm (chance 0.25);
- FAITHFUL infants read the mirror: SELF with the faithful mirror on at test ≥ 0.85.

**H1 — the mirror becomes inner (primary).** With the mirror off at test:
- FAITHFUL − YOKED SELF ≥ +0.03 on average and positive in ≥ 6/8 seeds, **and**
- FAITHFUL − ALONE SELF ≥ +0.03 on average and positive in ≥ 6/8 seeds.

**H2 — the self inherits the elder's blind spot.**
- FAITHFUL − BIASED PAIR23 ≥ +0.05 on average, positive in ≥ 6/8 seeds, **and**
- ALONE − BIASED PAIR23 ≥ +0.03 on average (the blind elder is worse than no elder on that distinction), **and**
- |FAITHFUL − BIASED PAIR01| < 0.03 on average (the damage is specific to the blind spot).

**H3 — the mirror teaches the infant to read its own insides.**
FAITHFUL INTERO_DROP − ALONE INTERO_DROP ≥ +0.02 on average, positive in ≥ 6/8 seeds.

**H4 — a partner who is learning too is enough.**
COLEARN − YOKED SELF ≥ +0.02 on average, positive in ≥ 6/8 seeds.

**H5 — the mirror speeds development.**
FAITHFUL − ALONE SELF at 500 steps ≥ +0.03 on average.

## What would kill the idea (in this form)

- H1 fails with FAITHFUL ≈ ALONE (|difference| < 0.01): a contingent mirror does not become inner
  in an ordinary sequence learner. The infant uses the mirror while it is there and nothing more.
- H1 passes against ALONE but not against YOKED: any extra input helps; contingency does not matter.
- H2 fails with PAIR23 equal across arms: the self is not partner-shaped here.

## Claude's guesses before running (for calibration of the reader, not gates)

H1 ~45 %, H2 ~35 %, H3 ~40 %, H4 ~30 %, H5 ~55 %. The main reason for doubt: a gradient-trained
learner that has the mirror only half the time may simply use it when present and learn the
self-reading route independently when absent, in which case nothing transfers.

## Ledger of changes after freezing

- Cosmetic: the training-loss log now calls `loss.detach()` (removes a warning; training unchanged).
  Made after the pilot smoke run, before the frozen run.
- Nothing else in the design, metrics, thresholds or seeds changed.
- After the frozen result, a separate post-hoc experiment was designed: see `POSTHOC.md`. It does
  not replace or reinterpret any gate here.
