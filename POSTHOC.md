# Post-hoc experiment v0.1 — the social world

**Status: designed after the frozen v0 result was known.** It is exploratory, not confirmatory.
Its predictions were written and committed before any v0.1 learner was trained.

## Why

Frozen v0 failed every hypothesis, and the reason is visible in its numbers. Infants raised
alone reached 0.774 self-knowledge against an ideal of 0.786. In v0 the drive shapes the infant's
own sensations, so its own prediction task already forces it to learn the drive. The mirror had
nothing left to teach, and was a small crutch (mirror arms slightly slower at every checkpoint).

Two things in v0 did not match the idea being tested:

1. The infant only **used** the elder's display as input. It never had to **predict** how the elder
   would react. The idea is that the self is learned by predicting how others respond to you.
2. The drive mattered for the infant's own sensations. The social-biofeedback idea concerns inner
   states that others react to, whether or not your own senses need them.

## Two changes, nothing else

1. **Social prediction.** The elder no longer shows a display half the time. At each step it
   reacts with probability 0.15 (a glance); when it reacts, the infant sees the display and a flag.
   The infant gets a second head that predicts, from its state at step t, the display the elder
   will show at step t+1, and is trained on it (soft-target cross-entropy, weight 1.0) whenever a
   reaction happens. It still predicts its own next sensation as before.
2. **How much the drive shapes the infant's own sensations**, swept: alpha ∈ {0.0, 0.35, 0.7}.
   At 0.7 the world is v0's. At 0.0 the drive touches only interoception and the face.

Arms: ALONE (no elder, no reactions, no social loss), YOKED (reaction timing and display from a
different episode in the batch), FAITHFUL, BIASED (cannot tell drive 2 from 3). Same frozen seeds,
same 2000 steps, same infant, same probes. Test: **no elder at all.**

## Predictions (written before running)

- **PH1 — the line.** At alpha 0.0, FAITHFUL − ALONE SELF ≥ +0.05, positive in ≥ 6/8 seeds.
  At alpha 0.7, |FAITHFUL − ALONE| < 0.02.
- **PH2 — contingency.** At alpha 0.0, FAITHFUL − YOKED SELF ≥ +0.05 (≥ 6/8), and
  |YOKED − ALONE| < 0.02.
- **PH3 — the inherited blind spot.** At alpha 0.0, FAITHFUL − BIASED PAIR23 ≥ +0.05 (≥ 6/8),
  and |FAITHFUL − BIASED PAIR01| < 0.03.
- **PH4 — monotone.** FAITHFUL − ALONE SELF at alpha 0.35 lies between the values at 0.0 and 0.7.

Claude's guesses: PH1 70 %, PH2 70 %, PH3 60 %, PH4 60 %.

## What is close to built in, and what is not

At alpha 0.0 the only thing that rewards knowing the drive is predicting the elder. So "the
faithful infant learns more about its drive than the alone infant" is close to guaranteed. What is
not guaranteed: that the knowledge **stays with the elder gone** (it has to live in how the infant
reads its own interoception), that it needs **contingency** (the yoked elder reacts just as often),
that the **blind spot transfers specifically** to drive 2 vs 3 and not 0 vs 1, and **where the line
falls** as the drive starts to matter for the infant's own sensations.
