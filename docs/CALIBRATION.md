# Calibration log

The world was tuned **only** against ideal observers (Bayes filters with the true model).
No learner was trained and no experiment arm was run while choosing these numbers.

Targets were written first (see `scripts/calibrate_world.py`):

| id | target | why |
|---|---|---|
| C1 | elder's face-only filter accuracy ≥ 0.93 | the other must read you well |
| C2 | ideal self-knower (sensations + interoception) 0.75–0.90 | there is something to learn, and it is learnable |
| C3 | interoception-only ideal 0.50–0.72 | the inner cue is weak but real |
| C4 | sensations-only ideal 0.50–0.75 | the drive also shows in what happens to you |
| C5 | knowing the drive improves next-sensation NLL by ≥ 0.12 nats | self-prediction needs self-knowledge |
| C6 | ideal self-knower separates drive 2 from 3 as well as other pairs | the biased elder's blind spot is not a blind spot of the world |

First try (`a_face 1.6, a_intero 0.45, alpha 0.55`) missed C1 (0.931), C2 (0.717) and C5 (0.094).
A grid over `a_face ∈ {1.8, 2.0}`, `a_intero ∈ {0.45, 0.55, 0.65}`, `alpha ∈ {0.6, 0.7, 0.8}` gave
eight settings meeting all targets. The one nearest the middle of every range was chosen:
`a_face 2.0, a_intero 0.55, alpha 0.70`.

Final values (`results/calibration.json`, 4000 episodes):

| quantity | value |
|---|---:|
| elder, face only | 0.959 |
| ideal self-knower, sensations + interoception | 0.773 |
| interoception only | 0.674 |
| sensations only | 0.679 |
| drive 2 vs 3 (ideal self-knower) | 0.883 |
| drive 0 vs 1 (ideal self-knower) | 0.884 |
| NLL gap, flat belief − oracle drive | 0.146 nats |
