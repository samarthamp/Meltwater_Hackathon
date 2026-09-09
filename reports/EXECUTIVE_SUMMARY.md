# Neural Hangman — Executive Summary

**Brand Buzzword Hackathon** · m.samartha@students.iiit.ac.in

A 28.6M-parameter bidirectional **character Transformer**, trained from scratch on the provided
`train.txt`, supervised by an exact dictionary-posterior oracle, corrected for covariate shift
with **DAgger self-play**, and shipped as a two-member ensemble.

**Public test set: 72.95%** — 182,384 of 250,000 words solved, leaderboard score 72.945.

---

## 1. The two measurements that determined every design decision

| Measurement | Value | Consequence |
|---|---|---|
| `train.txt` ∩ `test.txt` | **0 words** | Memorisation is worthless. Generalisation is the entire task. |
| On-policy game states with **no** consistent training-dictionary word | **45.1%** | Dictionary filtering is silent on nearly half the positions a real game reaches. A generalising model is required, not merely convenient. |

## 2. Results, on 17,530 held-out words disjoint from training

| Method | Win rate |
|---|---:|
| Static letter frequency (what `sample_submission.csv` does) | 14.18% |
| Interpolated character n-gram, noisy-OR pooled | 60.09% |
| **Exact Bayes dictionary posterior + backoff** (supplied the training labels) | **60.97%** |
| Single model `former_a` | 68.98% |
| **Submitted: uniform 2-member ensemble** | **69.61%** |

**The student beats the exact combinatorial oracle that taught it by ~8.6 points.** That is the
central result: a network trained on soft posteriors over millions of states generalises better
than the posterior itself, because the posterior is computed from a finite dictionary and the
network is not.

## 3. What distinguishes this from published solutions

Public solutions to this problem cluster at 42–68% and share three omissions:

1. **They discard the absent-letter set.** Knowing `q`, `x`, `z` are *not* in the word is strong
   evidence about what is. Here both guessed sets are first-class inputs, injected into every
   transformer block by **FiLM** modulation rather than routed through attention.
2. **They train on randomly masked words.** That distribution is unreachable — a live game carries
   at most five absent letters — and its absent sets are dominated by rare letters where a real
   game's are dominated by frequent ones. **DAgger self-play** fixes the mismatch: worth **+1.50
   points** on the transformer and **+5.84** on the weaker recurrent member.
3. **They pool per-position scores with a max.** The correct aggregation is noisy-OR,
   `1 − Π(1 − pᵢ)`. Worth **+8.0 points** on the n-gram baseline and **+3.2** on the model — a
   larger effect than any architectural choice here.

## 4. Things that did not work, reported because they were measured

| Attempted | Result |
|---|---|
| Qwen3-4B teacher, bidirectionalised (LLM2Vec) + LoRA, distilled | **Rejected.** 0.529 top-1 against the oracle's 0.656 and the student's 0.744. A control run with the distillation data *removed* scored higher. |
| Value-guided one-ply lookahead | **−13.9 points.** Even after three DAgger rounds of Monte-Carlo returns, the value head cannot rank actions. |
| Exact information-set search | **Neutral.** Candidate sets drawn from a disjoint dictionary carry no usable joint information. |
| Self-distillation as generation two | **−2.2 points** alone — but a **+0.63** ensemble member (paired McNemar, p = 0.004). |
| NLL-fitted ensemble calibration | **−1.46 points** versus a plain average. Reliable for *rejecting* a member, unreliable for *weighting* one. |

Two of those were my own methodological errors, and both are written up: choosing the
self-distillation gate by **top-1 hit rate of the target** (hard labels have a perfect top-1 and
are still worse to train on), and trusting **NLL-fitted calibration** without checking it tracks
win rate. Same mistake twice — optimising a proxy without verifying it moves with the objective.

## 5. Where the model still fails

76% of losses are words of length ≤ 8. And the losses are narrow: **42.1% of lost games ended with
exactly one letter still hidden** — 11.7 points of win rate. At each wrong guess, a correct letter
was the model's **2nd** choice 49.9% of the time and in its top 3 for 71.6%.

**The model almost always knows the answer; it orders it wrong.** That makes this a ranking
problem, and the highest-value remaining work is sequence-level optimisation of the true
objective (win rate) rather than the myopic per-turn hit rate that greedy argmax maximises.

## 6. On the public leaderboard

At submission the leaderboard showed **94.52**. We measured what produces a score of that shape
rather than speculating:

| Approach | Win rate |
|---|---:|
| Filter `test.txt` itself — the answer **is** in the candidate set | **94.26%** |
| Same, with a survival tie-break | 94.42% |
| **The identical solver on words not in its candidate set** | **2.73%** |

A **91.5-point collapse**. The private re-run is exactly that second condition. For scale, a
*perfect* Bayes posterior over `train.txt` scores 60.97% on unseen words — there is no route from
there to 94.5 using training data alone, because the information is not present.

Our 72.95% is the same number on both distributions. The measurement is reproducible via
`tools/leak_ceiling.py`; neither it nor any diagnostic is imported by the submission path.

## 7. Verification

* **27 unit tests**, including the batched game engine checked against a naive per-game scorer on
  2,000 random words (win flag, strike count, and exact guess string), and the matmul candidate
  filter proven bit-identical to the elementwise predicate it optimises.
* The submission CSV is **re-read and re-scored from scratch** after being written; generator and
  validator agree to the digit.
* The submission notebook was executed end to end against Kaggle's exact directory nesting on the
  full 250,000 words: 6.6 minutes, all cells passing.
* Every hyperparameter was fitted on a **5,000-word slice of the validation dictionary, disjoint
  from the 17,530 words reported on**. `test.txt` was never used to train or tune anything.

---

**Full detail:** `reports/MODEL_REPORT.md` (architecture, layer tables, every decision with its
ablation number) · `reports/RESEARCH_REPORT.md` (literature and what was borrowed) ·
`reports/COMPLIANCE.md` (rule-by-rule, with the greps that verify each claim) ·
`reports/FINETUNING_REPORT.md` (the Qwen stage in full).
